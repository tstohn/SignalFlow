"""Per-dataset highly variable genes (HVGs) and the per-dataset OT PCA.

The OT space is built from ONE dataset's own cells, never a basis shared across
datasets: HVGs over ALL cells of the dataset (control + perturbed), each HVG optionally
z-scored within the dataset (clipped at +-10), then PCA. Control -> perturbed optimal
transport runs in these PC scores (data/couple.py::build_per_dataset). The scores are
NOT whitened, so a noise PC weighs as little in the OT cost as it explains.

Everything works from per-gene running sums over row chunks, so a disk-backed context
of millions of cells never has to be dense in memory.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

CHUNK = 2048
N_BINS = 20             # mean bins of the "seurat" dispersion flavor (scanpy's default)
CLIP = 10.0             # max |z| after per-gene scaling (scanpy's `scale(max_value=10)`)

COUPLING_DEFAULTS = {
    "space": "dataset_hvg_pca",
    "n_hvg": 2000,
    "n_pcs": 100,
    "hvg_flavor": "seurat",
    "scale_genes": True,
    "n_sources": 10000,
    "max_fit_cells": 100000,
}
SPACES = ("dataset_hvg_pca",)
FLAVORS = ("seurat", "variance")


def coupling_cfg(cfg: dict) -> dict:
    out = {**COUPLING_DEFAULTS, **(cfg.get("data", {}).get("coupling") or {})}
    if out["space"] not in SPACES:
        raise SystemExit(f"data.coupling.space must be one of {SPACES}, got {out['space']!r}")
    if out["hvg_flavor"] not in FLAVORS:
        raise SystemExit(f"data.coupling.hvg_flavor must be one of {FLAVORS}, got {out['hvg_flavor']!r}")
    return out


# ---- which rows to fit on ---------------------------------------------------

def sample_rows(rows: np.ndarray, max_cells: int, seed: int, block: int = 1024) -> np.ndarray:
    """Sorted subset of `rows` of about `max_cells`, made of whole runs of `block`
    consecutive entries of `rows`, so a disk-backed read stays a handful of large reads."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if not max_cells or len(rows) <= max_cells:
        return rows
    starts = np.arange(0, len(rows), block)
    k = max(1, int(np.ceil(max_cells / block)))
    pick = np.sort(np.random.default_rng(seed).choice(len(starts), min(k, len(starts)), replace=False))
    return np.concatenate([rows[starts[i] : starts[i] + block] for i in pick])


# ---- per-gene statistics and HVG selection ----------------------------------

def gene_stats(read: Callable[[np.ndarray], np.ndarray], rows: np.ndarray, n_local: int) -> dict:
    """Per-gene sums over `rows`, read `CHUNK` at a time: `read(rows_chunk)` returns the
    dense lognorm block [len(chunk), n_local]."""
    s1 = np.zeros(n_local, np.float64)
    s2 = np.zeros(n_local, np.float64)
    e1 = np.zeros(n_local, np.float64)
    e2 = np.zeros(n_local, np.float64)
    for i in range(0, len(rows), CHUNK):
        X = read(rows[i : i + CHUNK]).astype(np.float64, copy=False)
        s1 += X.sum(0)
        s2 += (X * X).sum(0)
        E = np.expm1(X)
        e1 += E.sum(0)
        e2 += (E * E).sum(0)
    return {"n": len(rows), "s1": s1, "s2": s2, "e1": e1, "e2": e2}


def _var(sq: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    mean = s / max(n, 1)
    return np.maximum(sq / max(n, 1) - mean * mean, 0.0) * (n / max(n - 1, 1))


def select_hvg(stats: dict, n_top: int, flavor: str) -> np.ndarray:
    """Sorted LOCAL indices of the `n_top` most variable genes; every gene if there are
    no more than `n_top`. Genes that never vary in the sample are never picked.

    seurat:   scanpy's flavor="seurat" -- dispersion var/mean of the CPM values,
              log, then z-scored within 20 equal-width bins of log1p(mean), so a gene
              is variable relative to genes of similar expression.
    variance: plain variance of the lognorm values.
    """
    n, n_local = stats["n"], len(stats["s1"])
    if n_local <= n_top:
        return np.arange(n_local, dtype=np.int64)
    lvar = _var(stats["s2"], stats["s1"], n)
    if flavor == "variance":
        score = lvar.copy()
    else:
        mean = stats["e1"] / max(n, 1)
        var = _var(stats["e2"], stats["e1"], n)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.log(var / np.where(mean > 0, mean, 1e-12))
        disp[~np.isfinite(disp)] = -np.inf
        lmean = np.log1p(mean)
        edges = np.linspace(lmean.min(), lmean.max(), N_BINS + 1)
        b = np.clip(np.digitize(lmean, edges[1:-1]), 0, N_BINS - 1)
        score = np.full(n_local, -np.inf)
        for k in range(N_BINS):
            at = (b == k) & np.isfinite(disp)
            if not at.any():
                continue
            d = disp[at]
            sd = d.std(ddof=1) if at.sum() > 1 else 0.0
            # a gene alone in its bin (or a bin without spread) has no reference: scanpy scores it 1
            score[at] = (d - d.mean()) / sd if sd > 0 else 1.0
    score[lvar <= 0] = -np.inf
    top = np.argsort(-score, kind="stable")[:n_top]
    top = top[np.isfinite(score[top])]
    return np.sort(top).astype(np.int64)


# ---- the per-dataset OT PCA -------------------------------------------------

def fit_ot_pca(read: Callable[[np.ndarray], np.ndarray], rows: np.ndarray, hvg: np.ndarray,
               stats: dict, n_pcs: int, scale: bool) -> dict:
    """PCA of the HVG columns over `rows`. Genes are centered (and, with `scale`, divided
    by their std and clipped at +-CLIP) using `stats`, the sample the HVGs came from.
    Exact eigendecomposition of the [n_hvg x n_hvg] covariance, accumulated in chunks."""
    n = stats["n"]
    mu = (stats["s1"][hvg] / max(n, 1)).astype(np.float64)
    sd = np.sqrt(_var(stats["s2"][hvg], stats["s1"][hvg], n)) if scale else np.ones(len(hvg))
    sd = np.where(sd < 1e-8, 1.0, sd)
    h = len(hvg)
    s = np.zeros(h, np.float64)
    ss = np.zeros((h, h), np.float64)
    for i in range(0, len(rows), CHUNK):
        Z = _standardize(read(rows[i : i + CHUNK])[:, hvg], mu, sd, scale)
        s += Z.sum(0)
        ss += Z.T @ Z
    m = s / max(len(rows), 1)
    cov = (ss - len(rows) * np.outer(m, m)) / max(len(rows) - 1, 1)
    w, V = np.linalg.eigh(cov)
    k = min(int(n_pcs), h)
    order = np.argsort(w)[::-1][:k]
    frac = float(np.maximum(w[order], 0).sum() / max(float(np.trace(cov)), 1e-12))
    return {"hvg": hvg, "mu": mu, "sd": sd, "m": m, "V": V[:, order], "var": w[order],
            "var_frac": frac, "scale": bool(scale)}


def _standardize(X: np.ndarray, mu, sd, scale: bool) -> np.ndarray:
    Z = (X.astype(np.float64) - mu) / sd
    return np.clip(Z, -CLIP, CLIP) if scale else Z


def project_ot(read: Callable[[np.ndarray], np.ndarray], n_rows: int, pca: dict) -> np.ndarray:
    """[n_rows, k] PC scores of every row, read `CHUNK` rows at a time."""
    out = np.empty((n_rows, pca["V"].shape[1]), dtype=np.float32)
    for i in range(0, n_rows, CHUNK):
        rows = np.arange(i, min(i + CHUNK, n_rows))
        Z = _standardize(read(rows)[:, pca["hvg"]], pca["mu"], pca["sd"], pca["scale"])
        out[i : i + len(rows)] = ((Z - pca["m"]) @ pca["V"]).astype(np.float32)
    return out
