"""The linear population-mean model, for the genes the flow does NOT predict.

Switch: `model.other_genes` in the config
    source  every non-model gene keeps the source control cell's raw count (default)
    linear  every non-model gene is shifted by this linear model, in log1p(CPM) -- in
            training's cell-eval and in predict.py alike (the model genes stay the flow's)

THE MODEL (per knockdown p, cell line L, gene g; four numbers shared by all of them)
    shift(L, p, g) = a * d(p, g)       p's mean effect on g in OTHER cell lines
                   + b * s(L, p, g)    slope of g on the target gene over L's CONTROL cells,
                                       cov(g, target) / var(target)
                   + c * m(g)          g's generic response: its average shift over all
                                       knockdowns of the other lines
                   + e
    a..e by weighted least squares on the training lines' real mean shifts, every training
    context weighing the same, each seeing only OTHER cell lines (data.pca.cell_line_groups).
    Missing feature = 0.

TABLES (`make prepare` stage 4 when other_genes: linear; cached, holdout-independent)
    <processed_dir>/linear_genes/delta__<ctx>.npz   per knockdown: mean shift over the context's genes
    <processed_dir>/linear_genes/slope__<ctx>.npz   per measured target: slopes over its controls
    Only the TRAINING contexts' mean shifts are ever used for a run -- a held-out line's own
    shifts are never an input to its prediction.

PER RUN (next to the checkpoint): linear_genes.npz -- the coefficients, every training
knockdown's mean effect over all genes (averaged over the training lines) and the generic
response; predict.py computes the slopes from its input controls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

DIR = "linear_genes"
MODES = ("source", "linear")
MIN_CELLS = 20           # cells a knockdown needs for its mean shift to be used
CTRL_CELLS = 20000       # control cells per context (or per input line) for the slopes
BLOCK = 50_000
FIT_CHUNK = 512          # knockdowns per chunk when accumulating the least-squares fit
FEATURES = ("d: other-lines mean effect", "s: slope on target (controls)", "m: generic response", "e: intercept")


def outside_pcs(cfg: dict) -> bool:
    """`model.outside_pcs: linear`: on the MODEL genes, add the part of this linear model's shift
    that lies OUTSIDE the PCA space. A PC-space prediction (mean model, flow) can only express the
    1024 PC patterns, which hold ~35-58% of a knockdown's average effect; the rest is taken from
    here. Held-out VCC25 Validation, 2026-09-30: mean model +0.049 -> +0.096, linear PC formula
    +0.075 -> +0.103 cell-eval average. Needs model.other_genes: linear (the same linear shift)."""
    v = (cfg.get("model") or {}).get("outside_pcs", "none")
    if v not in ("none", "linear", None, False):
        raise SystemExit(f"model.outside_pcs must be 'none' or 'linear', got {v!r}")
    if v == "linear" and mode(cfg) != "linear":
        raise SystemExit("model.outside_pcs: linear needs model.other_genes: linear (it uses the same linear shift)")
    return v == "linear"


def outside_part(shift_model: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """[Gm] the component of a model-gene shift (model-gene order) OUTSIDE the PCs' span:
    s - (s @ L.T) @ L, loadings L [K, Gm] orthonormal rows."""
    s = np.asarray(shift_model, dtype=np.float64)
    L = np.asarray(loadings, dtype=np.float64)
    return (s - (s @ L.T) @ L).astype(np.float32)


def mode(cfg: dict) -> str:
    m = str((cfg.get("model") or {}).get("other_genes", "source"))
    if m not in MODES:
        raise SystemExit(f"model.other_genes must be one of {MODES}, got {m!r}")
    return m


def table_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["processed_dir"]) / DIR


def _fp(c) -> dict:
    st = Path(c._path).stat()
    return {"n_cells": int(len(c.pert)), "size": int(st.st_size), "mtime": int(st.st_mtime)}


# ---- per-context tables (cached) ------------------------------------------------

def mean_shifts(c, cache: Path) -> dict:
    """Per knockdown with >= MIN_CELLS cells: mean log1p(CPM) of its cells minus the control
    mean, over the context's local genes. One sequential pass over the file."""
    f = cache / f"delta__{c.name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if json.loads(str(z["fp"])) == _fp(c):
            return {"perts": z["perts"], "delta": z["delta"]}
    t0 = time.time()
    perts, inv = np.unique(c.pert, return_inverse=True)
    sums = np.zeros((len(perts), len(c.gene_idx)), dtype=np.float64)
    n = len(c.pert)
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        onehot = sp.csr_matrix((np.ones(hi - lo), (inv[lo:hi], np.arange(hi - lo))), shape=(len(perts), hi - lo))
        sums += (onehot @ c.csr(np.arange(lo, hi))).toarray()
    counts = np.bincount(inv, minlength=len(perts))
    means = sums / np.maximum(counts, 1)[:, None]
    keep = np.flatnonzero((counts >= MIN_CELLS) & (perts != 0))
    out = {"perts": perts[keep].astype(np.int32), "delta": (means[keep] - means[0]).astype(np.float32)}
    np.savez(f, **out, fp=json.dumps(_fp(c)))
    print(f"    mean shifts {c.name}: {len(keep):,} knockdowns x {len(c.gene_idx):,} genes  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return out


def slope_matrix(X: np.ndarray, tcols: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """[n_t, n_local] slopes cov(g, t) / var(t) of every column of X (control cells, log1p
    CPM) on each target column, and [n_t] bool: the target varies at all."""
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    Xt = Xt - Xt.mean(0, keepdim=True)
    T = Xt[:, torch.from_numpy(np.asarray(tcols, dtype=np.int64)).to(device)]
    n = max(Xt.shape[0] - 1, 1)
    var_t = (T * T).sum(0) / n
    slope = ((T.T @ Xt) / n / var_t.clamp_min(1e-8)[:, None]).cpu().numpy().astype(np.float32)
    return slope, (var_t > 1e-8).cpu().numpy()


def slopes(c, cache: Path, gene_names, pert_names, device) -> dict:
    """Per knockdown whose target gene the context measures (and varies): the slopes of every
    local gene on the target over the context's control cells."""
    from ..data.hvg import sample_rows

    f = cache / f"slope__{c.name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if json.loads(str(z["fp"])) == _fp(c):
            return {"perts": z["perts"], "slope": z["slope"]}
    t0 = time.time()
    local = {str(gene_names[g]): j for j, g in enumerate(c.gene_idx)}
    perts = np.array([int(p) for p in np.unique(c.pert) if p != 0 and str(pert_names[p]) in local], dtype=np.int32)
    tcols = np.array([local[str(pert_names[p])] for p in perts], dtype=np.int64)
    rows = sample_rows(c.control_rows, CTRL_CELLS, 0)
    slope, ok = slope_matrix(c.dense(rows), tcols, device)
    out = {"perts": perts[ok], "slope": slope[ok]}
    np.savez(f, **out, fp=json.dumps(_fp(c)))
    print(f"    slopes {c.name}: {int(ok.sum()):,} target genes x {len(c.gene_idx):,} genes over "
          f"{len(rows):,} controls  [{time.time() - t0:.0f}s]", flush=True)
    return out


def tables(train: list, contexts: list, cache: Path, gene_names, pert_names, device) -> tuple[dict, dict]:
    """Mean shifts of the TRAINING contexts, slopes of every context (both cached)."""
    cache.mkdir(parents=True, exist_ok=True)
    D = {c.name: mean_shifts(c, cache) for c in train}
    S = {c.name: slopes(c, cache, gene_names, pert_names, device) for c in contexts}
    return D, S


def build_tables(cfg: dict, exclude=(), device: str = "auto") -> None:
    """`make prepare` stage 4: every context's mean shifts and slopes (skipped when cached).
    Holdout-independent; a run only ever uses its training contexts' mean shifts."""
    import pandas as pd

    from ..data.dataset import load_contexts

    if mode(cfg) != "linear" and not (cfg.get("model") or {}).get("anchor"):
        print("  neither model.other_genes: linear nor model.anchor: nothing to do")
        return
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data"]["processed_dir"])
    _, contexts = load_contexts(root, exclude=list(exclude))
    gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    pert_names = pd.read_csv(root / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    tables(contexts, contexts, table_dir(cfg), gene_names, pert_names, device)
    print(f"  linear-model tables up to date in {table_dir(cfg)}")


# ---- features, fit, held-out prediction ---------------------------------------------

def generic_response(L, sources: list, D: dict, G: int) -> np.ndarray:
    """m [n_local(L)]: the average shift of each of L's genes over all knockdowns of `sources`."""
    n_loc = len(L.gene_idx)
    m_sum = np.zeros(n_loc, dtype=np.float64)
    m_cnt = np.zeros(n_loc, dtype=np.float32)
    gmap = np.full(G, -1, dtype=np.int64)
    gmap[L.gene_idx] = np.arange(n_loc)
    for C in sources:
        cols_C = gmap[C.gene_idx]
        have = cols_C >= 0
        m_sum[cols_C[have]] += D[C.name]["delta"][:, have].mean(0)
        m_cnt[cols_C[have]] += 1
    return (m_sum / np.maximum(m_cnt, 1)).astype(np.float32)


def features(L, perts: np.ndarray, sources: list, D: dict, S: dict, G: int, m: np.ndarray | None = None):
    """For context L and knockdowns `perts`: d [n_perts, n_local(L)], s [n_perts, n_local],
    m [n_local], with d and m from the contexts in `sources` (m may be passed in when the
    caller works through `perts` in chunks)."""
    n_loc = len(L.gene_idx)
    pos = {int(p): i for i, p in enumerate(perts)}
    d_sum = np.zeros((len(perts), n_loc), dtype=np.float64)
    d_cnt = np.zeros((len(perts), n_loc), dtype=np.float32)
    if m is None:
        m = generic_response(L, sources, D, G)
    gmap = np.full(G, -1, dtype=np.int64)
    gmap[L.gene_idx] = np.arange(n_loc)
    for C in sources:
        cols_C = gmap[C.gene_idx]
        have = cols_C >= 0
        dC = D[C.name]
        rows_C = [(k, pos[int(p)]) for k, p in enumerate(dC["perts"]) if int(p) in pos]
        if rows_C:
            kC, kL = map(np.array, zip(*rows_C))
            d_sum[np.ix_(kL, cols_C[have])] += dC["delta"][np.ix_(kC, np.flatnonzero(have))]
            d_cnt[np.ix_(kL, cols_C[have])] += 1
    d = (d_sum / np.maximum(d_cnt, 1)).astype(np.float32)
    s = np.zeros((len(perts), n_loc), dtype=np.float32)
    sL = S[L.name]
    at = {int(p): i for i, p in enumerate(sL["perts"])}
    for i, p in enumerate(perts):
        if int(p) in at:
            s[i] = sL["slope"][at[int(p)]]
    return d, s, m


def predict_shift(coef: np.ndarray, d, s, m) -> np.ndarray:
    return coef[0] * d + coef[1] * s + coef[2] * m[None, :] + coef[3]


def fit(train: list, D: dict, S: dict, G: int, line) -> tuple[np.ndarray, dict]:
    """(a, b, c, e) by weighted least squares; each training context weighs the same and sees
    only OTHER cell lines' mean effects (`line`: context name -> cell line)."""
    XtX, Xty = np.zeros((4, 4)), np.zeros(4)
    stats = {}
    for L in train:
        sources = [C for C in train if line(C.name) != line(L.name)]
        dL = D[L.name]
        m = generic_response(L, sources, D, G)
        w = 1.0 / dL["delta"].size
        # knockdown rows are independent of each other, so the normal equations accumulate
        # over row chunks (whole-context arrays reach tens of GB for the Orion contexts)
        for lo in range(0, len(dL["perts"]), FIT_CHUNK):
            d, s, _ = features(L, dL["perts"][lo:lo + FIT_CHUNK], sources, D, S, G, m=m)
            y = dL["delta"][lo:lo + FIT_CHUNK].astype(np.float64)
            F = [d.astype(np.float64), s.astype(np.float64), np.broadcast_to(m, y.shape).astype(np.float64),
                 np.ones_like(y)]
            for i in range(4):
                Xty[i] += w * (F[i] * y).sum()
                for j in range(i, 4):
                    XtX[i, j] += w * (F[i] * F[j]).sum()
                    XtX[j, i] = XtX[i, j]
            del d, s, y, F
        stats[L.name] = {"knockdowns": int(len(dL["perts"])), "genes": int(len(L.gene_idx))}
    return np.linalg.solve(XtX + 1e-9 * np.eye(4), Xty), stats


def heldout_shifts(V, train: list, coef: np.ndarray, D: dict, S: dict, G: int):
    """For every knockdown of held-out context V that a training context carries:
    ({pert: linear shift over V's local genes}, {pert: other-lines mean effect}, perts).
    Every training line counts as "other" for a held-out line."""
    carried = set(int(p) for c in train for p in np.unique(c.pert) if p != 0)
    v_perts = np.array([int(p) for p in np.unique(V.pert) if p != 0 and int(p) in carried], dtype=np.int32)
    d, s, m = features(V, v_perts, train, D, S, G)
    lin = predict_shift(coef, d, s, m)
    return ({int(p): lin[i].astype(np.float32) for i, p in enumerate(v_perts)},
            {int(p): d[i] for i, p in enumerate(v_perts)}, v_perts)


def fit_for_run(cfg: dict, train: list, contexts: list, G: int, device, say=print):
    """Tables (cached), coefficients on `train`. Returns (coef, D, S)."""
    import pandas as pd

    root = Path(cfg["data"]["processed_dir"])
    gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    pert_names = pd.read_csv(root / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    groups = ((cfg.get("data") or {}).get("pca") or {}).get("cell_line_groups") or {}
    D, S = tables(train, contexts, table_dir(cfg), gene_names, pert_names, str(device))
    coef, _ = fit(train, D, S, G, lambda n: groups.get(n, n))
    say("linear model for the non-model genes: " +
        "  ".join(f"{f.split(':')[0]}={c:+.3f}" for f, c in zip(FEATURES, coef)))
    return coef, D, S


# ---- the per-run table and prediction-time shifts --------------------------------------

def save_for_run(run_dir: Path, train: list, D: dict, coef: np.ndarray, G: int, pert_names) -> None:
    """linear_genes.npz next to the checkpoint: coefficients, each training knockdown's mean
    effect over all G genes (averaged over the training lines measuring the gene), and the
    generic response."""
    perts = sorted({int(p) for c in train for p in D[c.name]["perts"]})
    row = {p: i for i, p in enumerate(perts)}
    d_sum = np.zeros((len(perts), G), dtype=np.float32)
    d_cnt = np.zeros((len(perts), G), dtype=np.uint8)
    m_sum = np.zeros(G, dtype=np.float64)
    m_cnt = np.zeros(G, dtype=np.float32)
    for c in train:
        dC = D[c.name]
        rows = np.array([row[int(p)] for p in dC["perts"]], dtype=np.int64)
        d_sum[np.ix_(rows, c.gene_idx)] += dC["delta"]
        d_cnt[np.ix_(rows, c.gene_idx)] += 1
        m_sum[c.gene_idx] += dC["delta"].mean(0)
        m_cnt[c.gene_idx] += 1
    d = (d_sum / np.maximum(d_cnt, 1)).astype(np.float16)
    np.savez(run_dir / "linear_genes.npz", coef=np.asarray(coef, dtype=np.float64),
             pert_names=np.array([str(pert_names[p]) for p in perts]), d=d,
             m=(m_sum / np.maximum(m_cnt, 1)).astype(np.float32))


def input_shifts(tab_path: Path, X_log: sp.csr_matrix, gene_idx: np.ndarray, local_names: list[str],
                 want_perts: list[str], device, seed: int = 0) -> dict[int, np.ndarray]:
    """{position in want_perts: shift over the input line's local genes}, for predict.py: the
    mean effect and generic response from the run's table, the slopes from THIS line's
    unperturbed input cells (`X_log`, log1p CPM)."""
    z = np.load(tab_path)
    coef, m = z["coef"], z["m"][gene_idx].astype(np.float32)
    row = {str(p): i for i, p in enumerate(z["pert_names"])}
    local = {g: j for j, g in enumerate(local_names)}
    has = [j for j, p in enumerate(want_perts) if p in local]
    s_of: dict[int, np.ndarray] = {}
    if has:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(X_log.shape[0], size=min(CTRL_CELLS, X_log.shape[0]), replace=False))
        slope, ok = slope_matrix(np.asarray(X_log[rows].todense(), dtype=np.float32),
                                 np.array([local[want_perts[j]] for j in has]), device)
        s_of = {j: slope[k] for k, j in enumerate(has) if ok[k]}
    out = {}
    zeros = np.zeros(len(gene_idx), dtype=np.float32)
    for j, p in enumerate(want_perts):
        d = z["d"][row[p]][gene_idx].astype(np.float32) if p in row else zeros
        out[j] = (coef[0] * d + coef[1] * s_of.get(j, zeros) + coef[2] * m + coef[3]).astype(np.float32)
    return out
