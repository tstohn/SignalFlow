#!/usr/bin/env python3
"""
"10_..." -- REPLOG rpe1 and VCC25 Training only. Self-contained (the earlier helper
modules this whole script family used to import from -- gene_foldchange_overview.py,
gene_corr_vs_foldchange.py, lfc_metric_consistency.py, etc. -- were lost from disk
between sessions; this script re-implements the small set of primitives it needs
directly, rather than depending on files that vanished once already).

GENE FILTERING (reported as counts, not just applied silently):
    1. keep genes with mean RAW UMI count > 5 in the CONTROL population (not all
       cells -- consistent with every earlier gate in this project, which was always
       control-based). Excluded count is printed and saved.
    2. of the survivors, keep the top 500 by VARIANCE OF RAW COUNTS in the CONTROL
       population (HVG selection in the same raw-count space as the gate above, not
       CPM/log1p -- "variance in control" was not further qualified, and raw is the
       space the gate itself is defined in).

PERTURBATIONS: 100 random perturbations per dataset (independent draws per dataset,
not the same 100 gene names -- REPLOG rpe1 and VCC25 Training have different, mostly
non-overlapping perturbation panels), restricted to perturbations whose target gene
resolves in that dataset's own panel and has >= --min-cells cells.

PER PERTURBATION, per value space (RAW counts, CPM, CPM + log1p(CPM)), over the
500 HVGs (target gene excluded from the 500):
    delta[space]   mean(this perturbation's own cells) - mean(ALL control cells),
                   same space -- PERTURBED minus CONTROL (this script's own
                   convention; note "8_"'s density script used the reverse sign).
    covariance     covariance(gene, target), this perturbation's own cells, same space.
    slope          covariance / variance(target), same cells, same space.
    spearman       Spearman rank correlation(gene, target), same cells, same space.
Three spaces x three metrics = 9 (delta, metric) pairs, always same-space (a raw delta
is only ever compared to a raw-space metric, etc.) -- covariance and slope are
mathematically rank-identical (slope is covariance divided by one positive per-
perturbation scalar), so their two panels/curves are expected to be identical; kept
both because both were explicitly asked for.

TWO OUTPUTS PER (dataset, of the 9 combos):
    a) for 10 of the 100 perturbations (a random subset, illustrative examples): the
       actual scatter (one dot per HVG) -- one 3x3-panel PNG per perturbation (rows =
       space, columns = metric).
    b) for ALL 100: one Spearman r per perturbation (delta vs metric, across the 500
       HVGs) -- a density plot per combo, REPLOG rpe1 and VCC25 Training overlaid.

Output, in images/10_HVG_DELTA_METRIC_DENSITY/:
    gene_gate_summary.csv                     genes total / excluded / HVG-selected, per dataset
    scatter_<dataset>_<perturbation>.png       10 x 2 = 20 illustrative 3x3 grids
    density_<space>_<metric>.png               9 density plots, both datasets overlaid
    per_perturbation.csv                       every (dataset, perturbation, space, metric, r, n_used)
    summary.csv                                median r per dataset x combo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signalflow.data.prepare import _repaired_indptr  # noqa: E402

DATA = ROOT / "data" / "Combined"
OUT = ROOT / "images" / "10_HVG_DELTA_METRIC_DENSITY"
NNZ_PER_CHUNK = 150_000_000
SD_EPS = 1e-8

DATASETS = ("REPLOG__rpe1_raw_singlecell_01", "VCC25__adata_Training")
DATASET_COLOR = {"REPLOG__rpe1_raw_singlecell_01": "#2a78d6", "VCC25__adata_Training": "#eb6834"}
SPACES = ("raw", "cpm", "log1p")
METRICS = ("covariance", "slope", "spearman")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


# ---- reading -----------------------------------------------------------------

def _dec(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def var_names(h: h5py.File) -> np.ndarray:
    var = h["var"]
    return _dec(var[var.attrs["_index"]][:])


def obs_labels(h: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    tg = h["obs/target_gene"]
    if isinstance(tg, h5py.Group):
        labels = _dec(tg["categories"][:])[tg["codes"][:]]
    else:
        labels = _dec(tg[:])
    return labels, h["obs/is_control"][:].astype(bool)


def iter_chunks(h: h5py.File, nnz_budget: int = NNZ_PER_CHUNK):
    X = h["X"]
    n_rows, n_cols = (int(v) for v in X.attrs["shape"])
    indptr = _repaired_indptr(X)
    data, indices = X["data"], X["indices"]
    r = 0
    while r < n_rows:
        r1 = int(np.searchsorted(indptr, indptr[r] + nnz_budget, side="right")) - 1
        r1 = min(max(r1, r + 1), n_rows)
        lo, hi = int(indptr[r]), int(indptr[r1])
        block = sp.csr_matrix((data[lo:hi], indices[lo:hi], indptr[r : r1 + 1] - lo), shape=(r1 - r, n_cols))
        yield r, block
        r = r1


def control_stats(path: Path) -> dict:
    """One streaming pass: control-only mean/var raw counts, mean CPM, mean log1p(CPM)."""
    t0 = time.time()
    with h5py.File(path, "r") as h:
        genes = var_names(h)
        _, is_ctrl = obs_labels(h)
        n_genes = len(genes)
        sum_raw = np.zeros(n_genes, dtype=np.float64)
        sumsq_raw = np.zeros(n_genes, dtype=np.float64)
        sum_cpm = np.zeros(n_genes, dtype=np.float64)
        sum_log1p = np.zeros(n_genes, dtype=np.float64)
        n_ctrl = int(is_ctrl.sum())
        done = 0
        for r0, X in iter_chunks(h):
            m = is_ctrl[r0 : r0 + X.shape[0]]
            if m.any():
                Xc = X[m]
                lib = np.asarray(Xc.sum(axis=1)).ravel().astype(np.float64)
                scale = np.divide(1e6, lib, out=np.zeros_like(lib), where=lib > 0)
                raw_dense = np.asarray(Xc.todense(), dtype=np.float64)
                sum_raw += raw_dense.sum(axis=0)
                sumsq_raw += (raw_dense ** 2).sum(axis=0)
                cpm_dense = raw_dense * scale[:, None]
                sum_cpm += cpm_dense.sum(axis=0)
                sum_log1p += np.log1p(cpm_dense).sum(axis=0)
            done += X.shape[0]
            print(f"    control stats {path.stem}: {done:,}/{len(is_ctrl):,} cells  [{time.time() - t0:.0f}s]",
                  end="\r", flush=True)
    print()
    mean_raw = sum_raw / n_ctrl
    var_raw = sumsq_raw / n_ctrl - mean_raw ** 2
    return {"genes": genes, "n_control": n_ctrl, "mean_raw": mean_raw, "var_raw": np.maximum(var_raw, 0),
            "mean_cpm": sum_cpm / n_ctrl, "mean_log1p": sum_log1p / n_ctrl}


def build_perturbed_shards(path: Path, valid_perts: np.ndarray, min_cells: int) -> dict:
    """One sequential pass, preallocated buffers -- RAW counts per perturbation (full gene width)."""
    t0 = time.time()
    with h5py.File(path, "r") as h:
        genes = var_names(h)
        n_genes = len(genes)
        labels, is_ctrl = obs_labels(h)
        pos = {p: i + 1 for i, p in enumerate(valid_perts)}
        gid = np.full(len(labels), -1, dtype=np.int64)
        keep_pert = ~is_ctrl & np.isin(labels, valid_perts)
        gid[keep_pert] = [pos[p] for p in labels[keep_pert]]
        n_groups = len(valid_perts) + 1

        row_nnz = np.diff(_repaired_indptr(h["X"])).astype(np.int64)
        group_nnz = np.bincount(np.clip(gid, 0, None), weights=np.where(gid >= 0, row_nnz, 0),
                                minlength=n_groups).astype(np.int64)
        group_ncells = np.bincount(np.clip(gid, 0, None), weights=(gid >= 0).astype(np.int64),
                                   minlength=n_groups).astype(np.int64)

        data_buf = [np.empty(int(group_nnz[g]), dtype=np.float32) for g in range(1, n_groups)]
        idx_buf = [np.empty(int(group_nnz[g]), dtype=np.int32) for g in range(1, n_groups)]
        rowptr_buf = [np.zeros(int(group_ncells[g]) + 1, dtype=np.int64) for g in range(1, n_groups)]
        write_pos = np.zeros(n_groups, dtype=np.int64)
        row_cursor = np.zeros(n_groups, dtype=np.int64)

        done, n_rows = 0, len(labels)
        for r0, X in iter_chunks(h):
            g = gid[r0 : r0 + X.shape[0]]
            order = np.argsort(g, kind="stable")
            Xo, go = X[order], g[order]
            bounds = np.flatnonzero(np.diff(go)) + 1
            starts, ends = np.r_[0, bounds], np.r_[bounds, len(go)]
            for s, e in zip(starts, ends):
                gval = int(go[s])
                if gval <= 0:
                    continue
                sub = Xo[s:e]
                i = gval - 1
                nnz = len(sub.data)
                wp = int(write_pos[gval])
                data_buf[i][wp : wp + nnz] = sub.data
                idx_buf[i][wp : wp + nnz] = sub.indices
                write_pos[gval] += nnz
                rc = int(row_cursor[gval])
                lens = np.diff(sub.indptr)
                rowptr_buf[i][rc + 1 : rc + 1 + len(lens)] = rowptr_buf[i][rc] + np.cumsum(lens)
                row_cursor[gval] += len(lens)
            done += X.shape[0]
            print(f"    shards {path.stem}: {done:,}/{n_rows:,} cells  [{time.time() - t0:.0f}s]",
                  end="\r", flush=True)
    print()

    shards: dict[str, sp.csr_matrix] = {}
    for i, p in enumerate(valid_perts):
        gval = i + 1
        n = int(group_ncells[gval])
        if n < min_cells:
            continue
        shards[p] = sp.csr_matrix((data_buf[i], idx_buf[i], rowptr_buf[i]), shape=(n, n_genes))
    return {"genes": genes, "shards": shards}


# ---- per-gene metrics ----------------------------------------------------------

def gene_metrics(dense: np.ndarray, t: int) -> dict:
    """covariance(gene, target), variance(target), Spearman(gene, target) -- all genes at once."""
    mu = dense.mean(axis=0)
    Xc = dense - mu
    cov = (Xc * Xc[:, t : t + 1]).mean(axis=0)
    var_t = float(cov[t])
    slope = cov / max(var_t, SD_EPS)

    ranks = rankdata(dense, method="average", axis=0)
    Rm = ranks - ranks.mean(axis=0, keepdims=True)
    rt = ranks[:, t] - ranks[:, t].mean()
    num = (Rm * rt[:, None]).mean(axis=0)
    den = np.sqrt((Rm ** 2).mean(axis=0) * (rt ** 2).mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        spearman = num / den

    return {"mean": mu, "covariance": cov, "slope": slope, "spearman": spearman}


# ---- plotting ------------------------------------------------------------------

def plot_illustrative(dataset: str, pert: str, panels: dict) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for i, space in enumerate(SPACES):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            key = (space, metric)
            if key not in panels:
                ax.axis("off")
                continue
            x, y, r = panels[key]
            ax.scatter(x, y, s=10, color="#2a78d6", alpha=0.6, linewidths=0)
            ax.axhline(0, color=INK_2, linewidth=0.6)
            ax.axvline(0, color=INK_2, linewidth=0.6)
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{space} | {metric}   [Spearman r={r:+.3f}]", fontsize=9, color=INK)
            if i == 2:
                ax.set_xlabel(f"delta ({space})", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{space} value", fontsize=8.5)
            ax.tick_params(labelsize=7.5, colors=INK_2)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle(f"{dataset}   perturbation: {pert}   (500 HVGs, control-raw-count gated)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / f"scatter_{dataset}_{pert}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_density(space: str, metric: str, values: dict[str, np.ndarray]) -> Path:
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 5.5))
    lo = min(float(np.nanmin(v)) for v in values.values() if len(v))
    hi = max(float(np.nanmax(v)) for v in values.values() if len(v))
    pad = 0.05 * max(hi - lo, 1e-6)
    xs = np.linspace(lo - pad, hi + pad, 400)
    for ds, v in values.items():
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        color = DATASET_COLOR[ds]
        try:
            kde = gaussian_kde(v)
            ax.plot(xs, kde(xs), color=color, linewidth=2, label=f"{ds}  (n={len(v)}, median={np.median(v):+.3f})")
            ax.fill_between(xs, kde(xs), color=color, alpha=0.12)
        except np.linalg.LinAlgError:
            ax.hist(v, bins=20, density=True, color=color, alpha=0.4, label=f"{ds} (n={len(v)})")
        ax.axvline(np.median(v), color=color, linewidth=1, linestyle="--", alpha=0.8)
    ax.axvline(0, color=INK_2, linewidth=0.7)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Spearman r:  delta ({space})  vs  {metric} ({space}), across the 500 HVGs")
    ax.set_ylabel("density (over 100 perturbations)")
    ax.set_title(f"delta ({space}) vs {metric} ({space})   -- per-perturbation Spearman r, REPLOG rpe1 vs "
                 "VCC25 Training", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2)
    fig.tight_layout()
    p = OUT / f"density_{space}_{metric}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---- main pipeline ---------------------------------------------------------------

def process_dataset(name: str, args, rng_global: np.random.Generator) -> tuple[list[dict], dict]:
    path = DATA / f"{name}.h5ad"
    print(f"{name}:")
    cs = control_stats(path)
    genes, n_genes = cs["genes"], len(cs["genes"])
    gate = cs["mean_raw"] > args.min_mean_count
    n_excluded = n_genes - int(gate.sum())
    print(f"  genes: {n_genes:,} total, mean raw count > {args.min_mean_count:g} in control: "
          f"{int(gate.sum()):,} pass, {n_excluded:,} excluded")

    gated_idx = np.flatnonzero(gate)
    order = gated_idx[np.argsort(-cs["var_raw"][gated_idx])]
    hvg_idx = order[: args.n_hvg]
    hvg_mask = np.zeros(n_genes, dtype=bool)
    hvg_mask[hvg_idx] = True
    print(f"  HVG selected (top {args.n_hvg} by control raw-count variance, of the {int(gate.sum()):,} that "
          f"passed the gate): {len(hvg_idx):,}")

    gate_row = {"dataset": name, "n_genes_total": n_genes, "n_excluded_mean_gate": n_excluded,
                "n_pass_mean_gate": int(gate.sum()), "n_hvg_selected": len(hvg_idx)}

    with h5py.File(path, "r") as h:
        labels, is_ctrl = obs_labels(h)
    gpos = {g: i for i, g in enumerate(genes)}
    perts, counts = np.unique(labels[~is_ctrl], return_counts=True)
    valid = np.array([p for p, c in zip(perts, counts) if p in gpos and c >= args.min_cells])
    rng = np.random.default_rng(args.seed)
    n_sel = min(args.n_perturbations, len(valid))
    sel_perts = rng.choice(valid, size=n_sel, replace=False)
    illustrative = set(rng.choice(sel_perts, size=min(args.n_illustrative, n_sel), replace=False).tolist())
    print(f"  {len(valid):,} valid perturbations (target in panel, >= {args.min_cells} cells); "
          f"{n_sel} sampled, {len(illustrative)} illustrative\n")

    sh = build_perturbed_shards(path, sel_perts, args.min_cells)
    ctrl_mean = {"raw": cs["mean_raw"], "cpm": cs["mean_cpm"], "log1p": cs["mean_log1p"]}

    rows = []
    t0 = time.time()
    for k, p in enumerate(sel_perts):
        if p not in sh["shards"]:
            continue
        t = gpos[p]
        dense_raw = sh["shards"][p].toarray().astype(np.float64)
        lib = dense_raw.sum(axis=1)
        scale = np.divide(1e6, lib, out=np.zeros_like(lib), where=lib > 0)
        dense_cpm = dense_raw * scale[:, None]
        dense_log1p = np.log1p(dense_cpm)
        dense_by_space = {"raw": dense_raw, "cpm": dense_cpm, "log1p": dense_log1p}

        ok_genes = hvg_mask.copy()
        ok_genes[t] = False

        panels = {}
        for space in SPACES:
            dense = dense_by_space[space]
            gm = gene_metrics(dense, t)
            delta = gm["mean"] - ctrl_mean[space]
            for metric in METRICS:
                val = gm[metric]
                ok = ok_genes & np.isfinite(delta) & np.isfinite(val)
                n_used = int(ok.sum())
                if n_used < 3:
                    r_val = np.nan
                else:
                    with np.errstate(invalid="ignore"):
                        r_val = float(spearmanr(delta[ok], val[ok])[0])
                rows.append({"dataset": name, "perturbation": p, "space": space, "metric": metric,
                            "spearman_r": r_val, "n_used": n_used, "n_cells": dense.shape[0]})
                if p in illustrative and np.isfinite(r_val):
                    panels[(space, metric)] = (delta[ok], val[ok], r_val)
        if p in illustrative and panels:
            png = plot_illustrative(name, p, panels)
            print(f"    [{k + 1}/{n_sel}] {p}: wrote {png.relative_to(ROOT)}")
        elif (k + 1) % 20 == 0:
            print(f"    [{k + 1}/{n_sel}] processed  [{time.time() - t0:.0f}s]")
    print(f"  [{time.time() - t0:.0f}s total]\n")
    return rows, gate_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-mean-count", type=float, default=5.0, help="gate: control mean raw count must exceed this")
    ap.add_argument("--n-hvg", type=int, default=500)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--n-perturbations", type=int, default=100)
    ap.add_argument("--n-illustrative", type=int, default=10)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    OUT.mkdir(parents=True, exist_ok=True)
    rng_global = np.random.default_rng(args.seed)

    all_rows, gate_rows = [], []
    for name in args.datasets:
        rows, gate_row = process_dataset(name, args, rng_global)
        all_rows += rows
        gate_rows.append(gate_row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "per_perturbation.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT / "gene_gate_summary.csv", index=False)

    print("gene gate / HVG summary:")
    print(pd.DataFrame(gate_rows).to_string(index=False))
    print()

    summary_rows = []
    for space in SPACES:
        for metric in METRICS:
            sub = df[(df.space == space) & (df.metric == metric)]
            values = {ds: sub[sub.dataset == ds]["spearman_r"].to_numpy() for ds in args.datasets}
            png = plot_density(space, metric, values)
            print(f"wrote {png.relative_to(ROOT)}")
            for ds, v in values.items():
                v = v[np.isfinite(v)]
                summary_rows.append({"space": space, "metric": metric, "dataset": ds, "n_perts_used": len(v),
                                     "median_spearman_r": float(np.median(v)) if len(v) else np.nan})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
