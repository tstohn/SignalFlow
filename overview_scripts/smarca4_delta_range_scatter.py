#!/usr/bin/env python3
"""
"11_..." -- 5 reporter genes chosen to SPAN the delta(log1p) range under SMARCA4's own
perturbation (some that shift a lot, some that barely shift), instead of a uniform
random sample or a biggest-shift ranking. Per-cell scatter, CPM + log1p(CPM), control
vs perturbed, same construction as the "10_"/"11_"/"12_" TFAM family.

DATASET: VCC25 Training only. SMARCA4 is in both REPLOG rpe1's and VCC25 Training's
gene panel, but it is only actually a PERTURBATION (has its own knocked-down cells)
in VCC25 Training (1,015 cells) -- REPLOG rpe1 has 0 SMARCA4-perturbed cells, so
there is nothing to plot there.

GENE POOL: the same 500-HVG set hvg_delta_metric_density.py ("10_") built for VCC25
Training (control raw-count mean > 5 gate, then top 500 by control raw-count
variance) -- reused here via that script's own functions, not recomputed differently.

STRATIFIED SELECTION: delta(log1p) = mean log1p(CPM), SMARCA4's own cells, minus the
full-dataset control mean log1p(CPM), computed for every HVG (SMARCA4 itself
excluded). The 500 HVGs are ranked by this delta and split into 5 equal-sized bins
(quintiles of RANK, not of value, so each bin has ~100 genes regardless of how the
values cluster); one gene is drawn at random from each bin -- this is what "covers
the whole range" means here: one from the biggest-downshift group, one from
smallest-shift-near-zero, one from the biggest-upshift group, and two in between.

Output, in images/11_SMARCA4_DELTA_RANGE_SCATTER/:
    scatter_VCC25__adata_Training_SMARCA4_vs_<reporter>.png   (5 plots)
    selected_genes.csv   the 5 genes, their bin, and their delta(log1p)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from hvg_delta_metric_density import DATA, GRID, INK, build_perturbed_shards, control_stats, obs_labels, var_names  # noqa: E402

OUT = ROOT / "images" / "11_SMARCA4_DELTA_RANGE_SCATTER"
DATASET = "VCC25__adata_Training"
TARGET = "SMARCA4"
CTRL_COLOR, PERT_COLOR = "#2a78d6", "#e34948"


def read_rows(h: h5py.File, rows: np.ndarray):
    import scipy.sparse as sp

    from signalflow.data.prepare import _repaired_indptr

    X = h["X"]
    n_cols = int(X.attrs["shape"][1])
    indptr = _repaired_indptr(X)
    data, indices = X["data"], X["indices"]
    uniq, inv = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    brk = np.flatnonzero(np.diff(uniq) != 1) + 1
    starts, ends = np.r_[0, brk], np.r_[brk, len(uniq)]
    ds, ix = [], []
    for s, e in zip(starts, ends):
        lo, hi = int(indptr[uniq[s]]), int(indptr[uniq[e - 1] + 1])
        ds.append(data[lo:hi])
        ix.append(indices[lo:hi])
    lens = indptr[uniq + 1] - indptr[uniq]
    ptr = np.r_[0, np.cumsum(lens)]
    M = sp.csr_matrix((np.concatenate(ds), np.concatenate(ix), ptr), shape=(len(uniq), n_cols))
    return M[inv]


def pick_stratified(delta: np.ndarray, gene_idx: np.ndarray, n_bins: int, seed: int) -> np.ndarray:
    order = gene_idx[np.argsort(delta[gene_idx])]
    bins = np.array_split(order, n_bins)
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(b) for b in bins if len(b)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-mean-count", type=float, default=5.0)
    ap.add_argument("--n-hvg", type=int, default=500)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--n-reporters", type=int, default=5)
    ap.add_argument("--control-cells", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()
    DATASET_, TARGET_ = args.dataset, args.target

    import matplotlib
    matplotlib.use("Agg")

    OUT.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{DATASET_}.h5ad"

    print(f"{DATASET_}:")
    cs = control_stats(path)
    genes, n_genes = cs["genes"], len(cs["genes"])
    gpos = {g: i for i, g in enumerate(genes)}
    gate = cs["mean_raw"] > args.min_mean_count
    gated_idx = np.flatnonzero(gate)
    order = gated_idx[np.argsort(-cs["var_raw"][gated_idx])]
    hvg_idx = order[: args.n_hvg]
    print(f"  {int(gate.sum()):,}/{n_genes:,} genes pass the control mean-raw-count > {args.min_mean_count:g} "
          f"gate; {len(hvg_idx):,} HVGs selected (same as '10_')")

    t = gpos[TARGET_]
    sh = build_perturbed_shards(path, np.array([TARGET_]), args.min_cells)
    dense_raw = sh["shards"][TARGET_].toarray().astype(np.float64)
    lib = dense_raw.sum(axis=1)
    scale = np.divide(1e6, lib, out=np.zeros_like(lib), where=lib > 0)
    dense_log1p = np.log1p(dense_raw * scale[:, None])
    pert_mean_log1p = dense_log1p.mean(axis=0)
    delta_log1p = pert_mean_log1p - cs["mean_log1p"]
    print(f"  {TARGET_}: {dense_raw.shape[0]:,} perturbed cells")

    pool = hvg_idx[hvg_idx != t]
    sel_idx = pick_stratified(delta_log1p, pool, args.n_reporters, args.seed)
    sel_genes = genes[sel_idx]
    print(f"  stratified picks (delta(log1p) low -> high): "
          f"{list(zip(sel_genes.tolist(), np.round(delta_log1p[sel_idx], 3).tolist()))}\n")

    with h5py.File(path, "r") as h:
        labels, is_ctrl = obs_labels(h)
        ctrl_rows = np.flatnonzero(is_ctrl)
        pert_rows = np.flatnonzero(labels == TARGET_)
        rng = np.random.default_rng(args.seed)
        n_c = min(args.control_cells, len(ctrl_rows))
        sample_ctrl = np.sort(rng.choice(ctrl_rows, n_c, replace=False))
        rows = np.concatenate([sample_ctrl, np.sort(pert_rows)])
        is_pert_cell = np.r_[np.zeros(len(sample_ctrl), dtype=bool), np.ones(len(pert_rows), dtype=bool)]
        cols = [t] + sel_idx.tolist()
        X = read_rows(h, rows)
    lib2 = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    scale2 = np.divide(1e6, lib2, out=np.zeros_like(lib2), where=lib2 > 0)
    block = np.asarray(X[:, cols].todense()).astype(np.float64) * scale2[:, None]
    log1p_block = np.log1p(block)
    x = log1p_block[:, 0]

    rows_out = []
    for j, (g, gi) in enumerate(zip(sel_genes, sel_idx), start=1):
        y = log1p_block[:, j]
        png = plot_scatter(DATASET_, TARGET_, g, x, y, is_pert_cell, delta_log1p[gi])
        rows_out.append({"dataset": DATASET_, "target": TARGET_, "reporter": g,
                         "delta_log1p_under_target": float(delta_log1p[gi]),
                         "n_control": int((~is_pert_cell).sum()), "n_perturbed": int(is_pert_cell.sum()),
                         "figure": png.name})
        print(f"  wrote {png.relative_to(ROOT)}")

    df = pd.DataFrame(rows_out)
    csv_path = OUT / f"selected_genes_{DATASET_}_{TARGET_}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(ROOT)}")
    print(df.to_string(index=False))


def plot_scatter(dataset: str, target: str, reporter: str, x: np.ndarray, y: np.ndarray, is_pert: np.ndarray,
                 delta: float) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(x[~is_pert], y[~is_pert], color=CTRL_COLOR, s=14, linewidths=0, alpha=0.7, zorder=3,
              label=f"control ({int((~is_pert).sum()):,} cells)")
    ax.scatter(x[is_pert], y[is_pert], color=PERT_COLOR, s=14, linewidths=0, alpha=0.7, zorder=4,
              label=f"{target} perturbed ({int(is_pert.sum()):,} cells)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"{target} expression, CPM + log1p (this cell)")
    ax.set_ylabel(f"{reporter} expression, CPM + log1p (this cell)")
    ax.set_title(f"{dataset}\n{target} (perturbed, x) vs {reporter} (delta(log1p)={delta:+.3f} under {target}, y)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="best")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    p = OUT / f"scatter_{dataset}_{target}_vs_{reporter}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


if __name__ == "__main__":
    main()
