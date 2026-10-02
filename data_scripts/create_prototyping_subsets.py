#!/usr/bin/env python3
"""
Carve a tiny, fast-to-load prototyping subset out of each per-dataset h5ad
in data/Combined/ (the output of combine_datasets.py).

Every *.h5ad found directly under data/Combined/ is processed -- no
hardcoded file list, since combine_datasets.py already produced one file
per source dataset there (e.g. REPLOG__K562_essential_raw_singlecell_01.h5ad,
VCC25__adata_Training.h5ad, ...). Each of those files already has a
standardized `target_gene` / `context` / `is_control` obs schema and a
VCC26-filtered, symbol-indexed var axis (see combine_datasets.py), so this
script only has to do the subsetting -- no more Ensembl-vs-symbol handling.

For each Combined file, this script:
  1. Randomly picks N_PERTURBATIONS distinct perturbations (target genes).
  2. For each of those perturbations, draws a random cell count in
     [CELLS_MIN, CELLS_MAX] and samples that many cells (or all of them if
     fewer are available). Sizes are randomized *per group* on purpose --
     real perturbation datasets have wildly uneven group sizes, and a model
     trained only on evenly-sized batches won't have learned to handle that.
  3. Does the same for the unperturbed/control ("non-targeting") cells, as
     one extra group.
  4. Builds the gene panel from two sources, unioned:
       a. A VCC26 reference HVG set, computed ONCE up front from
          data/VCC26/controls/context_{A,B,C}.h5ad: the top
          VCC26_HVG_PER_CONTEXT (500) most-variable genes are ranked
          independently *within each context* (library-size normalized +
          log1p, ranked by variance), then the three 500-gene lists are
          unioned and deduplicated -- up to 1,500 genes, fewer if a gene
          lands in more than one context's top 500. This set is the SAME
          across every Combined file (it doesn't depend on the file being
          processed).
       b. That Combined file's OWN top N_TOP_VARIABLE_GENES_PER_DATASET
          (50) genes, ranked the same way but on that file's own
          already-subsampled control cells -- i.e. per-dataset, not global.
     The final panel for a given file = (the VCC26 set, restricted to genes
     that are actually present in that file's own var) UNION (that file's
     own top 50). Panel size therefore varies per file, since how much of
     the VCC26 set a file actually has depends on that file's own gene
     coverage (e.g. REPLOG's ~7.7-9k-gene panels vs VCC25's ~18k).
  5. Writes each file's subset to data/Prototype/ as its own small h5ad --
     one output file per Combined input file, kept as distinct datasets
     (same "separate files, not one union" convention as Combined/ itself).

Not run automatically -- draft for you to fine-tune before executing.

Requires: anndata, pandas, numpy, scipy, h5py  (pip install anndata pandas numpy scipy h5py)

Usage:
    python data_scripts/create_prototyping_subsets.py [--combined-dir DIR] [--output-dir OUT_DIR] [--seed N]
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import anndata as ad
    from anndata._core.sparse_dataset import SparseDataset
    from anndata.experimental import read_elem
except ImportError:
    sys.exit(
        "This script requires the 'anndata' package.\n"
        "Install it first: pip install anndata"
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine_datasets import (  # noqa: E402
    CONTROL_LABEL,
    CPM_TARGET_SUM,
    compute_lognorm_layer,
    resolve_perturbation_column,
)

# ----------------------------------------------------------------------------
# CONFIG -- tune freely
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

COMBINED_DIR = DATA_ROOT / "Combined"   # every *.h5ad directly in here is processed
VCC26_DIR = DATA_ROOT / "VCC26" / "controls"

N_PERTURBATIONS = 10           # distinct perturbations to sample per file
CELLS_MIN = 500                # min cells per group (perturbed or control)
CELLS_MAX = 1000               # max cells per group (inclusive)

VCC26_HVG_PER_CONTEXT = 500              # top HVGs taken independently from EACH of context_A/B/C
N_TOP_VARIABLE_GENES_PER_DATASET = 50    # each Combined file's own top HVGs, on top of the VCC26 set

RANDOM_SEED = 0

OUTPUT_DIR = DATA_ROOT / "Prototype"
OUTPUT_STATS_CSV = OUTPUT_DIR / "cell_subset_stats.csv"
OUTPUT_GENES_CSV = OUTPUT_DIR / "selected_genes.csv"

# ----------------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------------


def _decode_bytes(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x


def _read_index(group: h5py.Group, index_name: str):
    """VCC26's own files encode obs/var _index as a 'nullable-string-array'
    group (values + mask) -- an encoding this project's anndata (0.9.2)
    doesn't know how to decode via read_elem (added in a later anndata
    version than Python 3.8 allows here). Same issue/fix as
    overview scripts/umap_all_datasets.py's read_dataframe()."""
    item = group[index_name]
    if isinstance(item, h5py.Group) and item.attrs.get("encoding-type") == "nullable-string-array":
        values = [_decode_bytes(v) for v in item["values"][:]]
        if "mask" in item:
            mask = item["mask"][:]
            values = [None if m else v for v, m in zip(values, mask)]
        return values
    return [_decode_bytes(v) for v in item[:]]


def read_dataframe(group: h5py.Group) -> pd.DataFrame:
    """Like anndata.experimental.read_elem(group) for an obs/var dataframe,
    but with the index read via `_read_index` above -- everything else
    (each column) still goes through the normal read_elem."""
    index_name = group.attrs.get("_index", "_index")
    index = pd.Index(_read_index(group, index_name))
    col_order = [_decode_bytes(c) for c in group.attrs.get("column-order", [])]
    data = {col: read_elem(group[col]) for col in col_order}
    return pd.DataFrame(data, index=index)


def load_vcc26_context(path: Path) -> "ad.AnnData":
    """VCC26's context_{A,B,C}.h5ad are small (~200MB each, ~18.4k cells --
    all control cells, per data/VCC26/controls/manifest.json), so loaded in
    full rather than backed/chunked."""
    with h5py.File(path, "r") as f:
        obs = read_dataframe(f["obs"])
        var = read_dataframe(f["var"])
        X = read_elem(f["X"])
    return ad.AnnData(X=X, obs=obs, var=var)


def rank_genes_by_variance(X, var_names: pd.Index) -> pd.Index:
    """Library-size normalize + log1p, rank genes by variance descending.
    Shared by the VCC26 HVG computation and the per-dataset top-N below --
    same methodology both places."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    lib_size = X.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    norm = np.log1p(X / lib_size * np.median(lib_size))
    variance = norm.var(axis=0)
    order = np.argsort(variance)[::-1]
    return var_names.to_numpy()[order]


def compute_vcc26_hvg_set(vcc26_dir: Path, n_per_context: int) -> set:
    """Top `n_per_context` HVGs from EACH of context_A/B/C independently,
    then unioned and deduplicated (a gene landing in more than one
    context's top list is only counted once) -- per the brief, not a
    single ranking pooled across all three contexts together."""
    paths = sorted(vcc26_dir.glob("context_*.h5ad"))
    if not paths:
        sys.exit(f"No VCC26 context_*.h5ad files found in {vcc26_dir}.")
    hvg_genes = set()
    for path in paths:
        adata = load_vcc26_context(path)
        ranked = rank_genes_by_variance(adata.X, adata.var_names)
        top_genes = set(ranked[:n_per_context])
        print(f"  {path.name}: top {len(top_genes)} HVGs from {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        hvg_genes |= top_genes
    print(f"  VCC26 HVG set: {len(hvg_genes)} unique genes "
          f"(union of 3 x top-{n_per_context}, deduplicated)")
    return hvg_genes


def sample_cells(path: Path, n_perturbations: int, cells_min: int, cells_max: int, rng: np.random.Generator):
    # ad.read_h5ad(path, backed="r") keeps X lazy but eagerly loads every
    # other slot -- including `layers` -- into RAM via anndata's internal
    # read_elem(). Some of the Combined files carry a full lognorm layer
    # alongside raw X (e.g. the REPLOG gwps file is ~146GB total, with a
    # ~89GB layers/lognorm copy), so backed="r" still tries to pull tens of
    # GB into memory before we ever get to subsetting. We don't use
    # `layers` here, so read obs/var directly and keep X as a lazy
    # SparseDataset, row-indexing it only for the cells we actually sample.
    f = h5py.File(path, "r")
    obs = read_elem(f["obs"])
    var = read_elem(f["var"])
    X_backed = SparseDataset(f["X"])

    pert_col = resolve_perturbation_column(SimpleNamespace(obs=obs))
    labels = obs[pert_col].astype(str).to_numpy()
    is_control = labels == CONTROL_LABEL

    unique_perts = pd.unique(labels[~is_control])
    n_select = min(n_perturbations, len(unique_perts))
    selected_perts = rng.choice(unique_perts, size=n_select, replace=False)

    row_groups = []
    group_stats = []
    for pert in selected_perts:
        avail_idx = np.flatnonzero(labels == pert)
        n_target = int(rng.integers(cells_min, cells_max + 1))
        n_take = min(n_target, avail_idx.size)
        chosen = rng.choice(avail_idx, size=n_take, replace=False)
        row_groups.append(chosen)
        group_stats.append(dict(
            source_file=path.name, group=pert, is_control=False,
            n_cells_available=int(avail_idx.size), n_cells_target=n_target, n_cells_sampled=n_take,
        ))

    control_idx = np.flatnonzero(is_control)
    n_target_ctrl = int(rng.integers(cells_min, cells_max + 1))
    n_take_ctrl = min(n_target_ctrl, control_idx.size)
    chosen_ctrl = rng.choice(control_idx, size=n_take_ctrl, replace=False)
    row_groups.append(chosen_ctrl)
    group_stats.append(dict(
        source_file=path.name, group=CONTROL_LABEL, is_control=True,
        n_cells_available=int(control_idx.size), n_cells_target=n_target_ctrl, n_cells_sampled=n_take_ctrl,
    ))

    row_idx = np.sort(np.concatenate(row_groups))
    # Combined files already carry target_gene/context/is_control/etc --
    # they just come along for free with the row subset, nothing to rebuild.
    # `layers["lognorm"]` is deliberately NOT copied from the source file --
    # it's CPM-normalized against each cell's full ~7-20k-gene library size,
    # which no longer applies once we drop to the final gene panel. Lognorm
    # is instead recomputed from scratch on the final gene-reduced subset
    # (see main()), which gives different (smaller-library-size) values on
    # purpose -- that's the correct normalization for the subset as shipped.
    X_sub = X_backed[row_idx, :]
    sub = ad.AnnData(X=X_sub, obs=obs.iloc[row_idx], var=var)
    f.close()

    return sub, group_stats


def select_gene_panel(sub: "ad.AnnData", vcc26_hvg_set: set, n_top_variable_per_dataset: int):
    """(VCC26 HVG set, restricted to genes present in `sub`) UNION (`sub`'s
    own top `n_top_variable_per_dataset` genes by variance on its
    already-subsampled control cells)."""
    control_mask = sub.obs["is_control"].to_numpy()
    var_names = sub.var_names

    ranked = rank_genes_by_variance(sub.X[control_mask], var_names)
    n_top = min(n_top_variable_per_dataset, len(ranked))
    dataset_top_genes = set(ranked[:n_top])

    vcc26_present_genes = vcc26_hvg_set & set(var_names)

    selected_genes = vcc26_present_genes | dataset_top_genes
    selected_mask = var_names.isin(selected_genes)
    selected_idx = np.flatnonzero(selected_mask)

    def label(gene):
        in_vcc26 = gene in vcc26_present_genes
        in_dataset_top = gene in dataset_top_genes
        if in_vcc26 and in_dataset_top:
            return "vcc26_hvg+dataset_top"
        return "vcc26_hvg" if in_vcc26 else "dataset_top"

    genes_selected = var_names.to_numpy()[selected_idx]
    selection = pd.DataFrame({
        "gene": genes_selected,
        "selection_type": [label(g) for g in genes_selected],
    })
    return selected_idx, selection


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined-dir", default=str(COMBINED_DIR),
                     help="Every *.h5ad directly in this directory is processed.")
    ap.add_argument("--vcc26-dir", default=str(VCC26_DIR))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--stats-output", default=None, help="defaults to <output-dir>/cell_subset_stats.csv")
    ap.add_argument("--genes-output", default=None, help="defaults to <output-dir>/selected_genes.csv")
    ap.add_argument("--n-perturbations", type=int, default=N_PERTURBATIONS)
    ap.add_argument("--cells-min", type=int, default=CELLS_MIN)
    ap.add_argument("--cells-max", type=int, default=CELLS_MAX)
    ap.add_argument("--vcc26-hvg-per-context", type=int, default=VCC26_HVG_PER_CONTEXT)
    ap.add_argument("--n-top-variable-genes-per-dataset", type=int, default=N_TOP_VARIABLE_GENES_PER_DATASET)
    ap.add_argument("--cpm-target-sum", type=float, default=CPM_TARGET_SUM,
                     help="Target sum for the lognorm layer, recomputed on the final gene panel "
                          "(not copied from the source file's full-gene-panel lognorm).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    combined_dir = Path(args.combined_dir).resolve()
    vcc26_dir = Path(args.vcc26_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    stats_path = Path(args.stats_output).resolve() if args.stats_output else output_dir / "cell_subset_stats.csv"
    genes_path = Path(args.genes_output).resolve() if args.genes_output else output_dir / "selected_genes.csv"
    rng = np.random.default_rng(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(combined_dir.glob("*.h5ad"))
    if not source_files:
        sys.exit(f"No *.h5ad files found in {combined_dir} -- run combine_datasets.py first.")
    print(f"Found {len(source_files)} dataset(s) in {combined_dir}:")
    for p in source_files:
        print(f"  {p.name}")

    print(f"\nComputing the VCC26 reference HVG set from {vcc26_dir} ...")
    vcc26_hvg_set = compute_vcc26_hvg_set(vcc26_dir, args.vcc26_hvg_per_context)
    print()

    all_cell_stats = []
    all_gene_selections = []
    for path in source_files:
        print(f"Sampling {path.name} ...", flush=True)
        sub, group_stats = sample_cells(path, args.n_perturbations, args.cells_min, args.cells_max, rng)
        all_cell_stats.extend(group_stats)

        selected_idx, selection = select_gene_panel(sub, vcc26_hvg_set, args.n_top_variable_genes_per_dataset)
        selection.insert(0, "source_file", path.name)
        all_gene_selections.append(selection)
        n_vcc26_present = (selection["selection_type"] != "dataset_top").sum()

        sub = sub[:, selected_idx].copy()
        sub.uns["gene_panel_source"] = "vcc26_hvg (present in dataset) union dataset's own top-N HVGs on control cells"
        sub.uns["vcc26_hvg_per_context"] = int(args.vcc26_hvg_per_context)
        sub.uns["n_top_variable_genes_per_dataset"] = int(args.n_top_variable_genes_per_dataset)

        # Recomputed on the final gene panel, not copied from the source file --
        # per-cell library sizes (and so CPM) differ once most genes are
        # dropped, so this intentionally won't match the source's lognorm.
        sub.layers["lognorm"] = compute_lognorm_layer(sub.X, args.cpm_target_sum)

        n_perts = sum(1 for g in group_stats if not g["is_control"])
        out_path = output_dir / f"{path.stem}_subset.h5ad"
        sub.write_h5ad(out_path)
        print(
            f"  {n_perts} perturbations + control -> {sub.n_obs:,} cells x {sub.n_vars:,} genes "
            f"({n_vcc26_present} from VCC26 HVG set, {sub.n_vars - n_vcc26_present} dataset-only top HVGs "
            f"not already in that set) -> {out_path}",
            flush=True,
        )

    stats_df = pd.DataFrame(all_cell_stats)
    genes_df = pd.concat(all_gene_selections, ignore_index=True)

    print()
    print(stats_df.to_string(index=False))

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    genes_path.parent.mkdir(parents=True, exist_ok=True)
    genes_df.to_csv(genes_path, index=False)

    print(f"\nPer-group cell stats written to {stats_path}")
    print(f"Selected gene panels written to {genes_path}")
    print(f"Subsets written to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
