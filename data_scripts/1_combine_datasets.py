#!/usr/bin/env python3
"""
Filter REPLOG + NADIG + VCC25 to the VCC26 gene/perturbation panel and write
each source file to its own standalone h5ad under data/Combined/ -- NOT one
big merged file.

Why separate files instead of one union: REPLOG/NADIG measure ~8-9k genes,
VCC25 measures ~18k, out of the 18,533-gene VCC26 panel. Concatenating them
into one AnnData means an "outer join" that zero-pads every cell for genes
its own source file never measured -- indistinguishable from a real observed
zero, inflating apparent sparsity, and it would skew any downstream per-gene
statistic (e.g. variance for HVG selection) computed naively across the
union. It also doesn't fit in memory: REPLOG's biggest file alone
reconstructs to 50+ GB once you account for its ~35-45% nonzero density
(sparse storage barely helps at that density).

Instead: each dataset stays its own file with its own (possibly different)
gene panel -- exactly the genes it actually measured, restricted to VCC26 --
and manifest.json records which file has which panel. Cross-panel alignment
(shared genes vs. dataset-specific genes) becomes the model's job -- e.g. a
per-dataset encoder, or a masked loss keyed off each dataset's own panel
from the manifest -- rather than something baked into the data on disk.

Pipeline, per source file:
  1. Keep only genes (var) whose symbol is in the VCC26 valid gene list
     (data/VCC26/controls/gene_names.csv, 18,533 symbols).
  2. Keep only cells whose perturbation target is in that same list, or is
     a control cell (see CONTROL_LABEL / KEEP_CONTROL_CELLS below).
  3. Stream the kept rows to disk in chunks (never materializing the whole
     file in memory -- required for the ~2M-cell, 50+ GB REPLOG files) and
     merge the chunks into one h5ad per source file with a small hand-rolled
     out-of-core CSR merger (`merge_chunks_to_h5ad` below; anndata's own
     `concat_on_disk` needs anndata>=0.10, which needs Python>=3.9/3.10 --
     not available on this project's Python 3.8 venv, capped at anndata
     0.9.2). It only ever holds one chunk's data in memory at a time and
     was verified against a direct in-memory reference on synthetic data
     before being wired in here.
  4. Add `layers["lognorm"]` (CPM then log1p) computed per chunk -- a
     per-cell operation, so chunk-local results are identical to computing
     it on the whole file at once.
  5. Every file's stats (genes/cells/perturbations kept vs. discarded) and
     gene panel are recorded in one manifest.json in data/Combined/, for a
     training pipeline to load dataset configs programmatically.

Not run automatically -- draft for you to fine-tune before executing.

Requires: anndata (0.9+; uses anndata.experimental.read_elem/write_elem), pandas, numpy, scipy
    pip install anndata pandas numpy scipy

Usage:
    python data_scripts/combine_datasets.py [--data-root DATA_DIR] [--output-dir OUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import anndata as ad
except ImportError:
    sys.exit(
        "This script requires the 'anndata' package.\n"
        "Install it first: pip install anndata"
    )

import h5py

try:
    from anndata.experimental import read_elem, write_elem
except ImportError:
    # Newer anndata (>=0.11ish) relocated these to anndata.io.
    from anndata.io import read_elem, write_elem

# ----------------------------------------------------------------------------
# CONFIG -- tune freely
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

VCC26_GENE_LIST = DATA_ROOT / "VCC26" / "controls" / "gene_names.csv"
VCC26_GENE_COL = "gene_name"

# (dataset family, path relative to DATA_ROOT). TAHOE100M is intentionally
# excluded -- it's small-molecule drug perturbation, not genetic knockdown,
# and was not part of the requested merge.
SOURCE_FILES = [
    ("REPLOG", "REPLOG/K562_essential_raw_singlecell_01.h5ad"),
    ("REPLOG", "REPLOG/K562_gwps_raw_singlecell_01.h5ad"),
    ("REPLOG", "REPLOG/rpe1_raw_singlecell_01.h5ad"),
    ("NADIG", "NADIG/hepg2_raw_singlecell.h5ad"),
    ("NADIG", "NADIG/jurkat_raw_singlecell.h5ad"),
    ("VCC25", "VCC25/train/adata_Training.h5ad"),
    ("VCC25", "VCC25/test/adata_Test.h5ad"),
    ("VCC25", "VCC25/validation/adata_Validation.h5ad"),
]

OUTPUT_DIR = DATA_ROOT / "Combined"
CHUNK_DIR = OUTPUT_DIR / "_chunks"          # temp, deleted after each file merges (unless --keep-chunks)
OUTPUT_STATS_CSV = OUTPUT_DIR / "combine_stats.csv"
OUTPUT_MANIFEST_JSON = OUTPUT_DIR / "manifest.json"

# Control cells: every source file uses this exact label (verified against
# REPLOG/NADIG/VCC25/VCC26 -- all four use "non-targeting").
CONTROL_LABEL = "non-targeting"

# Literal reading of the brief is "only perturbations that are valid gene
# names are kept", which would also drop non-targeting control cells (since
# "non-targeting" isn't itself a gene name). That's almost certainly not
# what you want for downstream perturbation modeling, so controls are kept
# by default. Flip this to False to apply the filter literally.
KEEP_CONTROL_CELLS = True

# Rows are streamed off disk and written to temp chunk files this many cells
# at a time, so even the ~2M-cell REPLOG files never need to be fully
# materialized in memory. The lognorm computation (astype(float64) + a
# multiply, both full copies of the chunk) is the actual peak, not the raw
# chunk itself -- roughly 3-4x a chunk's own sparse size at 20_000. Lowered
# from 20_000 after that OOM'd on a 20GB machine; raise it back up if you
# have more headroom, or lower it further if this is still too tight.
CHUNK_SIZE = 5_000

# Cap the number of *kept* cells read from each file (after filtering), for
# a cheap smoke-test run of the pipeline. Set to None for the real run. Note:
# this alone does NOT make a smoke test fast on the huge dense REPLOG files --
# it still has to read scattered rows from across the whole file. Use
# MAX_INPUT_ROWS_PER_FILE below for that.
SUBSAMPLE_PER_FILE = None

# Cap each file to its first N rows BEFORE any filtering -- a single
# contiguous disk read, so this is the knob that actually makes a smoke test
# fast (seconds instead of minutes/hours on the multi-GB REPLOG files). Set
# to None for the real run. Good smoke-test value: 5_000-20_000.
MAX_INPUT_ROWS_PER_FILE = None

# Adds layers["lognorm"] = log1p(CPM(X)) to every output file, next to the
# raw counts in .X. CPM (target_sum=1e6) rather than median-total-count
# normalization, since the source datasets have quite different sequencing
# depths and a fixed target keeps them comparable. Computed per chunk (a
# per-cell operation, so this is identical to computing it on the whole
# file at once).
ADD_LOGNORM_LAYER = True
CPM_TARGET_SUM = 1e6

KEEP_CHUNK_FILES = False   # keep the intermediate per-chunk h5ad files (debugging) instead of deleting them

# ----------------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------------


def load_valid_genes(path: Path, col: str) -> np.ndarray:
    df = pd.read_csv(path)
    genes = pd.Index(df[col].astype(str).unique())
    print(f"Loaded {len(genes):,} valid gene names from {path}")
    return genes.to_numpy()


def resolve_gene_symbols(adata: "ad.AnnData") -> np.ndarray:
    """REPLOG/NADIG index var by Ensembl gene_id with symbols in a separate
    var['gene_name'] column; VCC25 indexes var directly by gene symbol."""
    first = str(adata.var_names[0])
    if first.startswith("ENSG"):
        return adata.var["gene_name"].astype(str).to_numpy()
    return adata.var_names.to_numpy().astype(str)


def resolve_perturbation_column(adata: "ad.AnnData") -> str:
    if "target_gene" in adata.obs.columns:
        return "target_gene"
    if "gene" in adata.obs.columns:
        return "gene"
    raise ValueError(
        f"No known perturbation column (target_gene/gene) found in obs: {list(adata.obs.columns)}"
    )


def compute_lognorm_layer(X: sp.csr_matrix, target_sum: float) -> sp.csr_matrix:
    """CPM-style per-cell normalization (scale each cell's counts to sum to
    `target_sum`) followed by log1p. log1p is applied only to the stored
    (nonzero) entries -- log1p(0) == 0, so this keeps the matrix sparse."""
    X = X.tocsr().astype(np.float64)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid dividing all-zero cells by zero
    scale = target_sum / row_sums
    normalized = X.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def merge_chunks_to_h5ad(chunk_paths: list, output_path: Path, layer_names: tuple = ()) -> None:
    """Merge many small per-chunk h5ad files (all sharing the identical var
    axis -- they're row-chunks of the same source file) into one output
    h5ad, without ever holding more than one chunk's worth of data in
    memory. Stands in for anndata.experimental.concat_on_disk, which needs
    anndata>=0.10 (Python>=3.9/3.10) -- unavailable on this project's
    Python 3.8 / anndata 0.9.2 venv. Verified against a direct in-memory
    vstack reference on synthetic data before being used here.
    """
    n_cols = None
    total_rows = 0
    total_nnz = {"X": 0}
    dtypes = {}
    for ln in layer_names:
        total_nnz[ln] = 0

    for p in chunk_paths:
        with h5py.File(p, "r") as f:
            shape = f["X"].attrs["shape"]
            if n_cols is None:
                n_cols = int(shape[1])
            elif n_cols != int(shape[1]):
                raise ValueError(f"Column mismatch across chunks for {output_path}: {n_cols} vs {shape[1]}")
            total_rows += int(shape[0])
            total_nnz["X"] += int(f["X"]["indptr"][-1])
            dtypes.setdefault("X", f["X"]["data"].dtype)
            for ln in layer_names:
                total_nnz[ln] += int(f["layers"][ln]["indptr"][-1])
                dtypes.setdefault(ln, f["layers"][ln]["data"].dtype)

    # obs/var are small (metadata only) -- concatenating them in memory is
    # cheap even for a multi-million-row file; it's only X/layers (the
    # actual expression values) that are too big to hold all at once.
    with h5py.File(chunk_paths[0], "r") as f:
        var_ref = read_elem(f["var"])
    obs_parts = []
    for p in chunk_paths:
        with h5py.File(p, "r") as f:
            obs_parts.append(read_elem(f["obs"]))
    obs_full = pd.concat(obs_parts, axis=0)

    # Write a correctly-structured "shell" (obs/var/uns/obsm/varm/obsp/varp
    # and all the right root-level attrs) via a normal small AnnData write,
    # with placeholder (empty) sparse X/layers of the right final shape --
    # then hand-fill the real data in below, chunk by chunk.
    shell = ad.AnnData(X=sp.csr_matrix((total_rows, n_cols), dtype=dtypes["X"]), obs=obs_full, var=var_ref)
    for ln in layer_names:
        shell.layers[ln] = sp.csr_matrix((total_rows, n_cols), dtype=dtypes[ln])
    shell.write_h5ad(output_path)

    with h5py.File(output_path, "a") as f:
        for slot, group_path in [("X", "X")] + [(ln, f"layers/{ln}") for ln in layer_names]:
            grp = f[group_path]
            del grp["data"], grp["indices"], grp["indptr"]
            # indices and indptr must share one dtype -- scipy's compiled
            # CSR routines (e.g. row fancy-indexing) reject a mismatched
            # pair with "Output dtype not compatible with inputs", even
            # though a plain read/write round-trip doesn't catch it.
            idx_dtype = np.int64 if total_nnz[slot] > np.iinfo(np.int32).max else np.int32
            grp.create_dataset("data", shape=(total_nnz[slot],), dtype=dtypes[slot])
            grp.create_dataset("indices", shape=(total_nnz[slot],), dtype=idx_dtype)
            grp.create_dataset("indptr", shape=(total_rows + 1,), dtype=idx_dtype)
            grp.attrs["shape"] = np.array([total_rows, n_cols])

            row_offset = 0
            nnz_offset = 0
            grp["indptr"][0] = 0
            for p in chunk_paths:
                with h5py.File(p, "r") as cf:
                    src = cf["X"] if slot == "X" else cf["layers"][slot]
                    c_nnz = int(src["indptr"][-1])
                    c_rows = int(src.attrs["shape"][0])
                    if c_nnz > 0:
                        grp["data"][nnz_offset:nnz_offset + c_nnz] = src["data"][:]
                        grp["indices"][nnz_offset:nnz_offset + c_nnz] = src["indices"][:]
                    grp["indptr"][row_offset + 1:row_offset + 1 + c_rows] = src["indptr"][1:].astype(np.int64) + nnz_offset
                    row_offset += c_rows
                    nnz_offset += c_nnz


def process_file(
    path: Path,
    family: str,
    valid_genes: np.ndarray,
    output_path: Path,
    chunk_dir: Path,
    chunk_size: int,
    subsample: int | None,
    max_input_rows: int | None,
    add_lognorm: bool,
    cpm_target_sum: float,
    keep_chunks: bool,
):
    adata = ad.read_h5ad(path, backed="r")
    if max_input_rows is not None and adata.n_obs > max_input_rows:
        # Cap to a contiguous prefix BEFORE any filtering -- single
        # contiguous disk read, and small enough to materialize outright
        # (a backed AnnData can't be indexed twice, "no view of a view").
        adata = adata[:max_input_rows].to_memory()

    # ---- gene / feature filtering ----
    n_genes_total = adata.n_vars
    symbols = resolve_gene_symbols(adata)
    gene_keep = np.isin(symbols, valid_genes)

    # A handful of Ensembl IDs map to the same symbol in REPLOG/NADIG;
    # keep only the first occurrence of each symbol among the kept genes.
    kept_idx = np.flatnonzero(gene_keep)
    dup_within_kept = pd.Series(symbols[kept_idx]).duplicated(keep="first").to_numpy()
    dup_mask = np.zeros(n_genes_total, dtype=bool)
    dup_mask[kept_idx[dup_within_kept]] = True

    keep_col_idx = np.flatnonzero(gene_keep & ~dup_mask)
    n_genes_kept = int(keep_col_idx.size)
    n_genes_dropped_as_duplicate = int(dup_mask.sum())
    n_genes_discarded = n_genes_total - n_genes_kept - n_genes_dropped_as_duplicate
    kept_gene_names = symbols[keep_col_idx].tolist()

    # ---- perturbation filtering ----
    pert_col = resolve_perturbation_column(adata)
    labels = adata.obs[pert_col].astype(str).to_numpy()
    n_cells_total = adata.n_obs

    is_control = labels == CONTROL_LABEL
    is_valid_pert = np.isin(labels, valid_genes)
    obs_keep_mask = is_valid_pert | (KEEP_CONTROL_CELLS & is_control)

    n_perturbations_total = int(pd.unique(labels[~is_control]).size)
    kept_pert_labels = labels[obs_keep_mask & ~is_control]
    n_perturbations_kept = int(pd.unique(kept_pert_labels).size)
    n_control_cells_total = int(is_control.sum())
    n_control_cells_kept = int((obs_keep_mask & is_control).sum())

    keep_row_idx = np.flatnonzero(obs_keep_mask)
    if subsample is not None and keep_row_idx.size > subsample:
        rng = np.random.default_rng(0)
        keep_row_idx = np.sort(rng.choice(keep_row_idx, size=subsample, replace=False))

    n_cells_kept = int(keep_row_idx.size)
    n_cells_discarded = n_cells_total - n_cells_kept

    # Unnamed index (-> writes as the plain "_index" AnnData uses by default),
    # matching VCC26's own context_A/B/C.h5ad var axis exactly (bare gene
    # symbols, no index name, no extra var columns).
    var_filtered = pd.DataFrame(index=pd.Index(kept_gene_names))

    stats = dict(
        dataset_family=family,
        source_file=path.name,
        output_path=str(output_path),
        n_genes_total=n_genes_total,
        n_genes_kept=n_genes_kept,
        n_genes_discarded=n_genes_discarded,
        n_genes_dropped_as_duplicate=n_genes_dropped_as_duplicate,
        n_cells_total=n_cells_total,
        n_cells_kept=n_cells_kept,
        n_cells_discarded=n_cells_discarded,
        n_control_cells_total=n_control_cells_total,
        n_control_cells_kept=n_control_cells_kept,
        n_perturbations_total=n_perturbations_total,
        n_perturbations_kept=n_perturbations_kept,
        n_perturbations_discarded=n_perturbations_total - n_perturbations_kept,
    )

    if n_cells_kept == 0:
        # Nothing survived filtering -- write an empty (but well-formed) file.
        empty = ad.AnnData(X=sp.csr_matrix((0, n_genes_kept)), obs=pd.DataFrame(), var=var_filtered)
        empty.write_h5ad(output_path)
        adata.file.close()
        return stats, kept_gene_names

    # ---- stream the kept rows to per-chunk h5ad files, never holding the
    # ---- whole (filtered) file in memory at once ----
    file_chunk_dir = chunk_dir / f"{family}__{path.stem}"
    file_chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    for start in range(0, keep_row_idx.size, chunk_size):
        chunk_rows = keep_row_idx[start:start + chunk_size]
        # h5py only allows fancy-indexing one axis at a time on a dense
        # backed dataset (REPLOG/NADIG), so rows and columns can't be
        # selected in the same call: slice rows only while still backed,
        # materialize, then subset columns on the resulting in-memory array.
        sub = adata[chunk_rows, :].to_memory()
        X_chunk = sub.X
        if not sp.issparse(X_chunk):
            X_chunk = sp.csr_matrix(X_chunk)
        X_chunk = X_chunk[:, keep_col_idx].tocsr()

        obs_chunk = sub.obs.copy()
        del sub  # drop the dense (REPLOG/NADIG) chunk array now, not after lognorm too
        obs_chunk["target_gene"] = labels[chunk_rows]
        # VCC26's own context_A/B/C.h5ad use exactly this column (its
        # manifest.json even names it via "context_col": "context") to say
        # which sub-dataset a cell belongs to -- one value per source file
        # here, same role as VCC26's per-context "A"/"B"/"C".
        obs_chunk["context"] = path.stem
        obs_chunk["is_control"] = is_control[chunk_rows]
        obs_chunk["dataset_family"] = family
        obs_chunk["source_file"] = path.name
        obs_chunk = obs_chunk.reset_index(drop=True)
        # Not the original barcode -- REPLOG-style multi-gem_group files can
        # repeat a raw barcode across gem groups, unsafe as a uniqueness
        # guarantee -- so obs_names is synthesized instead.
        obs_chunk.index = [f"{family}:{path.stem}:{start + i}" for i in range(len(obs_chunk))]

        chunk_adata = ad.AnnData(X=X_chunk, obs=obs_chunk, var=var_filtered)
        if add_lognorm:
            chunk_adata.layers["lognorm"] = compute_lognorm_layer(X_chunk, cpm_target_sum)

        chunk_path = file_chunk_dir / f"chunk_{start:09d}.h5ad"
        chunk_adata.write_h5ad(chunk_path)
        chunk_paths.append(chunk_path)
        del X_chunk, chunk_adata

    adata.file.close()

    # ---- merge this file's chunks into the final per-file output, on disk
    # ---- (no in-memory concat -- this is what keeps a ~2M-cell file from
    # ---- ever needing 50+ GB of RAM at once) ----
    if len(chunk_paths) == 1:
        shutil.move(str(chunk_paths[0]), str(output_path))
    else:
        merge_chunks_to_h5ad(chunk_paths, output_path, layer_names=("lognorm",) if add_lognorm else ())

    if not keep_chunks:
        shutil.rmtree(file_chunk_dir, ignore_errors=True)

    return stats, kept_gene_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--chunk-dir", default=str(CHUNK_DIR))
    ap.add_argument("--stats-output", default=None, help="defaults to <output-dir>/combine_stats.csv")
    ap.add_argument("--manifest-output", default=None, help="defaults to <output-dir>/manifest.json")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--subsample-per-file", type=int, default=SUBSAMPLE_PER_FILE)
    ap.add_argument("--max-input-rows", type=int, default=MAX_INPUT_ROWS_PER_FILE,
                     help="Cap each file to its first N rows before filtering -- a fast smoke-test knob "
                          "(single contiguous read, unlike --subsample-per-file). E.g. 10000.")
    ap.add_argument("--drop-control-cells", action="store_true",
                     help="Apply the gene-name filter literally to controls too (drops all non-targeting cells).")
    ap.add_argument("--skip-lognorm", action="store_true",
                     help="Don't add the layers['lognorm'] (CPM + log1p) layer -- keep raw counts only.")
    ap.add_argument("--cpm-target-sum", type=float, default=CPM_TARGET_SUM)
    ap.add_argument("--keep-chunks", action="store_true",
                     help="Keep the intermediate per-chunk h5ad files instead of deleting them after merging.")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    chunk_dir = Path(args.chunk_dir).resolve()
    stats_path = Path(args.stats_output).resolve() if args.stats_output else output_dir / "combine_stats.csv"
    manifest_path = Path(args.manifest_output).resolve() if args.manifest_output else output_dir / "manifest.json"

    global KEEP_CONTROL_CELLS
    if args.drop_control_cells:
        KEEP_CONTROL_CELLS = False

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    valid_genes = load_valid_genes(data_root / "VCC26" / "controls" / "gene_names.csv", VCC26_GENE_COL)

    stats_rows = []
    manifest_datasets = []
    for family, relpath in SOURCE_FILES:
        path = data_root / relpath
        out_path = output_dir / f"{family}__{path.stem}.h5ad"
        print(f"Processing {family}/{path.name} -> {out_path.name} ...", flush=True)
        stats, gene_panel = process_file(
            path, family, valid_genes, out_path, chunk_dir,
            args.chunk_size, args.subsample_per_file, args.max_input_rows,
            not args.skip_lognorm, args.cpm_target_sum, args.keep_chunks,
        )
        stats_rows.append(stats)
        manifest_datasets.append(dict(**stats, gene_panel=gene_panel))
        print(
            f"  genes kept {stats['n_genes_kept']:,}/{stats['n_genes_total']:,} "
            f"(dup dropped {stats['n_genes_dropped_as_duplicate']}) | "
            f"cells kept {stats['n_cells_kept']:,}/{stats['n_cells_total']:,} "
            f"(controls kept {stats['n_control_cells_kept']:,}/{stats['n_control_cells_total']:,}) | "
            f"perturbations kept {stats['n_perturbations_kept']:,}/{stats['n_perturbations_total']:,} "
            f"-> {out_path}",
            flush=True,
        )

    if not args.keep_chunks:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    stats_df = pd.DataFrame(stats_rows)
    print()
    print(stats_df.to_string(index=False))

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)

    manifest = dict(
        vcc26_gene_list_path=str(VCC26_GENE_LIST),
        control_label=CONTROL_LABEL,
        control_cells_kept=bool(KEEP_CONTROL_CELLS),
        lognorm_layer_added=bool(not args.skip_lognorm),
        lognorm_target_sum=float(args.cpm_target_sum) if not args.skip_lognorm else None,
        datasets=manifest_datasets,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPer-file stats written to {stats_path}")
    print(f"Manifest (dataset paths + gene panels) written to {manifest_path}")
    print(f"{len(SOURCE_FILES)} separate h5ad files written to {output_dir} "
          f"(kept as distinct datasets -- no cross-file union/padding).")


if __name__ == "__main__":
    sys.exit(main())
