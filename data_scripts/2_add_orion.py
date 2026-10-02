#!/usr/bin/env python3
"""
Add X-Atlas/Orion (HCT116 + HEK293T genome-wide Perturb-seq) to data/Combined/, in exactly
the format 1_combine_datasets.py wrote for REPLOG / NADIG / VCC25:

    <out>/ORION__HCT116.h5ad, <out>/ORION__HEK293T.h5ad
        X                  raw UMI counts (float32, CSR), genes = the VCC26 panel genes Orion measures
        layers["lognorm"]  log1p(CPM) over those genes (1_combine_datasets.compute_lognorm_layer)
        obs                target_gene, guide_id, batch, context, is_control, dataset_family, source_file
        var                bare gene symbols (VCC26 naming)

INPUT  /home/ext1/data/X-Atlas-Orion/data/<CELL_LINE>_Batch*.parquet: one row per cell, with
       the nonzero genes as (gene_token_id, gene_expression) lists, plus metadata; gene tokens
       -> Ensembl ID + symbol in metadata/gene_metadata.parquet.

GENE NAMES: Orion uses the 10x GRCh38 2024-A reference, i.e. NEWER official symbols than VCC26
and the other datasets (AARS -> AARS1, H2AFX -> H2AX, MARCH5 -> MARCHF5, ...). Every Orion gene
is mapped to a VCC26 symbol, in this order of trust:
    1. direct      its symbol IS a VCC26 symbol
    2. ensembl     its Ensembl ID carries an OLD symbol in the REPLOG / NADIG source files
                   (their var: Ensembl ID + symbol), and that old symbol is a VCC26 gene
    3. rename      unambiguous HGNC renames: "<X>1" -> "<X>" (X ending in a letter) and "MARCHF<n>" -> "MARCH<n>",
                   only if the old name is a VCC26 gene and not an Orion symbol itself
One Orion gene per VCC26 symbol (the most trusted match wins). The knockdown targets
(gene_target) are renamed with the same map; "Non-Targeting" -> "non-targeting".
The full map is written to <out>/orion_gene_mapping.csv.

CELLS (a size decision, see the conversation of 2026-09-28): per cell line, cells passing
Orion's guide filter, whose knockdown maps to a VCC26 gene (or controls); at most
--cells-per-pert cells per knockdown (default 200) and --control-cells controls (default
30,000), drawn at random (seeded). All ~8 M cells would not fit the SSD for `make prepare`.

Run (takes a few hours; reads ~126 GB of parquet, writes ~300 GB):
    .venv/bin/python data_scripts/2_add_orion.py [--cells-per-pert 200] [--control-cells 30000]
Then `make prepare` converts only the two new files (stage 1 is incremental) and rebuilds
the rest; data/Combined is a symlink to /home/ext1/data/Combined, so nothing else to copy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

REPO = Path(__file__).resolve().parent.parent
ORION = Path("/home/ext1/data/X-Atlas-Orion")
OUT = Path("/home/ext1/data/Combined")
VCC26_GENES = REPO / "data" / "VCC26" / "controls" / "gene_names.csv"
BRIDGE_FILES = [Path("/home/ext1/data/REPLOG") / f for f in
                ("K562_essential_raw_singlecell_01.h5ad", "K562_gwps_raw_singlecell_01.h5ad",
                 "rpe1_raw_singlecell_01.h5ad")] + \
               [Path("/home/ext1/data/NADIG") / f for f in ("hepg2_raw_singlecell.h5ad", "jurkat_raw_singlecell.h5ad")]
CELL_LINES = ("HCT116", "HEK293T")
ORION_CONTROL = "Non-Targeting"
FAMILY = "ORION"

# the SAME lognorm and merge code the other Combined files were written with
_spec = importlib.util.spec_from_file_location("combine", Path(__file__).with_name("1_combine_datasets.py"))
combine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine)
CONTROL_LABEL = combine.CONTROL_LABEL


# ---- gene names ------------------------------------------------------------------

def load_bridge() -> dict[str, str]:
    """Ensembl ID -> OLD symbol, from the REPLOG / NADIG source files' var."""
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    bridge: dict[str, str] = {}
    for f in BRIDGE_FILES:
        with h5py.File(f, "r") as h:
            var = read_elem(h["var"])
        ids = var.index.astype(str)
        sym = var["gene_name"].astype(str) if "gene_name" in var.columns else pd.Series(ids, index=var.index)
        for e, s in zip(ids, sym):
            if e.startswith("ENSG"):
                bridge.setdefault(e.split(".")[0], s)
    return bridge


def build_gene_map(meta: pd.DataFrame, vcc26: list[str], bridge: dict[str, str]) -> pd.DataFrame:
    vset = set(vcc26)
    orion_syms = set(meta["gene_name"].astype(str))
    cand = []                                   # (priority, token, ensembl, orion symbol, vcc26 symbol, method)
    for tok, ens, sym in zip(meta["gene_token_id"], meta["ensembl_id"].astype(str), meta["gene_name"].astype(str)):
        if sym in vset:
            cand.append((0, tok, ens, sym, sym, "direct"))
            continue
        old = bridge.get(ens.split(".")[0])
        if old is not None and old in vset and old not in orion_syms:
            cand.append((1, tok, ens, sym, old, "ensembl"))
            continue
        for old in ([sym[:-1]] if sym.endswith("1") and sym[-2:-1].isalpha() else []) + (["MARCH" + sym[6:]] if sym.startswith("MARCHF") else []):
            if old in vset and old not in orion_syms:
                cand.append((2, tok, ens, sym, old, "rename"))
                break
    df = pd.DataFrame(cand, columns=["priority", "gene_token_id", "ensembl_id", "orion_symbol", "vcc26_symbol", "method"])
    df = df.sort_values(["priority", "gene_token_id"]).drop_duplicates("vcc26_symbol", keep="first")
    order = {g: i for i, g in enumerate(vcc26)}
    df["vcc26_order"] = df["vcc26_symbol"].map(order)
    return df.sort_values("vcc26_order").drop(columns="priority").reset_index(drop=True)


# ---- cells -----------------------------------------------------------------------

def select_cells(cell_line: str, name_map: dict[str, str], vset: set, cells_per_pert: int, control_cells: int,
                 seed: int, max_files: int | None = None) -> tuple[list[Path], pd.DataFrame, dict]:
    """Which rows of which batch files to keep: per knockdown at most `cells_per_pert`, controls
    at most `control_cells`. Reads only the small metadata columns."""
    files = sorted((ORION / "data").glob(f"{cell_line}_Batch*.parquet"),
                   key=lambda p: int(p.stem.split("Batch")[1]))[:max_files]
    parts = []
    for k, f in enumerate(files):
        t = pq.read_table(f, columns=["gene_target", "pass_guide_filter"]).to_pandas()
        t["file"] = k
        t["row"] = np.arange(len(t))
        parts.append(t)
    cells = pd.concat(parts, ignore_index=True)
    n_total = len(cells)
    cells = cells[cells["pass_guide_filter"] == 1]
    tgt = cells["gene_target"].astype(str)
    mapped = tgt.map(lambda s: CONTROL_LABEL if s == ORION_CONTROL else name_map.get(s, s if s in vset else None))
    cells = cells.assign(target_gene=mapped)
    n_unmapped_perts = int(tgt[mapped.isna()].nunique())
    cells = cells[cells["target_gene"].notna()]
    rng = np.random.default_rng(seed)
    is_ctrl = cells["target_gene"] == CONTROL_LABEL
    ctrl = cells[is_ctrl]
    if len(ctrl) > control_cells:
        ctrl = ctrl.iloc[np.sort(rng.choice(len(ctrl), control_cells, replace=False))]
    pert = cells[~is_ctrl]
    shuffled = pert.iloc[rng.permutation(len(pert))]
    pert = shuffled.groupby("target_gene", sort=False).head(cells_per_pert)
    keep = pd.concat([ctrl, pert]).sort_values(["file", "row"]).reset_index(drop=True)
    stats = {"n_cells_total": n_total, "n_cells_kept": len(keep), "n_control_cells_kept": int(len(ctrl)),
             "n_perturbations_kept": int(pert["target_gene"].nunique()),
             "n_perturbation_labels_unmapped": n_unmapped_perts}
    return files, keep, stats


# ---- writing -----------------------------------------------------------------------

def _gather(offsets: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat value positions of `rows` in a list column, and each row's length."""
    starts, lens = offsets[rows], offsets[rows + 1] - offsets[rows]
    shift = np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens)
    return shift + np.arange(int(lens.sum())), lens


def write_cell_line(cell_line: str, files: list[Path], keep: pd.DataFrame, token_col: np.ndarray,
                    var: pd.DataFrame, out_path: Path, chunk_dir: Path, chunk_size: int, target_sum: float) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    done = chunk_dir / "chunks_done.json"
    if done.exists():                        # an earlier run wrote every chunk: only the merge is left
        chunk_paths = [chunk_dir / n for n in json.loads(done.read_text())]
        print(f"    {cell_line}: reusing {len(chunk_paths)} finished chunks in {chunk_dir}", flush=True)
        _finish(chunk_paths, out_path, chunk_dir)
        return
    n_genes = len(var)
    chunk_paths, buf_X, buf_obs, n_written = [], [], [], 0
    t0 = time.time()

    def flush():
        nonlocal buf_X, buf_obs, n_written
        if not buf_X:
            return
        X = sp.vstack(buf_X, format="csr")
        obs = pd.concat(buf_obs, ignore_index=True)
        obs.index = [f"{FAMILY}:{cell_line}:{n_written + i}" for i in range(len(obs))]
        a = ad.AnnData(X=X, obs=obs, var=var)
        a.layers["lognorm"] = combine.compute_lognorm_layer(X, target_sum)
        p = chunk_dir / f"chunk_{n_written:09d}.h5ad"
        a.write_h5ad(p)
        chunk_paths.append(p)
        n_written += X.shape[0]
        buf_X, buf_obs = [], []

    by_file = keep.groupby("file")
    for k, f in enumerate(files):
        if k not in by_file.groups:
            continue
        sel = by_file.get_group(k)
        pf = pq.ParquetFile(f)
        rg_start = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
        for g in range(pf.num_row_groups):
            rows = sel["row"].to_numpy()
            rows = rows[(rows >= rg_start[g]) & (rows < rg_start[g + 1])] - rg_start[g]
            if not len(rows):
                continue
            t = pf.read_row_group(g, columns=["gene_token_id", "gene_expression", "guide_target", "sample"])
            tok_arr = t.column("gene_token_id").combine_chunks()
            exp_arr = t.column("gene_expression").combine_chunks()
            pos, lens = _gather(tok_arr.offsets.to_numpy(), rows)
            toks = tok_arr.values.to_numpy()[pos]
            vals = exp_arr.values.to_numpy()[pos]
            cols = token_col[toks]
            ok = cols >= 0
            row_id = np.repeat(np.arange(len(rows)), lens)[ok]
            X = sp.csr_matrix((vals[ok].astype(np.float32), (row_id, cols[ok])), shape=(len(rows), n_genes))
            X.sort_indices()
            meta_rows = sel[(sel["row"] >= rg_start[g]) & (sel["row"] < rg_start[g + 1])]
            guide = np.asarray(t.column("guide_target").to_pylist(), dtype=object)[rows]
            batch = np.asarray(t.column("sample").to_pylist(), dtype=object)[rows]
            buf_X.append(X)
            buf_obs.append(pd.DataFrame({
                "target_gene": meta_rows["target_gene"].to_numpy(), "guide_id": guide, "batch": batch,
                "context": cell_line, "is_control": meta_rows["target_gene"].to_numpy() == CONTROL_LABEL,
                "dataset_family": FAMILY, "source_file": f.name}))
            if sum(x.shape[0] for x in buf_X) >= chunk_size:
                flush()
        print(f"    {cell_line}: file {k + 1}/{len(files)}  {n_written:,} cells written  [{time.time() - t0:.0f}s]",
              flush=True)
    flush()
    done.write_text(json.dumps([q.name for q in chunk_paths]))
    _finish(chunk_paths, out_path, chunk_dir)


def _finish(chunk_paths: list[Path], out_path: Path, chunk_dir: Path) -> None:
    """Merge the chunks into <out_path>.tmp, then rename: a half-written file never sits in
    Combined/ under a *.h5ad name for `make prepare` to pick up."""
    print(f"    merging {len(chunk_paths)} chunks -> {out_path}", flush=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    merge_chunks(chunk_paths, tmp, layer_names=("lognorm",))
    tmp.replace(out_path)
    shutil.rmtree(chunk_dir, ignore_errors=True)


def merge_chunks(chunk_paths: list[Path], output_path: Path, layer_names: tuple = ()) -> None:
    """1_combine_datasets.merge_chunks_to_h5ad with one fix: each chunk's (int32) indptr is
    cast to int64 before the running nnz offset is added. Orion files pass 2^31 nonzeros,
    where the original raises OverflowError (numpy 2); the earlier datasets never did."""
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    n_cols, total_rows, total_nnz, dtypes = None, 0, {s: 0 for s in ("X", *layer_names)}, {}
    obs_parts = []
    for p in chunk_paths:
        with h5py.File(p, "r") as f:
            shape = f["X"].attrs["shape"]
            if n_cols is None:
                n_cols = int(shape[1])
                var_ref = read_elem(f["var"])
            elif n_cols != int(shape[1]):
                raise ValueError(f"column mismatch across chunks: {n_cols} vs {shape[1]}")
            total_rows += int(shape[0])
            for s in total_nnz:
                g = f["X"] if s == "X" else f["layers"][s]
                total_nnz[s] += int(g["indptr"][-1])
                dtypes.setdefault(s, g["data"].dtype)
            obs_parts.append(read_elem(f["obs"]))
    shell = ad.AnnData(X=sp.csr_matrix((total_rows, n_cols), dtype=dtypes["X"]),
                       obs=pd.concat(obs_parts, axis=0), var=var_ref)
    for s in layer_names:
        shell.layers[s] = sp.csr_matrix((total_rows, n_cols), dtype=dtypes[s])
    shell.write_h5ad(output_path)
    with h5py.File(output_path, "a") as f:
        for slot, path in [("X", "X")] + [(s, f"layers/{s}") for s in layer_names]:
            grp = f[path]
            del grp["data"], grp["indices"], grp["indptr"]
            idx_dtype = np.int64 if total_nnz[slot] > np.iinfo(np.int32).max else np.int32
            grp.create_dataset("data", shape=(total_nnz[slot],), dtype=dtypes[slot])
            grp.create_dataset("indices", shape=(total_nnz[slot],), dtype=idx_dtype)
            grp.create_dataset("indptr", shape=(total_rows + 1,), dtype=idx_dtype)
            grp.attrs["shape"] = np.array([total_rows, n_cols])
            grp["indptr"][0] = 0
            row_off = nnz_off = 0
            for p in chunk_paths:
                with h5py.File(p, "r") as cf:
                    src = cf["X"] if slot == "X" else cf["layers"][slot]
                    c_nnz, c_rows = int(src["indptr"][-1]), int(src.attrs["shape"][0])
                    if c_nnz:
                        grp["data"][nnz_off:nnz_off + c_nnz] = src["data"][:]
                        grp["indices"][nnz_off:nnz_off + c_nnz] = src["indices"][:]
                    grp["indptr"][row_off + 1:row_off + 1 + c_rows] = src["indptr"][1:].astype(np.int64) + nnz_off
                    row_off += c_rows
                    nnz_off += c_nnz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--cells-per-pert", type=int, default=200)
    ap.add_argument("--control-cells", type=int, default=30000)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--cell-lines", nargs="*", default=list(CELL_LINES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=None, help="only the first N batch files per line (smoke test)")
    ap.add_argument("--mapping-only", action="store_true", help="write the gene map and stop (fast)")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vcc26 = pd.read_csv(VCC26_GENES)["gene_name"].astype(str).tolist()
    meta = pd.read_parquet(ORION / "metadata" / "gene_metadata.parquet")
    print(f"Orion: {len(meta):,} genes in its reference; VCC26: {len(vcc26):,} genes")
    gmap = build_gene_map(meta, vcc26, load_bridge())
    gmap.to_csv(out / "orion_gene_mapping.csv", index=False)
    counts = gmap["method"].value_counts().to_dict()
    print(f"gene map: {len(gmap):,}/{len(vcc26):,} VCC26 genes found in Orion  "
          f"(direct {counts.get('direct', 0):,}, via Ensembl {counts.get('ensembl', 0):,}, "
          f"renames {counts.get('rename', 0):,})  -> {out / 'orion_gene_mapping.csv'}")
    if args.mapping_only:
        return

    name_map = dict(zip(gmap["orion_symbol"], gmap["vcc26_symbol"]))
    token_col = np.full(int(meta["gene_token_id"].max()) + 1, -1, dtype=np.int64)
    token_col[gmap["gene_token_id"].to_numpy()] = np.arange(len(gmap))
    var = pd.DataFrame(index=pd.Index(gmap["vcc26_symbol"].tolist()))

    all_stats = []
    for cl in args.cell_lines:
        print(f"\n== {cl}: selecting cells ...", flush=True)
        files, keep, st = select_cells(cl, name_map, set(vcc26), args.cells_per_pert, args.control_cells, args.seed,
                                     args.max_files)
        print(f"   {st['n_cells_kept']:,} of {st['n_cells_total']:,} cells kept: {st['n_perturbations_kept']:,} "
              f"knockdowns (<= {args.cells_per_pert} cells each) + {st['n_control_cells_kept']:,} controls; "
              f"{st['n_perturbation_labels_unmapped']} knockdown labels not in VCC26 dropped", flush=True)
        out_path = out / f"{FAMILY}__{cl}.h5ad"
        write_cell_line(cl, files, keep, token_col, var, out_path, out / f"_chunks_orion_{cl}",
                        args.chunk_size, combine.CPM_TARGET_SUM)
        all_stats.append({"dataset_family": FAMILY, "source_file": f"{cl}_Batch*.parquet", "output_path": str(out_path),
                          "n_genes_total": int(len(meta)), "n_genes_kept": int(len(var)), **st,
                          "cells_per_pert": args.cells_per_pert, "control_cells_cap": args.control_cells})
        print(f"   wrote {out_path}  [{time.time() - t0:.0f}s]", flush=True)

    (out / "orion_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"\ndone: {len(all_stats)} file(s), stats in {out / 'orion_stats.json'}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
