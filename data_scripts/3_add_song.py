#!/usr/bin/env python3
"""
Add Song et al. genome-wide CRISPRi Perturb-seq in Jurkat T cells (GEO GSM7951413-GSM7951428,
16 10x lanes, downloaded by data/download_song.sh into /home/ext1/data/Song25/) to data/Combined/,
in exactly the format 1_combine_datasets.py / 2_add_orion.py wrote:

    <out>/SONG25__jurkat.h5ad   (or SONG25__jurkat_<condition>.h5ad per condition, --split-conditions)
        X                  raw UMI counts (float32, CSR), genes = the VCC26 panel genes Song measures
        layers["lognorm"]  log1p(CPM) over those genes (1_combine_datasets.compute_lognorm_layer)
        obs                target_gene, guide_id, batch, context, is_control, dataset_family, source_file
        var                bare gene symbols (VCC26 naming)

`make prepare` globs data/Combined/*.h5ad, so nothing else has to be registered: the new context is
converted, OT-coupled, added to the shared PCA space and delta fingerprints, and trained on / predicted
from like every other context. (The model genes are the genes EVERY context measures, so they shrink
by the few genes Song lacks; the PCA basis is refit, hence a new model run is needed. Old runs keep
working: prediction uses the run's own pca.npz.)

INPUT (per lane, prefix <GSM>_channel<N>_):  {transcriptome,guides,labels}_{matrix.mtx,barcode.tsv,features.tsv}.gz
    transcriptome  20,606 genes x 6.79M barcodes (the WHOLE 10x whitelist: almost all are empty droplets)
    guides         83,401 guide features ("A1BG_1" .. "non-targeting_250") x barcodes, guide UMI counts
    labels         7 condition hashtags (activated1..6, untreated) x barcodes

WHAT THE DATA LOOKS LIKE (measured on lane 1, 2026-09-29) AND WHAT THE DEFAULTS DO ABOUT IT
  * Cells: barcodes with >= --min-umi transcript UMIs (default 2000; lane 1: 42k cells).
  * Guides are HIGH-MOI and noisy: a real cell carries ~75 guide features, the top guide is ~9% of
    the cell's guide UMIs, and the guide count histogram has no clean bimodal split. So "the top guide"
    is NOT the perturbation. A cell is assigned to a knockdown only if EXACTLY ONE target gene has
    >= --min-guide-umi guide UMIs summed over its guides (default 10; lane 1: ~20% of cells). Cells
    with two or more such genes are dropped (unresolvable combinations). This is a heuristic: tune it
    with --stats-only before the long run.
  * Controls: cells whose one passing target is "non-targeting" (--controls nt; only 250 of 83k guides,
    so few cells), cells with NO gene at >= --min-guide-umi (--controls noguide; many, but they are
    "no detected guide", not proven non-targeting), or both (--controls both).
  * Condition: the hashtag with the most UMIs, needs >= --min-label-umi and --label-dominance x the
    runner-up (default 10 and 3x; lane 1: ~73% of cells pass).
    Default: all conditions POOLED in one context (most cells per knockdown; data.pca.min_cells is 20).
    --split-conditions writes one file/context per condition, --conditions picks a subset.
  * Yield is the limiting factor: expect on the order of 10 cells per knockdown when pooled, so few
    knockdowns will reach data.pca.min_cells (20) for the delta fingerprint. Run --stats-only first.

GENE NAMES: Song uses the same newer 10x/GENCODE symbols as Orion (AARS -> AARS1 ...). The gene map
is built by 2_add_orion.build_gene_map (direct symbol, then Ensembl ID via the REPLOG/NADIG old
symbols, then unambiguous HGNC renames), written to <out>/song_gene_mapping.csv. Guide target names
(the guide feature name minus "_<n>") are renamed with the same map.

REUSED FROM 2_add_orion.py: gene mapping, and the merge that casts every chunk's indptr to int64 before
adding the running nnz offset (the int32 overflow that crashed the first Orion build). Existing output
files are never overwritten unless --overwrite is given.

Run (reads ~3.5 GB of gz matrices twice; a few hours at most, mostly the mtx parsing):
    .venv/bin/python data_scripts/3_add_song.py --stats-only          # yields only, writes nothing
    .venv/bin/python data_scripts/3_add_song.py [--cells-per-pert 200] [--control-cells 30000]
Then `make prepare` converts only the new file(s) (stage 1 is incremental) and rebuilds the rest.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(fname.split(".")[0].replace("2_", "m2_"), HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orion = _load("2_add_orion.py")           # gene mapping + the int64-safe chunk merge
combine = orion.combine
CONTROL_LABEL = combine.CONTROL_LABEL     # "non-targeting"

SONG = Path("/home/ext1/data/Song25/jurkat_genomewide")
OUT = orion.OUT
VCC26_GENES = orion.VCC26_GENES
FAMILY = "SONG25"
CELL_TYPE = "jurkat"
FILES = [f"{m}_{k}" for m in ("transcriptome", "guides", "labels") for k in ("matrix.mtx.gz", "barcode.tsv.gz", "features.tsv.gz")]


# ---- reading -----------------------------------------------------------------------

def find_samples(root: Path, max_samples: int | None) -> list[tuple[str, str]]:
    """[(GSM id, path prefix up to and including '_channel<N>_')], failing on an incomplete download."""
    samples, incomplete = [], []
    for d in sorted(root.glob("GSM*")):
        tr = list(d.glob("*_transcriptome_matrix.mtx.gz"))
        if not tr:
            incomplete.append(d.name)
            continue
        prefix = str(d / tr[0].name[: -len("transcriptome_matrix.mtx.gz")])
        missing = [f for f in FILES if not Path(prefix + f).exists()]
        if missing:
            incomplete.append(f"{d.name} (missing {missing[0]} ...)")
            continue
        samples.append((d.name, prefix))
    if incomplete:
        raise SystemExit(f"incomplete download, rerun data/download_song.sh: {incomplete}")
    if not samples:
        raise SystemExit(f"no GSM*/ sample folders in {root}")
    return samples[:max_samples]


def read_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["id", "name"], dtype=str)


def read_mtx(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Matrix Market (features x barcodes) -> 0-based feature, barcode, value arrays, n_features, n_barcodes."""
    with gzip.open(path, "rt") as f:
        f.readline()
        f.readline()
        n_feat, n_bc, nnz = map(int, f.readline().split())
    df = pd.read_csv(path, sep=" ", skiprows=3, header=None, names=["f", "c", "v"], dtype=np.int32)
    if len(df) != nnz:
        raise SystemExit(f"{path}: header says {nnz} entries, read {len(df)} (truncated download?)")
    return df["f"].to_numpy() - 1, df["c"].to_numpy() - 1, df["v"].to_numpy(), n_feat, n_bc


def read_barcodes(path: str) -> np.ndarray:
    return pd.read_csv(path, header=None, dtype=str)[0].to_numpy()


# ---- per-lane cell selection ---------------------------------------------------------

def select_lane(gsm: str, prefix: str, guide_code: np.ndarray, guide_names: np.ndarray, code_label: np.ndarray,
                label_cond: np.ndarray, args) -> pd.DataFrame:
    """One row per usable cell of this lane: c (barcode column), target_gene, guide_id, condition, is_control."""
    fT, cT, vT, _, n_bc = read_mtx(prefix + "transcriptome_matrix.mtx.gz")
    umi = np.bincount(cT, weights=vT, minlength=n_bc)
    keep = umi >= args.min_umi
    del fT, cT, vT

    # guides: per cell and target gene, summed guide UMIs
    fg, cg, vg, _, n_bc_g = read_mtx(prefix + "guides_matrix.mtx.gz")
    m = keep[cg]
    g = pd.DataFrame({"c": cg[m], "f": fg[m], "code": guide_code[fg[m]], "v": vg[m]})
    del fg, cg, vg
    gs = g.groupby(["c", "code"], sort=False)["v"].sum().reset_index()
    sig = gs[gs["v"] >= args.min_guide_umi]
    n_sig = sig.groupby("c").size()
    cells = np.flatnonzero(keep)
    n_sig = n_sig.reindex(cells, fill_value=0)

    single = sig[sig["c"].map(n_sig) == 1][["c", "code"]]
    a = single.merge(g, on=["c", "code"])
    top = a.loc[a.groupby("c")["v"].idxmax()]
    tgt = pd.DataFrame({"c": top["c"].to_numpy(), "target_gene": code_label[top["code"].to_numpy()],
                        "guide_id": guide_names[top["f"].to_numpy()]})
    tgt = tgt[tgt["target_gene"].notna()]                       # dropped: knockdowns outside the VCC26 panel
    tgt["is_control"] = tgt["target_gene"] == CONTROL_LABEL
    if args.controls == "noguide":
        tgt = tgt[~tgt["is_control"]]
    if args.controls in ("noguide", "both"):
        none = n_sig.index[n_sig.to_numpy() == 0]
        tgt = pd.concat([tgt, pd.DataFrame({"c": none, "target_gene": CONTROL_LABEL, "guide_id": "none",
                                            "is_control": True})], ignore_index=True)

    # condition hashtag: most UMIs, enough of them and clearly above the runner-up
    fl, cl, vl, _, _ = read_mtx(prefix + "labels_matrix.mtx.gz")
    m = keep[cl]
    d = pd.DataFrame({"c": cl[m], "f": fl[m], "v": vl[m]}).sort_values(["c", "v"], ascending=[True, False])
    d["rank"] = d.groupby("c").cumcount()
    first = d[d["rank"] == 0].set_index("c")
    second = d[d["rank"] == 1].set_index("c")["v"].reindex(first.index, fill_value=0)
    ok = (first["v"] >= args.min_label_umi) & (first["v"] >= args.label_dominance * second)
    cond = pd.Series(label_cond[first["f"].to_numpy()], index=first.index)[ok]

    out = tgt.merge(cond.rename("condition"), left_on="c", right_index=True)
    if args.conditions:
        out = out[out["condition"].isin(args.conditions)]
    out["sample"] = gsm
    print(f"    {gsm}: {int(keep.sum()):,} cells >= {args.min_umi} UMI -> {int(n_sig.eq(1).sum()):,} single-target, "
          f"{int(n_sig.eq(0).sum()):,} no-guide -> {len(out):,} usable ({out['is_control'].sum():,} control)", flush=True)
    return out


def group_of(cond: pd.Series, args) -> pd.Series:
    """Output file/context of each cell."""
    if args.split_conditions:
        return CELL_TYPE + "_" + cond
    return pd.Series(CELL_TYPE, index=cond.index)


def cap_cells(T: pd.DataFrame, args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    T = T.iloc[rng.permutation(len(T))]
    pert = T[~T["is_control"]].groupby(["group", "target_gene"], sort=False).head(args.cells_per_pert)
    pert = pert[pert.groupby(["group", "target_gene"], sort=False)["c"].transform("size") >= args.min_cells_per_pert]
    ctrl = T[T["is_control"]].groupby("group", sort=False).head(args.control_cells)
    return pd.concat([ctrl, pert])


def report(T: pd.DataFrame, min_cells_fp: int = 20) -> list[dict]:
    rows = []
    for grp, t in T.groupby("group"):
        per = t[~t["is_control"]].groupby("target_gene").size()
        rows.append({"group": grp, "cells": len(t), "controls": int(t["is_control"].sum()), "knockdowns": len(per),
                     "median_cells_per_kd": float(per.median()) if len(per) else 0.0,
                     f"kd_ge_{min_cells_fp}_cells": int((per >= min_cells_fp).sum())})
        print(f"   {grp}: {len(t):,} cells | {int(t['is_control'].sum()):,} controls | {len(per):,} knockdowns, "
              f"median {rows[-1]['median_cells_per_kd']:.1f} cells each, {rows[-1][f'kd_ge_{min_cells_fp}_cells']:,} "
              f"with >= {min_cells_fp} (data.pca.min_cells)", flush=True)
    return rows


# ---- writing -------------------------------------------------------------------------

def sample_matrix(prefix: str, cells: np.ndarray, feat_col: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """Raw counts of `cells` (barcode columns, in the given order) over the mapped panel genes."""
    fT, cT, vT, _, n_bc = read_mtx(prefix + "transcriptome_matrix.mtx.gz")
    row_of = np.full(n_bc, -1, dtype=np.int64)
    row_of[cells] = np.arange(len(cells))
    m = row_of[cT] >= 0
    fT, cT, vT = fT[m], cT[m], vT[m]
    col = feat_col[fT]
    m = col >= 0
    X = sp.csr_matrix((vT[m].astype(np.float32), (row_of[cT[m]], col[m])), shape=(len(cells), n_genes))
    X.sort_indices()
    return X


def write_group(name: str, T: pd.DataFrame, samples: dict[str, str], feat_col: np.ndarray, var: pd.DataFrame,
                out_path: Path, chunk_dir: Path, chunk_size: int) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_genes = len(var)
    chunk_paths, buf_X, buf_obs, n_written = [], [], [], 0
    t0 = time.time()

    def flush():
        nonlocal buf_X, buf_obs, n_written
        if not buf_X:
            return
        X = sp.vstack(buf_X, format="csr")
        obs = pd.concat(buf_obs, ignore_index=True)
        obs.index = [f"{FAMILY}:{name}:{n_written + i}" for i in range(len(obs))]
        a = ad.AnnData(X=X, obs=obs, var=var)
        a.layers["lognorm"] = combine.compute_lognorm_layer(X, combine.CPM_TARGET_SUM)
        p = chunk_dir / f"chunk_{n_written:09d}.h5ad"
        a.write_h5ad(p)
        chunk_paths.append(p)
        n_written += X.shape[0]
        buf_X, buf_obs = [], []

    for k, (gsm, sel) in enumerate(T.groupby("sample", sort=True)):
        X = sample_matrix(samples[gsm], sel["c"].to_numpy(), feat_col, n_genes)
        buf_X.append(X)
        buf_obs.append(pd.DataFrame({
            "target_gene": sel["target_gene"].to_numpy(), "guide_id": sel["guide_id"].to_numpy(),
            "batch": gsm, "context": name, "is_control": sel["is_control"].to_numpy(),
            "dataset_family": FAMILY, "source_file": f"{gsm}_transcriptome_matrix.mtx.gz"}))
        if sum(x.shape[0] for x in buf_X) >= chunk_size:
            flush()
        print(f"    {name}: lane {k + 1}/{T['sample'].nunique()}  {n_written:,} cells written  [{time.time() - t0:.0f}s]",
              flush=True)
    flush()
    (chunk_dir / "chunks_done.json").write_text(json.dumps([q.name for q in chunk_paths]))
    orion._finish(chunk_paths, out_path, chunk_dir)


# ---- main ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(SONG))
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--min-umi", type=int, default=2000, help="min transcript UMIs for a barcode to count as a cell")
    ap.add_argument("--min-guide-umi", type=int, default=10, help="guide UMIs (summed over a gene's guides) for a gene to count as present in a cell")
    ap.add_argument("--controls", choices=["nt", "noguide", "both"], default="nt")
    ap.add_argument("--min-label-umi", type=int, default=10)
    ap.add_argument("--label-dominance", type=float, default=3.0)
    ap.add_argument("--conditions", nargs="*", default=None, help="keep only these (activated1..activated6, untreated); default all")
    ap.add_argument("--split-conditions", action="store_true", help="one file/context per condition instead of one pooled")
    ap.add_argument("--cells-per-pert", type=int, default=200)
    ap.add_argument("--control-cells", type=int, default=30000)
    ap.add_argument("--min-cells-per-pert", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=None, help="only the first N lanes (smoke test)")
    ap.add_argument("--stats-only", action="store_true", help="select cells and print the yield; write nothing")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing SONG25__*.h5ad")
    args = ap.parse_args()
    out = Path(args.output_dir)
    t0 = time.time()

    samples_l = find_samples(Path(args.input_dir), args.max_samples)
    samples = dict(samples_l)
    print(f"{len(samples)} lanes: {', '.join(samples)}")

    # ---- features: identical in every lane (one 10x reference, one guide library), checked
    p0 = samples_l[0][1]
    feats, gfeats, lfeats = (read_features(p0 + f"{m}_features.tsv.gz") for m in ("transcriptome", "guides", "labels"))
    bc0 = read_barcodes(p0 + "transcriptome_barcode.tsv.gz")
    for gsm, pre in samples_l:
        for ref, m in ((feats, "transcriptome"), (gfeats, "guides"), (lfeats, "labels")):
            if not read_features(pre + f"{m}_features.tsv.gz").equals(ref):
                raise SystemExit(f"{gsm}: {m} features differ from {samples_l[0][0]}'s")
        for m in ("guides", "labels"):
            if not np.array_equal(read_barcodes(pre + f"{m}_barcode.tsv.gz"), read_barcodes(pre + "transcriptome_barcode.tsv.gz")):
                raise SystemExit(f"{gsm}: {m} barcodes are not in the transcriptome's order")
    del bc0

    vcc26 = pd.read_csv(VCC26_GENES)["gene_name"].astype(str).tolist()
    vset = set(vcc26)
    meta = pd.DataFrame({"gene_token_id": np.arange(len(feats)), "ensembl_id": feats["id"], "gene_name": feats["name"]})
    print(f"Song: {len(meta):,} genes in its reference; VCC26: {len(vcc26):,} genes")
    gmap = orion.build_gene_map(meta, vcc26, orion.load_bridge())
    counts = gmap["method"].value_counts().to_dict()
    print(f"gene map: {len(gmap):,}/{len(vcc26):,} VCC26 genes found in Song  "
          f"(direct {counts.get('direct', 0):,}, via Ensembl {counts.get('ensembl', 0):,}, renames {counts.get('rename', 0):,})")
    name_map = dict(zip(gmap["orion_symbol"], gmap["vcc26_symbol"]))
    feat_col = np.full(len(feats), -1, dtype=np.int64)
    feat_col[gmap["gene_token_id"].to_numpy()] = np.arange(len(gmap))
    var = pd.DataFrame(index=pd.Index(gmap["vcc26_symbol"].tolist()))

    # ---- guide feature -> target gene code; label feature -> condition
    guide_names = gfeats["name"].to_numpy(dtype=object)
    base = pd.Series(guide_names).str.replace(r"_\d+$", "", regex=True)
    guide_code, uniques = pd.factorize(base)
    code_label = np.array([CONTROL_LABEL if u.lower() == "non-targeting" else name_map.get(u, u if u in vset else None)
                           for u in uniques], dtype=object)
    label_cond = lfeats["name"].str.replace(r"_\d+$", "", regex=True).to_numpy(dtype=object)
    print(f"guides: {len(guide_names):,} features -> {len(uniques):,} targets "
          f"({int(pd.notna(code_label).sum()):,} in the VCC26 panel); conditions: {sorted(set(label_cond))}")
    if args.conditions:
        bad = set(args.conditions) - set(label_cond)
        if bad:
            raise SystemExit(f"unknown --conditions {sorted(bad)}; available: {sorted(set(label_cond))}")

    # ---- phase 1: which cells (cheap tables, caps applied across lanes)
    print("\n== selecting cells ...", flush=True)
    parts = [select_lane(gsm, pre, guide_code, guide_names, code_label, label_cond, args) for gsm, pre in samples_l]
    T = pd.concat(parts, ignore_index=True)
    T["group"] = group_of(T["condition"], args)
    T = cap_cells(T, args)
    print(f"\n   after caps ({args.cells_per_pert}/knockdown, {args.control_cells} controls, >= {args.min_cells_per_pert} per knockdown):")
    stats = report(T)
    if args.stats_only:
        print("\n--stats-only: nothing written.")
        return

    # ---- phase 2: write
    out.mkdir(parents=True, exist_ok=True)
    targets = {grp: out / f"{FAMILY}__{grp}.h5ad" for grp in sorted(T["group"].unique())}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"already exist (use --overwrite to replace): {existing}")
    gmap.rename(columns={"orion_symbol": "song_symbol"}).to_csv(out / "song_gene_mapping.csv", index=False)
    all_stats = []
    for grp, out_path in targets.items():
        print(f"\n== {grp}: writing {out_path.name}", flush=True)
        t = T[T["group"] == grp].sort_values(["sample", "c"])
        write_group(grp, t, samples, feat_col, var, out_path, out / f"_chunks_song_{grp}", args.chunk_size)
        all_stats.append({"dataset_family": FAMILY, "source_file": "GSM7951413-28", "output_path": str(out_path),
                          "n_genes_total": int(len(meta)), "n_genes_kept": int(len(var)),
                          **next(s for s in stats if s["group"] == grp), "args": vars(args)})
        print(f"   wrote {out_path}  [{time.time() - t0:.0f}s]", flush=True)

    (out / "song_stats.json").write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\ndone: {len(all_stats)} file(s), stats in {out / 'song_stats.json'}  [{time.time() - t0:.0f}s]\n"
          f"next: make prepare")


if __name__ == "__main__":
    main()
