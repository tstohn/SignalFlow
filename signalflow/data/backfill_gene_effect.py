"""Backfill gene_effect into contexts already converted by an older `prepare`,
without repeating the expensive raw .h5ad -> contexts conversion.

`contexts/<name>.h5` already holds everything gene_corr.PertEffectAccum needs (the
lognorm expression, `pert`, `control_rows`) -- this reads those straight from the
already-converted file via ContextStore.csr(), computes the same per-gene
standardized effect size `prepare.py`'s `_convert_context` would have written, and
appends it as ONE new dataset into the EXISTING file. Nothing else in the file is
read, deleted, or touched.

Run:
    python -m signalflow.data.backfill_gene_effect --config configs/v0_0_0.yaml [--only NAME ...] [--force]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from . import gene_corr
from .dataset import ContextStore, load_contexts

CHUNK = 4096


def compute_gene_effect(c: ContextStore, n_pert_vocab: int) -> np.ndarray:
    """[n_local] float32, streamed over `c`'s own cells -- same maths as
    prepare.py._convert_context, reading from the already-converted file instead
    of the raw source.
    """
    n_local = len(c.gene_idx)
    n_cells = len(c.pert)
    ctx_perts = np.array(sorted({int(p) for p in np.unique(c.pert) if p != 0}), dtype=np.int32)
    # cols=[] : only control_stats() is used from this accumulator, so the (expensive)
    # cross term gene_corr.StreamingCorr would otherwise compute is skipped entirely
    ctrl_acc = gene_corr.StreamingCorr(n_local, np.zeros(0, dtype=np.int64))
    effect_acc = gene_corr.PertEffectAccum(n_local, n_pert_vocab, ctx_perts)
    for start in range(0, n_cells, CHUNK):
        end = min(start + CHUNK, n_cells)
        chunk = c.csr(np.arange(start, end))
        ctrl_mask = c.is_control[start:end]
        if ctrl_mask.any():
            ctrl_acc.add(chunk[ctrl_mask])
        effect_acc.add(chunk, c.pert[start:end])
    mu, sd = ctrl_acc.control_stats()
    return effect_acc.finish(mu, sd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="+", default=None, metavar="CONTEXT", help="just these contexts")
    ap.add_argument("--force", action="store_true", help="recompute even if gene_effect already exists")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pdir = Path(cfg["data"]["processed_dir"])
    # correlation choice is irrelevant here; pearson's corr_rows is always required in the
    # main file (unlike spearman's side file), so this never depends on `make spearman` state
    meta, stores = load_contexts(pdir)

    todo = [c for c in stores if args.only is None or c.name in args.only]
    if args.only:
        missing = set(args.only) - {c.name for c in stores}
        if missing:
            raise SystemExit(f"no such context: {sorted(missing)}")

    t_all = time.time()
    for k, c in enumerate(todo, 1):
        if c.gene_effect is not None and not args.force:
            print(f"[{k}/{len(todo)}] {c.name}: gene_effect already present, skipping (--force to recompute)")
            continue
        t0 = time.time()
        main_path = pdir / next(x["file"] for x in meta["contexts"] if x["name"] == c.name)
        print(f"[{k}/{len(todo)}] {c.name}: computing gene_effect ({len(c.pert):,} cells) ...", flush=True)
        effect = compute_gene_effect(c, meta["n_perts"])
        # load_contexts keeps disk-backed contexts' own read handle open for reuse across the
        # run; done reading from `c`, so close it before reopening the same file for the write
        # (HDF5 refuses a second, read-write open while a read-only one is still live).
        if c._h5 is not None:
            c._h5.close()
        with h5py.File(main_path, "a") as h5:
            if "gene_effect" in h5:
                del h5["gene_effect"]
            h5.create_dataset("gene_effect", data=effect)
        print(f"          wrote gene_effect  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
