"""h5ad -> compact per-context arrays the model can stream.

One .h5 per context (= per source file). Nothing is padded to the global gene
space on disk: each context keeps its own compact matrix plus `gene_idx`, the
map from its local column j to the global readout index. The scatter into
global space (and the mask that goes with it) happens per batch, in
`dataset.py`. That is what keeps this workable when the readout panel grows
from 1,213 genes to 18,533.

ONE COMMAND, FOUR STAGES, EACH SKIPPED WHEN IT IS UP TO DATE
    1. contexts   h5ad -> contexts/<name>.h5 (below). Redone when a source file, the
                  file list or a vocabulary changed (hours for the big files).
    2. coupling   per-dataset HVG PCA + control->perturbed OT (data/couple.py), per
                  context. Independent of any holdout.
    3. PCA space  the shared PCA basis over the model genes + the two perturbation
                  fingerprints (data/pca_space.py), into pca_space/<label>/. It DEPENDS
                  on the held-out lines (`--holdout`, default data.pca.holdout =
                  VCC25__adata_Validation; `none` = nothing, for a final model): they
                  never enter the basis fit or the delta fingerprints. Another holdout is
                  another artifact; stages 1-2 are reused, so that costs minutes.
    4. linear     only with model.other_genes: linear -- per context, every knockdown's
                  mean shift and every measured target's slopes over the controls, for
                  the linear model on the non-model genes (models/linear_genes.py).
                  Holdout-independent, into linear_genes/.
    `--force` redoes stages 1-3.

WHY THIS READS ROW-CHUNKS INSTEAD OF `ad.read_h5ad(f)`
    Some source files are far bigger than any reasonable machine's RAM (one
    prototype context here is 1.9M cells x 5.6B nonzeros, ~150GB loaded the
    naive way, for both `X` and the `lognorm` layer at once). `X` and the
    named `layer` are read directly via h5py, one row-block at a time, off the
    on-disk CSR layout anndata itself writes (`data`/`indices`/`indptr`
    datasets under a group with `encoding-type: csr_matrix`) -- bypassing
    anndata's own (backed-mode) reader for these two arrays entirely, since
    only `.X` is reliably lazy there, not `.layers`. `obs`/`var` stay cheap at
    any cell count, so those are still read via `ad.read_h5ad(f, backed="r")`.
    Everything derived per cell (`lib`, `n_det`, `mean_log`) is a row-local sum
    and chunks trivially; the per-line gene correlations (`gene_corr.py`) use
    a streaming accumulator for the same reason. The output itself streams
    into a resizable HDF5 dataset rather than an in-memory array passed to
    `np.savez_compressed`, because the compact output of that one context is
    itself ~45GB -- too big to exist as a single numpy array on a machine
    this size, streamed input or not.

Layout written to <out>/:

    meta.json               vocab paths, per-context summary
    gene_vocab.csv          readout space  (index = row order)
    pert_vocab.csv          perturbation space, row 0 = "non-targeting"
    contexts/<name>.h5      per context:
        X_data/X_indices/X_indptr/X_shape   lognorm expression, CSR
                             (X_indptr is always int64: a context's nonzero
                             count can exceed int32 range)
        gene_idx     int32 [n_local]     local column -> global gene index
        pert         int32 [n_cells]     -> pert vocab (0 = control)
        is_control   bool  [n_cells]
        lib          f32   [n_cells]     total UMI count (from raw .X)
        scalars      f32   [n_cells, 3]  log1p(total UMI), log1p(genes detected), mean lognorm
        control_rows int32 [n_control]   row ids of control cells
        corr_perts   int32 [k]          pert indices this context carries AND measures
        corr_rows    f16   [k, n_local] PEARSON per-line gene-gene correlation of that
                                        perturbed gene with every local gene, over CONTROL
                                        cells (see gene_corr.py). Not used by the
                                        current (PCA-space) model; kept for a later
                                        correlation conditioning
        gene_effect  f32   [n_local]    per-gene standardized effect size, averaged over
                                        this context's perturbations with enough cells
                                        (gene_corr.PertEffectAccum) -- how much a gene
                                        moves under perturbation relative to its own
                                        control-cell noise. Not used by the current model.
    contexts/<name>.spearman.h5   same corr_perts/corr_rows, SPEARMAN instead of Pearson.
                             Always written alongside the main file; never touched by hand.

Run:
    python -m signalflow.data.prepare --config configs/prototype.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import yaml

from . import couple, gene_corr, pca_space
from ..models import linear_genes
from .vocab import CONTROL_LABEL, GeneVocab, PertVocab

N_SCALAR_STATS = 3

# The streaming conversion itself is memory-bounded (CHUNK_NNZ_TARGET below), but the
# Spearman pass computed alongside it is not: it holds one context's control cells as
# sparse ranks all at once ("a few GB for the densest files" was the budget the old
# stand-alone add_spearman.py script was written around -- but that ran ONE context at a
# time, with nothing else in flight ("run it with little else going on"). Several heavy
# contexts finish the streaming pass at nearly the same time and so tend to start Spearman
# together, which multiplies that cost by the worker count -- this, not CPU count, is the
# default's actual constraint. GB_PER_WORKER is deliberately a pessimistic reading of "a
# few GB" (observed failure: ~7 workers OOM'd a 31GB box), and OS_RESERVE_GB keeps the main
# process and everything else on the machine some headroom instead of budgeting to the byte.
GB_PER_WORKER = 8
OS_RESERVE_GB = 4


def _default_workers(n_files: int) -> int:
    cpu = os.cpu_count() or 1
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        mem_cap = max(1, int((mem_gb - OS_RESERVE_GB) // GB_PER_WORKER))
    except (ValueError, OSError, AttributeError):
        mem_cap = cpu
    return max(1, min(cpu, mem_cap, n_files))

# target nonzeros per streamed row-block (both the source read and the correlation
# accumulator), NOT a target row count: source files vary a lot in density (measured:
# ~2,900 nonzeros/cell for a ~7,700-gene panel vs. ~8,700 nonzeros/cell for an
# 18,077-gene one), and a chunk's memory footprint tracks its nonzero count, not its
# row count. A fixed row count sized for the sparser files silently produced a ~17GB
# chunk (not "well under 1GB") on the denser ones and OOM'd. Not a correctness knob.
CHUNK_NNZ_TARGET = 20_000_000
MIN_CHUNK_CELLS = 1_000

# element count per HDF5 chunk of the resizable X_data/X_indices datasets.
H5_CHUNK_ELEMS = 1_000_000


def _context_name(path: Path) -> str:
    return path.stem.replace("_subset", "")


def _read_obs_var(path: Path):
    """`obs` and `var_names` only -- NOT `ad.read_h5ad(path, backed="r")`, confirmed
    (anndata 0.13.2) to read `layers` fully into memory regardless of `backed`; only
    `.X` is actually lazy there. `anndata.io.read_elem` reads exactly the one group
    it's pointed at, so this never touches `X` or `layers` at all.
    """
    with h5py.File(path, "r") as f:
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
    return obs, var.index.astype(str).tolist()


def _csr_group(h5file: h5py.File, path: str) -> h5py.Group:
    g = h5file[path]
    enc = g.attrs.get("encoding-type")
    if enc != "csr_matrix":
        raise SystemExit(f"{h5file.filename}:{path}: expected a csr_matrix, got {enc!r}")
    return g


def _repaired_indptr(group: h5py.Group) -> np.ndarray:
    """`indptr`, corrected if it carries a specific upstream corruption: some source
    files' on-disk indptr (declared int64) was evidently accumulated with int32
    arithmetic during generation, so once a file's total nonzero count crosses 2**31
    a stretch of it silently wraps (observed in
    REPLOG__K562_gwps_raw_singlecell_01.h5ad, 5.57B nonzeros). indptr must be
    nondecreasing; `np.unwrap` undoes exactly this encoding, since the true values
    are monotonic and the corruption is a +-2**32 discontinuity.
    """
    indptr = group["indptr"][:].astype(np.int64)
    if (np.diff(indptr) >= 0).all():
        return indptr
    fixed = np.rint(np.unwrap(indptr, period=2**32)).astype(np.int64)
    expected_last = int(group["data"].shape[0])
    if not (np.diff(fixed) >= 0).all() or fixed[-1] != expected_last:
        raise SystemExit(
            f"{group.file.filename}:{group.name}: indptr looks corrupted and could not "
            f"be repaired automatically (expected last value {expected_last}, got {fixed[-1]})"
        )
    print(f"  {Path(group.file.filename).name}: repaired an overflow-corrupted indptr in {group.name}")
    return fixed


def _read_chunk(group: h5py.Group, indptr: np.ndarray, start: int, end: int, n_cols: int) -> sp.csr_matrix:
    """One row-block [start:end) of an on-disk CSR group, as an in-memory csr_matrix."""
    lo, hi = int(indptr[start]), int(indptr[end])
    data = group["data"][lo:hi]
    indices = group["indices"][lo:hi]
    local_indptr = indptr[start : end + 1] - lo
    return sp.csr_matrix((data, indices, local_indptr), shape=(end - start, n_cols))


class _ResizableCSR:
    """Growable X_data/X_indices of one context's output .h5, appended chunk by chunk."""

    def __init__(self, h5out: h5py.File) -> None:
        chunk = (H5_CHUNK_ELEMS,)
        self.data_ds = h5out.create_dataset(
            "X_data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=chunk
        )
        self.idx_ds = h5out.create_dataset(
            "X_indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=chunk
        )
        self.nnz = 0

    def append(self, chunk: sp.csr_matrix) -> np.ndarray:
        """Write `chunk`'s data/indices; return its indptr, offset into the whole file."""
        n_new = chunk.data.shape[0]
        self.data_ds.resize(self.nnz + n_new, axis=0)
        self.idx_ds.resize(self.nnz + n_new, axis=0)
        self.data_ds[self.nnz : self.nnz + n_new] = chunk.data.astype(np.float32, copy=False)
        self.idx_ds[self.nnz : self.nnz + n_new] = chunk.indices.astype(np.int32, copy=False)
        offset_indptr = chunk.indptr[1:].astype(np.int64) + self.nnz
        self.nnz += n_new
        return offset_indptr


def _convert_context(f: Path, ctx_idx: int, genes: GeneVocab, perts: PertVocab,
                      layer: str, out: Path) -> dict:
    """One source file -> contexts/<name>.h5 plus its meta.json entry.

    Standalone (not a closure) so it can run in a worker process: everything it
    needs -- the file, its index, the two vocabularies, the layer name, the output
    root -- is passed in and is picklable, and everything it produces either goes
    to its own output file or comes back as the small dict `write_contexts` collects.
    Raised as RuntimeError (not SystemExit) so the message survives the process
    boundary and still stops the run, via the parent's `future.result()`.
    """
    try:
        name = _context_name(f)

        # ---- cheap metadata: obs/var stay small at any cell count ---------
        obs, var_names_all = _read_obs_var(f)
        missing = [g for g in var_names_all if g not in genes]
        if missing:
            keep_mask = np.array([g in genes for g in var_names_all])
            print(f"  {name}: dropping {len(missing)} genes not in gene vocab")
            local_genes = [g for g, k in zip(var_names_all, keep_mask) if k]
        else:
            keep_mask = None
            local_genes = var_names_all
        gene_idx = np.array(genes.indices(local_genes), dtype=np.int32)
        n_local = len(local_genes)

        tg = obs["target_gene"].astype(str).to_numpy()
        unknown = sorted({g for g in tg if g not in perts})
        if unknown:
            raise RuntimeError(
                f"{name}: {len(unknown)} perturbation(s) missing from the pert "
                f"vocab, e.g. {unknown[:5]}. Point data.pert_vocab_csv at a "
                f"gene list that covers them."
            )
        pert = np.array(perts.indices(tg), dtype=np.int32)
        is_control = pert == perts.control_index
        control_rows = np.flatnonzero(is_control).astype(np.int32)
        if len(control_rows) < 2:
            raise RuntimeError(f"{name}: only {len(control_rows)} control cells")

        gene_to_local = {g: j for j, g in enumerate(local_genes)}
        ctx_perts = np.array(sorted({int(p) for p in np.unique(pert) if p != 0}), dtype=np.int32)
        keep_p, cols = gene_corr.local_columns([perts.names[p] for p in ctx_perts], gene_to_local)
        corr_perts = ctx_perts[keep_p] if len(keep_p) else np.zeros(0, dtype=np.int32)
        cols_arr = np.array(cols, dtype=np.int64)
        if len(control_rows) < gene_corr.MIN_CELLS and len(cols):
            print(
                f"  {name}: only {len(control_rows)} control cells -- its gene-gene "
                f"correlations are noisy (< {gene_corr.MIN_CELLS})"
            )

        n_cells = len(obs)

        # ---- the big arrays: streamed row-block by row-block ----------------
        acc = gene_corr.StreamingCorr(n_local, cols_arr)
        effect_acc = gene_corr.PertEffectAccum(n_local, len(perts), ctx_perts)
        ctrl_blocks = []       # this context's control-cell rows only, for the spearman pass below
        lib = np.empty(n_cells, dtype=np.float32)
        n_det = np.empty(n_cells, dtype=np.float32)
        mean_log = np.empty(n_cells, dtype=np.float32)
        out_indptr = np.zeros(n_cells + 1, dtype=np.int64)

        out_path = out / "contexts" / f"{name}.h5"
        with h5py.File(f, "r") as h5in, h5py.File(out_path, "w") as h5out:
            xg = _csr_group(h5in, "X")
            lg = _csr_group(h5in, f"layers/{layer}")
            indptr_x = _repaired_indptr(xg)
            indptr_l = _repaired_indptr(lg)
            if len(indptr_x) - 1 != n_cells or len(indptr_l) - 1 != n_cells:
                raise RuntimeError(f"{name}: X/{layer} row count disagrees with obs")
            n_cols_x = int(xg.attrs["shape"][1])
            n_cols_l = int(lg.attrs["shape"][1])

            # denser panels need smaller row-blocks to hit the same nnz-per-chunk
            # budget -- see CHUNK_NNZ_TARGET
            avg_nnz_per_row = max(int(indptr_x[-1]) / max(n_cells, 1), 1.0)
            chunk_cells = max(MIN_CHUNK_CELLS, int(CHUNK_NNZ_TARGET / avg_nnz_per_row))

            writer = _ResizableCSR(h5out)
            for start in range(0, n_cells, chunk_cells):
                end = min(start + chunk_cells, n_cells)
                raw_chunk = _read_chunk(xg, indptr_x, start, end, n_cols_x)
                log_chunk = _read_chunk(lg, indptr_l, start, end, n_cols_l)
                if keep_mask is not None:
                    raw_chunk = raw_chunk[:, keep_mask]
                    log_chunk = log_chunk[:, keep_mask]

                lib[start:end] = np.asarray(raw_chunk.sum(axis=1)).ravel()
                n_det[start:end] = np.asarray((raw_chunk > 0).sum(axis=1)).ravel()
                # float64 sum, so training and `prediction/predict.py` agree to float32
                # rounding instead of depending on float32 accumulation order
                mean_log[start:end] = (
                    np.asarray(log_chunk.sum(axis=1, dtype=np.float64)).ravel() / n_local
                ).astype(np.float32)

                ctrl_mask = is_control[start:end]
                if ctrl_mask.any():
                    ctrl_chunk = log_chunk[ctrl_mask]
                    acc.add(ctrl_chunk)
                    ctrl_blocks.append(ctrl_chunk)
                effect_acc.add(log_chunk, pert[start:end])

                out_indptr[start + 1 : end + 1] = writer.append(log_chunk)

            scalars = np.stack([np.log1p(lib), np.log1p(n_det), mean_log], axis=1).astype(np.float32)
            corr = acc.finish().astype(np.float16)
            ctrl_mu, ctrl_sd = acc.control_stats()
            gene_effect = effect_acc.finish(ctrl_mu, ctrl_sd)

            h5out.create_dataset("X_indptr", data=out_indptr)
            h5out.create_dataset("X_shape", data=np.array([n_cells, n_local], dtype=np.int64))
            h5out.create_dataset("gene_idx", data=gene_idx)
            h5out.create_dataset("pert", data=pert)
            h5out.create_dataset("is_control", data=is_control)
            h5out.create_dataset("lib", data=lib)
            h5out.create_dataset("scalars", data=scalars)
            h5out.create_dataset("control_rows", data=control_rows)
            h5out.create_dataset("corr_perts", data=corr_perts)
            h5out.create_dataset("corr_rows", data=corr)
            h5out.create_dataset("gene_effect", data=gene_effect)

        # ---- spearman, as a side file next to the main one ------------------
        # Not streamable like the pearson accumulator above -- a rank transform needs
        # every control cell's value for a gene at once -- so the control-cell blocks
        # gathered during the pass above (already the only rows this needs, and already
        # in RAM) are stacked here and ranked in one pass. Always computed, regardless of
        # which `data.correlation` this run's config asks for, so switching configs never
        # needs a re-run: see `dataset.py`, which requires this file when spearman is asked
        # for and now errors instead of silently falling back.
        X_ctrl = sp.vstack(ctrl_blocks, format="csr") if ctrl_blocks else sp.csr_matrix((0, n_local))
        corr_sp = gene_corr.corr_rows(X_ctrl, cols_arr, method="spearman").astype(np.float16)
        side_path = out_path.with_name(out_path.stem + ".spearman.h5")
        side_tmp = side_path.with_name(side_path.name + ".tmp")
        with h5py.File(side_tmp, "w") as sh:
            sh.create_dataset("corr_perts", data=corr_perts)
            sh.create_dataset("corr_rows", data=corr_sp)
            sh.attrs.update(method="spearman", n_control=int(X_ctrl.shape[0]), source=name)
        side_tmp.replace(side_path)

        entry = {
            "index": ctx_idx,
            "name": name,
            "file": f"contexts/{name}.h5",
            "source": str(f),
            "n_cells": int(n_cells),
            "n_local_genes": int(n_local),
            "n_control": int(len(control_rows)),
            "perts": sorted({str(g) for g in tg if g != CONTROL_LABEL}),
            "n_corr_rows": int(len(corr_perts)),
        }
        print(
            f"  {name}: {n_cells} cells, {n_local} genes, "
            f"{len(entry['perts'])} perts, {len(control_rows)} controls, "
            f"corr rows {len(corr_perts)}/{len(ctx_perts)}"
        )
        return entry
    except SystemExit as e:
        raise RuntimeError(str(e)) from e


def _fresh_entries(out: Path, genes: GeneVocab, perts: PertVocab, layer: str) -> dict[str, dict]:
    """source path -> its meta.json entry, for every existing context that can be kept as is:
    same layer and vocabularies as now, context + spearman file present and newer than the source."""
    if not (out / "meta.json").exists() or not (out / "gene_vocab.csv").exists() \
            or not (out / "pert_vocab.csv").exists():
        return {}
    meta = json.loads((out / "meta.json").read_text())
    if meta.get("layer") != layer or GeneVocab.from_csv(out / "gene_vocab.csv").names != genes.names \
            or PertVocab.from_csv(out / "pert_vocab.csv").names != perts.names:
        return {}
    fresh = {}
    for c in meta["contexts"]:
        ctx = out / c["file"]
        side = ctx.with_name(ctx.stem + ".spearman.h5")
        src = Path(c["source"])
        if ctx.exists() and side.exists() and src.exists() and ctx.stat().st_mtime >= src.stat().st_mtime:
            fresh[c["source"]] = c
    return fresh


def write_contexts(cfg: dict, workers: int | None = None, reuse: bool = True) -> Path:
    """Convert the source files to contexts/. With `reuse`, contexts that are already up to
    date (see `_fresh_entries`) are kept and only new or changed source files are converted."""
    data_cfg = cfg["data"]
    src = Path(data_cfg["source_dir"])
    out = Path(data_cfg["processed_dir"])
    (out / "contexts").mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(data_cfg.get("glob", "*.h5ad")))
    if not files:
        raise SystemExit(f"no h5ad files under {src}")

    layer = data_cfg.get("layer", "lognorm")

    # ---- vocabularies -------------------------------------------------
    panels = {}
    for f in files:
        _, var_names = _read_obs_var(f)
        panels[_context_name(f)] = var_names

    gene_list = data_cfg.get("gene_vocab_csv")
    genes = (
        GeneVocab.from_csv(gene_list)
        if gene_list
        else GeneVocab.from_union(panels.values())
    )
    perts = PertVocab.from_csv(data_cfg["pert_vocab_csv"])

    # decided before the vocabularies are (re)written: a kept context's gene_idx / pert
    # indices are only valid against the vocabularies it was converted with
    fresh = _fresh_entries(out, genes, perts, layer) if reuse else {}
    genes.to_csv(out / "gene_vocab.csv", "gene_name")
    perts.to_csv(out / "pert_vocab.csv", "pert")

    # ---- contexts: independent of one another, so run across --workers processes --
    results: dict[int, dict] = {}
    todo = []
    for ctx_idx, f in enumerate(files):
        if str(f) in fresh:
            results[ctx_idx] = {**fresh[str(f)], "index": ctx_idx}
            print(f"  {_context_name(f)}: up to date, kept")
        else:
            todo.append((ctx_idx, f))
    workers = max(1, min(int(workers), len(todo)) if workers else _default_workers(len(todo)))
    print(f"  converting {len(todo)} of {len(files)} context(s) with {workers} worker process(es)")
    if workers <= 1:
        for ctx_idx, f in todo:
            results[ctx_idx] = _convert_context(f, ctx_idx, genes, perts, layer, out)
    elif todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_convert_context, f, ctx_idx, genes, perts, layer, out): ctx_idx
                for ctx_idx, f in todo
            }
            for fut in as_completed(futs):
                ctx_idx = futs[fut]
                try:
                    results[ctx_idx] = fut.result()
                except BrokenProcessPool as e:
                    raise SystemExit(
                        f"a worker died without a Python error -- almost always the OS OOM-killing "
                        f"it (each context can need several GB during the correlation pass; {workers} "
                        f"ran at once here). Re-run with a smaller --workers (e.g. --workers "
                        f"{max(1, workers // 2)}) or WORKERS=<n> via the Makefile. ({e})"
                    ) from e
                except RuntimeError as e:
                    raise SystemExit(str(e)) from e
    contexts = [results[i] for i in range(len(files))]

    meta = {
        "gene_vocab": "gene_vocab.csv",
        "pert_vocab": "pert_vocab.csv",
        "n_genes": len(genes),
        "n_perts": len(perts),
        "n_contexts": len(contexts),
        "n_scalars": N_SCALAR_STATS,
        "layer": layer,
        "contexts": contexts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"\nwrote {out}  |  {len(genes)} readout genes, {len(perts)} pert slots, "
        f"{len(contexts)} contexts"
    )
    return out


def _contexts_stale(cfg: dict) -> str | None:
    """Why stage 1 has to run, or None if contexts/ already matches the sources and vocabularies."""
    d = cfg["data"]
    out = Path(d["processed_dir"])
    if not (out / "meta.json").exists():
        return "no processed contexts yet"
    meta = json.loads((out / "meta.json").read_text())
    files = sorted(Path(d["source_dir"]).glob(d.get("glob", "*.h5ad")))
    if {str(f) for f in files} != {c["source"] for c in meta["contexts"]}:
        return "the set of source files changed"
    for c in meta["contexts"]:
        ctx = out / c["file"]
        side = ctx.with_name(ctx.stem + ".spearman.h5")
        if not ctx.exists() or not side.exists():
            return f"{c['name']}: context file missing"
        if ctx.stat().st_mtime < Path(c["source"]).stat().st_mtime:
            return f"{c['name']}: source file is newer than its context"
    if meta.get("layer") != d.get("layer", "lognorm"):
        return "data.layer changed"
    if d.get("gene_vocab_csv") and GeneVocab.from_csv(d["gene_vocab_csv"]).names != \
            GeneVocab.from_csv(out / "gene_vocab.csv").names:
        return "data.gene_vocab_csv changed"
    if PertVocab.from_csv(d["pert_vocab_csv"]).names != PertVocab.from_csv(out / "pert_vocab.csv").names:
        return "data.pert_vocab_csv changed"
    return None


def prepare(cfg: dict, holdout=(), exclude=(), reg: float = 0.05, iters: int = 200,
            argmax: bool = False, device: str = "auto", workers: int | None = None,
            force: bool = False) -> Path:
    """All preparation: contexts, per-dataset OT coupling, PCA space -- each stage skipped
    when it is already up to date for this config (`force` redoes all three).

    `holdout` names the contexts kept out of the PCA fit and the delta fingerprints
    (empty = data.pca.holdout, ["none"] = nothing). `workers` is how many contexts convert
    in parallel (default `_default_workers`: bounded by CPU count, file count and RAM).
    """
    out = Path(cfg["data"]["processed_dir"])
    why = "--force" if force else _contexts_stale(cfg)
    if why:
        print(f"STAGE 1/4  contexts: converting ({why})")
        write_contexts(cfg, workers=workers, reuse=not force)
    else:
        print(f"STAGE 1/4  contexts: up to date in {out}")
    print("STAGE 2/4  per-dataset OT coupling")
    couple.build_per_dataset(cfg, exclude=exclude, reg=reg, iters=iters, argmax=argmax,
                             device=device, force=force)
    print("STAGE 3/4  PCA space + perturbation fingerprints")
    pca_space.build(cfg, holdout=list(holdout) or None, exclude=exclude, device=device, force=force)
    print("STAGE 4/4  linear-model tables for the non-model genes (model.other_genes)")
    linear_genes.build_tables(cfg, exclude=exclude, device=device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="contexts to convert in parallel (default: bounded by CPU count, "
                         "file count, and available RAM -- lower this if it OOMs)")
    ap.add_argument("--force", action="store_true", help="redo every stage, even the up-to-date ones")
    couple.add_arguments(ap)
    args = ap.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()), holdout=args.holdout,
            exclude=args.exclude, reg=args.reg, iters=args.iters, argmax=args.argmax,
            device=args.device, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
