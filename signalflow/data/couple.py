"""The control -> perturbed coupling, solved once per run configuration.

WHAT THIS REPLACES
    Training used to pick x0 for a target cell by drawing a control at random,
    so the straight line x1 - x0 the flow regresses on was mostly pairing noise.
    This solves, per (context, perturbation), the ENTROPIC OPTIMAL TRANSPORT
    problem that moves the whole control density onto that perturbation's
    density, and reads one source control per perturbed cell off the plan. The
    result is a fixed table, written next to the processed contexts, that
    `data/dataset.py` gathers from at training time -- no solver in the loop.

WHY THE WHOLE DENSITY AND NOT A MATCHING
    A bipartite matching forces distinct sources, which is impossible here:
    every context has far more perturbed cells than controls (1.6x to 25x). The
    Kantorovich plan has no such constraint. With uniform marginals a control
    carries mass 1/n_control and a perturbed cell needs 1/n_pert, so each
    control ends up serving ~n_pert/n_control targets -- many-to-one falls out
    of the marginal constraint instead of being imposed, and the source
    distribution the model trains from stays the true control distribution.

PER PERTURBATION, NOT PER CONTEXT
    Each perturbation moves cells somewhere different. Pooling a context's
    perturbed cells into one target density would be a mixture dominated by
    whichever perturbations have the most cells, and the plan would stop
    answering the question we care about -- for THIS perturbed cell, which
    control plausibly sat where it started. It is also what the model
    conditions on, so the coupling matches the thing being learned.

CONTROL CELLS KEEP THEIR RANDOM SOURCE
    Controls appear as targets too (pert 0), the anchor that pins "no net
    displacement". Transporting controls onto themselves maps each to itself,
    which would collapse the real control-to-control spread that anchor needs,
    so their rows stay -1 here and `dataset.py` draws for them as before.

THE SPACE IS THE PCs, AND ONLY THE PCs
    Distances are measured on each dataset's own HVG-PCA scores (data/hvg.py) alone,
    never on sequencing-depth scalars: perturbed cells often differ from controls in
    depth, so including them would pair cells by library size and make that artifact
    the thing the flow learns to undo.

MEMORY
    One group at a time, and the plan is never formed. Sinkhorn runs on the
    [n_control, n_group] cost matrix (727 MB at the worst group in this data)
    with every reduction done in column blocks, and the readout needs only the
    row potential f: the plan's column j is proportional to exp((f_i - C_ij)/eps),
    so one block of columns at a time is enough to sample from it.

THE SOURCES ARE THE CELLS TRAINING CAN ACTUALLY DRAW
    Not every control is a candidate: a disk-backed context keeps only a seeded
    subset in RAM (`train.control_cache_cells`), and `ContextStore.source_rows`
    promises training never reads a control off disk. So this transports from
    exactly `source_rows()` after replicating the caching calls training makes --
    otherwise most matches would point outside the cache and every batch would
    hit the disk. The subset is a uniform sample of the controls, so the density
    being moved is the same one; it is just restricted to the cells x0 can be.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Column block for every reduction over the cost matrix. Bounds the temporaries
# Sinkhorn would otherwise allocate at full [n_control, n_group] size.
COL_BLOCK = 2048


def _sq_dists(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """[n_src, n_tgt] squared euclidean distances, via ||a||^2 + ||b||^2 - 2a.b."""
    return (
        (src * src).sum(1)[:, None]
        + (tgt * tgt).sum(1)[None, :]
        - 2.0 * (src @ tgt.T)
    ).clamp_min_(0.0)


def _blocked_lse_over_cols(cost: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    """logsumexp_j((g_j - C_ij) / eps) for every row i, accumulated block by block.

    Streaming max/sum rather than one [n_src, n_tgt] temporary.
    """
    n = cost.shape[0]
    run_max = torch.full((n,), -torch.inf, device=cost.device, dtype=cost.dtype)
    run_sum = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    for s in range(0, cost.shape[1], COL_BLOCK):
        blk = (g[None, s : s + COL_BLOCK] - cost[:, s : s + COL_BLOCK]) / eps
        blk_max = blk.max(dim=1).values
        new_max = torch.maximum(run_max, blk_max)
        run_sum = run_sum * torch.exp(run_max - new_max) + torch.exp(blk - new_max[:, None]).sum(1)
        run_max = new_max
    return run_max + torch.log(run_sum.clamp_min(1e-300))


def sinkhorn_potential(
    cost: torch.Tensor, eps: float, iters: int = 200, tol: float = 1e-6
) -> torch.Tensor:
    """Row potential f of the entropic plan pi_ij = exp((f_i + g_j - C_ij) / eps).

    Log-domain, so a small eps does not underflow. Uniform marginals: a_i =
    1/n_src, b_j = 1/n_tgt. Only f is returned -- g is constant down a column,
    so it cannot change which source a column prefers or its relative weights.
    """
    n_src, n_tgt = cost.shape
    log_a = -float(np.log(n_src))
    log_b = -float(np.log(n_tgt))
    f = torch.zeros(n_src, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(n_tgt, device=cost.device, dtype=cost.dtype)

    for _ in range(iters):
        f_prev = f
        # column constraint, block by block: each block reduces over rows only
        for s in range(0, n_tgt, COL_BLOCK):
            blk = (f[:, None] - cost[:, s : s + COL_BLOCK]) / eps
            g[s : s + COL_BLOCK] = eps * (log_b - torch.logsumexp(blk, dim=0))
        # row constraint, streamed across column blocks
        f = eps * (log_a - _blocked_lse_over_cols(cost, g, eps))
        if torch.max(torch.abs(f - f_prev)).item() < tol * eps:
            break
    return f


def _draw_sources(
    cost: torch.Tensor, f: torch.Tensor, eps: float, gen: torch.Generator, argmax: bool
) -> torch.Tensor:
    """One source row per target, read off the plan a column block at a time.

    Sampling (the default) rather than argmax because the MARGINAL is a property
    of the plan, not of its per-column maximum: drawing from column j in
    proportion to pi_ij reproduces the control distribution in expectation,
    while argmax can pile many targets onto a handful of favoured controls.
    """
    out = torch.empty(cost.shape[1], dtype=torch.long, device=cost.device)
    for s in range(0, cost.shape[1], COL_BLOCK):
        logits = (f[:, None] - cost[:, s : s + COL_BLOCK]) / eps      # + g_j: constant per column
        if argmax:
            out[s : s + COL_BLOCK] = logits.argmax(dim=0)
        else:
            probs = torch.softmax(logits, dim=0).T.contiguous()        # [block, n_src]
            out[s : s + COL_BLOCK] = torch.multinomial(probs, 1, generator=gen).squeeze(1)
    return out


def couple_context(
    state: np.ndarray,
    pert: np.ndarray,
    control_rows: np.ndarray,
    n_pcs: int,
    reg: float = 0.05,
    iters: int = 200,
    argmax: bool = False,
    device: torch.device | str = "cpu",
    seed: int = 0,
    progress: str = "",
) -> np.ndarray:
    """int32 [n_cells]: the source control row for every perturbed cell, -1 elsewhere.

    PC SPACE ONLY. `state` is [PCs | log UMI, log genes detected, mean lognorm] and
    only the first `n_pcs` columns are used here. Those three scalars are sequencing
    depth, not biology, and perturbed cells routinely differ from controls in depth --
    transporting on them would couple cells by library size and write that artifact
    into every training pair. They still condition the model (see models/encoders.py);
    they just have no say in who is matched with whom.

    `reg` is relative -- eps = reg * median(cost) per group, so one setting means
    the same thing whatever the scale of the PC space.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    pcs = np.ascontiguousarray(state[:, :n_pcs], dtype=np.float32)
    S = torch.as_tensor(pcs, device=device)
    src = S[torch.as_tensor(control_rows, device=device, dtype=torch.long)]

    match = np.full(len(pert), -1, dtype=np.int32)
    groups = [int(p) for p in np.unique(pert) if p != 0]
    for k, p in enumerate(groups):
        rows = np.flatnonzero(pert == p)
        tgt = S[torch.as_tensor(rows, device=device, dtype=torch.long)]
        cost = _sq_dists(src, tgt)
        eps = max(float(reg) * float(cost.median()), 1e-8)
        f = sinkhorn_potential(cost, eps, iters=iters)
        pick = _draw_sources(cost, f, eps, gen, argmax).cpu().numpy()
        match[rows] = control_rows[pick].astype(np.int32)
        del cost, tgt
        if progress and (k + 1) % 500 == 0:
            print(f"    {progress}: {k + 1}/{len(groups)} groups", flush=True)
    return match


def usage_report(match: np.ndarray, n_control: int) -> str:
    """How well the readout kept the control marginal.

    The benchmark is not "every control used" -- with n draws from n_control
    options even a perfectly uniform readout leaves some untouched. It is the
    distinct count uniform sampling would give, n_control * (1 - (1-1/n)^draws);
    coming in well under that is the collapse we would be worried about.
    """
    used = match[match >= 0]
    if not len(used):
        return "no perturbed cells"
    distinct = len(np.unique(used))
    expected = n_control * (1.0 - (1.0 - 1.0 / n_control) ** len(used))
    counts = np.bincount(used, minlength=n_control)
    return (
        f"{distinct:,} distinct controls used (uniform would give {expected:,.0f}, "
        f"{distinct / expected:.0%} of it), reuse max {int(counts.max())} "
        f"vs mean {len(used) / n_control:.1f}"
    )


# ---- per-dataset artifacts (data.coupling.space: dataset_hvg_pca). One directory per
# CONTEXT, independent of any holdout: fit on that dataset's own cells only, so which
# lines train does not change it, and a held-out line's own pairs are built exactly like
# the others.

PER_DATASET_DIR = "coupling"
OT_KEYS = ("n_hvg", "n_pcs", "hvg_flavor", "scale_genes", "n_sources", "max_fit_cells")


def _fingerprint(c) -> dict:
    st = Path(c._path).stat()
    return {"n_cells": int(len(c.pert)), "n_controls": int(len(c.control_rows)),
            "n_local_genes": int(len(c.gene_idx)), "file_size": int(st.st_size),
            "file_mtime": int(st.st_mtime)}


def source_pool(c, n_sources: int, seed: int) -> np.ndarray:
    """Sorted control rows the per-dataset coupling may pick x0 from: all controls, or a
    fixed seeded subset of `n_sources` (0 = all)."""
    rows = np.sort(c.control_rows.astype(np.int64))
    if n_sources and len(rows) > n_sources:
        rows = np.sort(np.random.default_rng(seed).choice(rows, int(n_sources), replace=False))
    return rows


def build_per_dataset(cfg: dict, exclude=(), reg=0.05, iters=200, argmax=False,
                      device="auto", force=False) -> Path:
    """For every non-excluded context: HVGs over ALL its cells -> per-gene centered (+ scaled,
    clipped) -> PCA -> Sinkhorn coupling of each perturbation's cells to a fixed control pool,
    in those PC scores. Writes match.npy, sources.npy, ot_hvg.npy, ot_pca.npz into
    <processed_dir>/coupling/<context>/, with manifest.json. A context whose manifest already
    matches (same processed file, same settings) is skipped unless `force`.
    """
    from . import hvg
    from .dataset import load_contexts          # local: dataset imports nothing from here

    root = Path(cfg["data"]["processed_dir"])
    seed = int(cfg.get("seed", 0))
    ccfg = hvg.coupling_cfg(cfg)
    base = root / PER_DATASET_DIR
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    want_ot = {"params": {k: ccfg[k] for k in OT_KEYS},
               "solver": {"reg": float(reg), "iters": int(iters), "readout": "argmax" if argmax else "sample"}}

    _, contexts = load_contexts(root, exclude=list(exclude))
    print(f"{base}\n  per-dataset OT ({ccfg['n_hvg']} HVGs -> {ccfg['n_pcs']} PCs) on {len(contexts)} "
          f"context(s), device {device}", flush=True)
    t0 = time.time()
    for k, c in enumerate(contexts):
        out = base / c.name
        man_path = out / "manifest.json"
        fp = _fingerprint(c)
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        if man.get("context") != fp or man.get("seed") != seed:
            man = {"context": fp, "seed": seed}          # the processed file changed: start over
        tag = f"  [{k + 1}/{len(contexts)}] {c.name}"
        if not (force or man.get("ot", {}).get("params") != want_ot["params"]
                or man.get("ot", {}).get("solver") != want_ot["solver"]):
            print(f"{tag}: up to date", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        n, n_local = len(c.pert), len(c.gene_idx)
        read = lambda rows, c=c: c.dense(rows)
        fit_rows = hvg.sample_rows(np.arange(n), ccfg["max_fit_cells"], seed)
        st = hvg.gene_stats(read, fit_rows, n_local)
        hv = hvg.select_hvg(st, int(ccfg["n_hvg"]), ccfg["hvg_flavor"])
        pca = hvg.fit_ot_pca(read, fit_rows, hv, st, int(ccfg["n_pcs"]), bool(ccfg["scale_genes"]))
        pcs = hvg.project_ot(read, n, pca)
        src = source_pool(c, int(ccfg["n_sources"]), seed)
        match = couple_context(pcs, c.pert, src, pcs.shape[1], reg=reg, iters=iters,
                               argmax=argmax, device=device, seed=seed, progress=c.name)
        np.save(out / "match.npy", match)
        np.save(out / "sources.npy", src)
        np.save(out / "ot_hvg.npy", c.gene_idx[hv].astype(np.int64))
        np.savez_compressed(out / "ot_pca.npz", mu=pca["mu"], sd=pca["sd"], m=pca["m"],
                            V=pca["V"], var=pca["var"])
        man["ot"] = {**want_ot, "info": {
            "n_hvg_used": int(len(hv)), "n_pcs_used": int(pcs.shape[1]),
            "var_explained": round(float(pca["var_frac"]), 4),
            "n_fit_cells": int(len(fit_rows)), "n_sources": int(len(src)),
            "n_coupled": int((match >= 0).sum())}}
        print(f"{tag}: OT on {len(hv):,} HVGs -> {pcs.shape[1]} PCs "
              f"({pca['var_frac']:.0%} of HVG variance, fit on {len(fit_rows):,} cells), "
              f"{int((match >= 0).sum()):,} cells coupled from {len(src):,} sources\n"
              f"        {usage_report(match, len(src))}", flush=True)
        man_path.write_text(json.dumps(man, indent=2))
        print(f"        [{time.time() - t1:.0f}s]", flush=True)
    print(f"\nper-dataset stage done  [{time.time() - t0:.0f}s]")
    return base


def load_per_dataset(root: Path, c, cfg: dict) -> dict:
    """The OT coupling of context `c` (match table + source pool), checked against the
    config (the solver settings -- reg/iters/readout -- are not part of the check)."""
    from . import hvg

    d = Path(root) / PER_DATASET_DIR / c.name
    man_path = d / "manifest.json"
    fix = "run `make prepare CONFIG=<this config>`"
    if not man_path.exists():
        raise SystemExit(f"{c.name}: no per-dataset coupling in {d} -- {fix}")
    man = json.loads(man_path.read_text())
    if man.get("context") != _fingerprint(c):
        raise SystemExit(f"{c.name}: {d} was built from a different processed file -- {fix}")
    want = {k: hvg.coupling_cfg(cfg)[k] for k in OT_KEYS}
    if man.get("ot", {}).get("params") != want:
        raise SystemExit(f"{c.name}: OT artifacts built with {man.get('ot', {}).get('params')}, "
                         f"the config asks for {want} -- {fix}")
    return {"match": np.load(d / "match.npy"), "sources": np.load(d / "sources.npy")}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """The basis/coupling flags, shared so `prepare` exposes exactly these."""
    ap.add_argument("--holdout", nargs="*", default=[], metavar="CONTEXT",
                    help="PCA space (data/pca_space.py): contexts kept out of the PCA fit and out of "
                         "the delta fingerprints -- the lines you validate on "
                         "(default: data.pca.holdout, i.e. VCC25__adata_Validation; `none` = nothing, "
                         "for a final `full` model)")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="CONTEXT",
                    help="contexts left out of the run entirely")
    ap.add_argument("--reg", type=float, default=0.05,
                    help="entropic regularisation, relative to each group's median cost "
                         "(smaller = sharper coupling)")
    ap.add_argument("--iters", type=int, default=200, help="max Sinkhorn iterations per group")
    ap.add_argument("--argmax", action="store_true",
                    help="take each column's most likely source instead of sampling it "
                         "(sharper, but can pile targets onto a few controls)")
    ap.add_argument("--device", default="auto")
