"""Train the conditional flow-matching velocity field, in one of three modes.

    python -m signalflow.training.train --config configs/prototype.yaml [--mode holdout|crossval|full]

    holdout    keep ONE whole cell line out as validation. Quick, for everyday model
               changes. Stops early on that line's score (plateau rule).
    crossval   hold each cell line out in turn -> one model per line. Slow but reliable:
               it shows how much the score varies from line to line, and it recommends
               how many epochs the final model should train for.
    full       train on EVERY cell line, no validation. The final model. It trains exactly
               `epochs` epochs (no stopping is possible without held-out data): use the
               epoch count that crossval recommended.

There is no train/val/test split of CELLS. A cell line is either training data or held
out, decided here per run. The model lives in a PCA space that `make prepare`
fits WITHOUT the held-out line(s) (data/pca_space.py); a run uses the artifact whose
held-out set covers its validation line and refuses to start without one, so a new
holdout needs `make prepare HOLDOUT=<line>` first (a final `full` model: HOLDOUT=none). The basis, the model genes and the
delta-fingerprint table are copied next to the checkpoint (`prediction/predict.py`
reads them from there).

WHERE THE RUN CHOICES ARE SET: IN THE YAML, AND ONLY THERE
    `max_epochs`, `full_epochs` and the `early_stop` block (metric, patience,
    window, min_delta) say what a run does. The only flag is `--mode`, which the Makefile
    targets pass. There are no code or Makefile defaults: a missing key is an error, so
    there is no hidden second value.

EPOCHS
    `train.max_epochs` is the MAXIMUM in holdout/crossval: early stopping only ever stops sooner.
    `train.full_epochs` is the exact count in `full` mode. Treat it as a hyperparameter: crossval finds it,
    full uses it.

THE TWO-STAGE MODEL  (`model.centered: true`, stage 2; see the Makefile's TWO-STAGE section)
    Stage 1, the mean model (training/train_mean.py), predicts each knockdown's AVERAGE shift and
    is trained first. This run is stage 2: it loads that model FROZEN (runs/<config>/mean/<same
    holdout folder>/mean.pt, or mean.checkpoint) and trains the flow CENTERED -- its target is
    the perturbed cell minus its knockdown's real average -- so it learns only how cells scatter
    around the average. At scoring/prediction: every cell = source + mean-model average + the
    flow's wiggle with its own group average removed. The mean model is copied into this run's
    folder (mean.pt), so prediction needs only this folder.

EMA  (`train.ema_decay`, e.g. 0.999)
    Held-out loss, cell-eval and every checkpoint use a slowly-updated average of the weights
    ("model" in the .pt); the optimiser's own weights are kept as "model_raw".

WHAT EARLY STOPPING WATCHES  (`train.early_stop.metric`)
    vcc        the cell-eval average on the held-out line: what the challenge scores, but
               noisy and slower, so it is measured every `train.cell_eval_every` epochs and
               smoothed (see `stopping.py`).
    val_loss   the flow-matching loss on the held-out line: cheap, every epoch, but only a
               proxy (mostly irreducible noise), and it need not agree with the metrics.
    The TRAINING loss is deliberately not an option: it falls every epoch whether or not
    the model generalises, so it cannot detect overfitting. It is only used as a guard
    (a non-finite training loss aborts the run).

WHAT A RUN WRITES  (into runs/<config name>/holdout_<line>/, .../crossval/fold_<line>/ or .../full/;
                   train.out_dir overrides runs/<config name>)
    last.pt       the most recent completed epoch (the model to use for `full`)
    best.pt       lowest held-out loss so far           (holdout / crossval)
    best_vcc.pt   highest held-out cell-eval score      (holdout / crossval, cell-eval on)
    pca.npz, model_genes.csv, fingerprints.npz   the PCA basis, the model genes and the
                  delta-fingerprint table over the training lines -- they belong to the model
    run.json      mode, contexts, perturbations trained on, epochs, best epoch/score
    history.json  every epoch's numbers
    tb/           TensorBoard curves (`make tensorboard`)
    train.log     everything printed, plus what is kept off the terminal
Everything is rewritten after EVERY epoch, so Ctrl-C costs at most the epoch in progress.

TWO OUTPUT CHANNELS
    The terminal gets the summary: one line per epoch and, every `cell_eval_every` epochs
    plus the last, a table of the six cell-eval2 VCC26 members with ONE average.
    `train.log` gets all of that PLUS what is kept off the terminal: per-context cell-eval
    detail and cell-eval2's own warnings, tagged with epoch, method and context. cell-eval2
    itself is not modified; its warnings are redirected from our side
    (see `evaluation/cell_eval.captured`).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data import couple, hvg, pca_space
from ..data.dataset import FlowDataset, densify_batch, load_contexts, make_loader
from ..data.vocab import CONTROL_LABEL
from ..models.build import build_model, pick_device
from ..models import linear_genes
from ..models.flow import cfm_loss
from .stopping import PlateauStopper
from .tb import RunLog


_T_START = time.time()


class _EpochBar:
    """`epoch   2 [||||||||.            ]  28%  10,400 cells/s  loss 8.8321  eta 3:12`

    ONE line per epoch, redrawn in place with a carriage return (never a new line), and only
    20 times per epoch (every 5%) so the terminal is not kept busy. Written to the real
    terminal only, so train.log does not fill with progress lines."""

    W, PARTS = 20, 20

    def __init__(self, ep: int, total_steps: int, t0: float) -> None:
        self.ep, self.total, self.t0 = ep, max(total_steps, 1), t0
        self.bucket = -1

    @staticmethod
    def render(ep: int, frac: float, cells_s: float, loss: float, eta: float, W: int = 20) -> str:
        k = min(int(frac * W), W)
        bar = "|" * k + ("." if k < W else "") + " " * max(W - k - 1, 0)
        return (f"epoch {ep:3d} [{bar}] {100 * frac:3.0f}%  {cells_s:>7,.0f} cells/s  "
                f"loss {loss:.4f}  eta {int(eta // 60)}:{int(eta % 60):02d}")

    def update(self, step: int, cells: int, run_t) -> None:
        bucket = step * self.PARTS // self.total
        if bucket == self.bucket:
            return
        self.bucket = bucket
        el = max(time.time() - self.t0, 1e-9)
        line = self.render(self.ep, step / self.total, cells / el, float(run_t) / max(cells, 1),
                           el / max(step, 1) * (self.total - step), self.W)
        sys.__stdout__.write("\r" + line + "   ")
        sys.__stdout__.flush()

    def close(self) -> None:
        sys.__stdout__.write("\r" + " " * 100 + "\r")      # clear the bar; the epoch summary follows
        sys.__stdout__.flush()


def _say(msg: str) -> None:
    """A progress line with the time since the run started, so a long silent phase is never a mystery."""
    t = time.time() - _T_START
    print(f"[{int(t // 60):02d}:{int(t % 60):02d}] {msg}", flush=True)


def to_device(batch, device, space):
    """A sparse BlockStream batch -> model genes -> PCA scores, on the GPU."""
    return densify_batch(batch, device, space)


def _fingerprint_report(train_ctx, val_ctx, K: int) -> None:
    """How many perturbed cells get each fingerprint, per context."""
    print("fingerprints (share of perturbed cells):  delta = other cell lines' mean effect,  "
          "cov = covariance with the target in this line's controls")
    for tag, cs in (("train", train_ctx), ("held-out", val_ctx)):
        for c in cs:
            row = c.fp_lookup[c.pert[c.pert != 0]]
            if not len(row):
                continue
            d = float((c.fp_rows[row, K] > 0).mean())
            v = float((c.fp_rows[row, 2 * K + 1] > 0).mean())
            print(f"  {tag:8s} {c.name:44s} delta {d:6.1%}   cov {v:6.1%}   neither (dropped) "
                  f"{float(((c.fp_rows[row, K] == 0) & (c.fp_rows[row, 2 * K + 1] == 0)).mean()):6.1%}")


def _context_weights(ds: FlowDataset, balance) -> list[float] | None:
    """train.context_balance: per-context loss weight w_c ~ n_c ** -alpha (n_c = the
    context's cells per epoch), scaled so the mean weight per cell is 1. alpha = 1 (`true`):
    every context contributes the same share of the gradient; 0.5: shares grow with sqrt(n_c);
    false / 0: off (shares follow cell counts)."""
    alpha = 1.0 if balance is True else float(balance or 0.0)
    if alpha <= 0:
        return None
    n = np.maximum(np.asarray(ds.epoch_counts, dtype=np.float64), 1.0)
    w = n ** -alpha
    return (w * n.sum() / (w * n).sum()).tolist()


def _print_balance(ds: FlowDataset, w, balance) -> None:
    n = np.asarray(ds.epoch_counts, dtype=np.float64)
    ww = np.ones(len(n)) if w is None else np.asarray(w)
    share = ww * n / (ww * n).sum()
    print(f"epoch composition: pert_cap {ds.pert_cap or 'off'} cells per perturbation, "
          f"context_balance {balance}")
    for c, k, s, x in zip(ds.contexts, n, share, ww):
        print(f"  {c.name:44s} {int(round(k)):>10,} cells/epoch   {k / n.sum():6.1%} of cells   "
              f"{s:6.1%} of the gradient   (weight {x:.2f} per cell)")


@torch.no_grad()
def run_val(model, loader, sigma, device, space) -> dict[str, float]:
    was_training = model.training
    model.eval()
    tot = {"loss": 0.0, "identity_loss": 0.0}
    n = 0
    for batch in loader:
        batch = to_device(batch, device, space)
        _, s = cfm_loss(model, batch, sigma)
        b = batch["z0"].shape[0]
        for k in tot:
            tot[k] += s[k] * b
        n += b
    model.train(was_training)
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out["frac_var_explained"] = 1.0 - out["loss"] / max(out["identity_loss"], 1e-12)
    return out


def _save(path: Path, model, cfg, epoch: int, raw=None, **extra) -> None:
    """`model` = the weights to USE (the EMA copy when train.ema_decay > 0); `raw` = the
    optimiser's own current weights, kept alongside as "model_raw" (only to resume from)."""
    ck = {"model": model.state_dict(), "config": cfg, "epoch": epoch, **extra}
    if raw is not None:
        ck["model_raw"] = raw.state_dict()
    torch.save(ck, path)


def _mean_checkpoint(cfg: dict, run_dir: Path) -> Path:
    """The frozen stage-1 mean model a centered flow run builds on: `mean.checkpoint` if set,
    else runs/<config>/mean/<the same holdout folder>/mean.pt (a crossval fold_<line> uses
    holdout_<line>, the full run uses full)."""
    explicit = (cfg.get("mean") or {}).get("checkpoint")
    if explicit:
        return Path(explicit)
    if run_dir.parent.name == "crossval":
        return run_dir.parent.parent / "mean" / ("holdout_" + run_dir.name[len("fold_"):]) / "mean.pt"
    return run_dir.parent / "mean" / run_dir.name / "mean.pt"


def _gene_dir(cfg: dict, run_dir: Path) -> Path:
    """The stage-1 GENE model folder a centered flow run (model.stage1: gene) builds on:
    `gene.checkpoint` (a folder) if set, else runs/<config>/gene/<the same holdout folder>."""
    explicit = (cfg.get("gene") or {}).get("checkpoint")
    if explicit:
        return Path(explicit)
    if run_dir.parent.name == "crossval":
        return run_dir.parent.parent / "gene" / ("holdout_" + run_dir.name[len("fold_"):])
    return run_dir.parent / "gene" / run_dir.name


def _link(src: Path, dst: Path) -> None:
    """Hard link (no second copy of a big file), copy where that is impossible."""
    import os

    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _own_averages(art, c, space, pcfg, held_out: bool) -> dict[int, np.ndarray]:
    """{knockdown: its REAL average PC shift in context c} -- the centered flow's targets. A training
    line's come from `prepare`'s delta table; the held-out line's are computed here, and only
    ever used to center its held-out LOSS (never as an input, never for a prediction)."""
    if not held_out and c.name in art.delta:
        return art.delta[c.name]
    d = pca_space.delta_table(c, space, c.model_col, int(pcfg["min_cells"]))
    return dict(zip(d["perts"].tolist(), d["fp"]))


class _Tee:
    """Everything printed goes to the terminal AND the log file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def _logged(run_dir: Path, header: str):
    """Tee stdout into `<run_dir>/train.log` (appended to, never overwritten) and yield a
    `to_log(line)` that writes to the log file only."""
    logf = open(run_dir / "train.log", "a", buffering=1)
    logf.write(f"\n=== {header}  {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")

    def to_log(line: str) -> None:
        logf.write(f"{datetime.now():%H:%M:%S} {line}\n")

    terminal = sys.stdout
    sys.stdout = _Tee(terminal, logf)
    try:
        yield to_log
    finally:
        sys.stdout = terminal
        logf.close()


def train_run(
    cfg: dict,
    meta: dict,
    contexts: list,
    train_names: list[str],
    val_names: list[str],
    run_dir: Path,
    epochs: int,
    mode: str,
    stop: dict | None,
    device_arg: str | None = None,
) -> dict:
    """One training run. `val_names` empty means no validation (full mode).

    Returns a summary dict (also written to run.json).
    """
    tcfg, mcfg = cfg["train"], cfg["model"]
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    header = f"{mode}" + (f"  holding out {', '.join(val_names)}" if val_names else "  (all contexts)")
    with _logged(run_dir, f"train {header}") as to_log:
        return _train_run(cfg, meta, contexts, train_names, val_names, run_dir, epochs, mode,
                          stop, device_arg, to_log, tcfg, mcfg, seed)


def _train_run(cfg, meta, contexts, train_names, val_names, run_dir, epochs, mode,
               stop, device_arg, to_log, tcfg, mcfg, seed) -> dict:
    train_ctx = [c for c in contexts if c.name in train_names]
    val_ctx = [c for c in contexts if c.name in val_names]
    all_ctx = train_ctx + val_ctx
    device = pick_device(device_arg or tcfg.get("device", "auto"))
    bs = int(tcfg.get("batch_size", 256))
    sigma = float(tcfg.get("sigma", 0.0))
    n_perts = meta["n_perts"]
    root = Path(cfg["data"]["processed_dir"])
    t0 = time.time()

    # ---- control -> perturbed pairing: the per-dataset OT tables `prepare` wrote
    _say("SETUP 2/5  loading the per-dataset OT coupling `prepare` wrote, caching its control pools...")
    ccfg = hvg.coupling_cfg(cfg)
    for c in all_ctx:
        art_c = couple.load_per_dataset(root, c, cfg)
        c.set_sources(art_c["sources"])
        c.match = art_c["match"]
    print(f"per-dataset OT coupling attached for {len(all_ctx)} context(s) "
          f"({ccfg['n_hvg']} HVGs -> {ccfg['n_pcs']} PCs each)", flush=True)

    # ---- the PCA space and the perturbation fingerprints: `make prepare`, fit
    # WITHOUT the held-out line(s) -- `find` refuses any artifact that saw them
    pcfg = pca_space.pca_cfg(cfg)
    art_dir = pca_space.find(root, [c.name for c in val_ctx], cfg, all_ctx, full=(mode == "full"))
    _say(f"SETUP 3/5  PCA space + fingerprints from {art_dir}")
    art = pca_space.Artifact(art_dir)
    space = art.space
    K = space.n_pcs
    train_names_ = [c.name for c in train_ctx]
    sc = pca_space.scales(art, train_names_)
    for c in all_ctx:
        c.model_col = pca_space.model_cols(c.gene_idx, space.gene_idx, c.name)
        c.fp_lookup, c.fp_rows = pca_space.context_fingerprints(
            art, c.name, np.unique(c.pert), train_names_, n_perts, sc, pcfg)
    # cell-state conditioning: every control cell's k-NN mean in the PCA space, precomputed
    # by `make prepare` (controls only, so the held-out line has it too)
    state_k = int(pcfg["state_knn"])
    if state_k > 0:
        for c in all_ctx:
            c.state_rows, c.state = art.state(c.name, state_k)
        print(f"cell state: mean of each source cell's {state_k} nearest controls (data.pca.state_knn)")
    pert_names = [str(x) for x in np.loadtxt(root / "pert_vocab.csv", dtype=str, delimiter="\t", skiprows=1)]
    pca_space.save_for_run(run_dir, art, train_names_, sc, pert_names)

    # the genes the flow does not predict: copied from the source cell, or the linear
    # population-mean model (model.other_genes), fit on THIS run's training lines only
    # ... and the anchored model (model.anchor) starts from the same linear fit on the model genes
    other_genes = linear_genes.mode(cfg)
    anchored = bool(mcfg.get("anchor", False))
    other_shift, lin_coef, anchor_cfg = None, None, None
    if other_genes == "linear" or anchored:
        _say("          fitting the linear population-mean model on the training lines "
             f"({'non-model genes' if other_genes == 'linear' else ''}"
             f"{' + ' if other_genes == 'linear' and anchored else ''}{'anchor start gains' if anchored else ''})...")
        lin_coef, D_lin, S_lin = linear_genes.fit_for_run(cfg, train_ctx, all_ctx, meta["n_genes"], device)
        if other_genes == "linear":
            other_shift = {c.name: linear_genes.heldout_shifts(c, train_ctx, lin_coef, D_lin, S_lin,
                                                               meta["n_genes"])[0] for c in val_ctx}
            linear_genes.save_for_run(run_dir, train_ctx, D_lin, lin_coef, meta["n_genes"], pert_names)
        del D_lin, S_lin
    if anchored:
        for c in all_ctx:
            c.anchor_s, c.anchor_m = pca_space.context_anchor(art, c.name, c.fp_lookup, len(c.fp_rows),
                                                              train_names_, pcfg)
        anchor_cfg = {"gain0": [float(x) for x in lin_coef[:3]], "scale_delta": sc["delta"]}
        print("anchored model: v = g_d*D + g_s*S + g_m*M + r; start gains "
              f"g_d={lin_coef[0]:+.3f} g_s={lin_coef[1]:+.3f} g_m={lin_coef[2]:+.3f} (the linear fit), learned from there")
    # ---- the TWO-STAGE model (model.centered): the flow learns only how cells scatter around each
    # knockdown's average; the average comes from the FROZEN stage-1 mean model (train_mean.py)
    centered = bool(mcfg.get("centered", False))
    stage1 = str(mcfg.get("stage1", "mean"))
    if stage1 not in ("mean", "gene"):
        raise SystemExit(f"model.stage1 must be 'mean' or 'gene', got {stage1!r}")
    mean_net, mean_info, gene_names, gene_dir = None, None, None, None
    if centered and stage1 == "gene":
        # stage 1 = the GENE MODEL (train_gene.py): its held-out predictions were saved by that run
        if anchored:
            raise SystemExit("model.centered and model.anchor exclude each other. Set model.anchor: false.")
        gene_dir = _gene_dir(cfg, run_dir)
        if not (gene_dir / "gene.pt").is_file():
            raise SystemExit(f"\nmodel.stage1: gene: no gene model in {gene_dir}.\n-> train it first: "
                             f"`make gene-holdout HOLDOUT={val_names[0] if val_names else '...'}` "
                             f"(or `make gene-full` for a full run), or set gene.checkpoint.")
        g_info = json.loads((gene_dir / "run.json").read_text())
        leak = set(g_info.get("train_contexts", [])) & set(val_names)
        if leak:
            raise SystemExit(f"{gene_dir} was trained ON the held-out line(s) {sorted(leak)}: it would leak the answer")
        for vn in val_names:
            if not (gene_dir / f"heldout_pred_{vn}.npz").is_file():
                raise SystemExit(f"{gene_dir} has no predictions for {vn}: train the gene model holding out {vn}")
        for f in ("gene.pt", "gene_tables.npz"):
            _link(gene_dir / f, run_dir / f)              # prediction reads them from the flow's folder
        print(f"TWO-STAGE model: gene model {gene_dir} (best epoch {g_info.get('best_epoch')}) owns the "
              f"average (move only); this flow is CENTERED and adds the scatter around it")
    elif centered:
        from ..models import mean_model as mm

        if anchored:
            raise SystemExit("model.centered and model.anchor exclude each other: the centered flow has no "
                             "average to anchor (the mean model owns it). Set model.anchor: false.")
        mpath = _mean_checkpoint(cfg, run_dir)
        if not mpath.is_file():
            raise SystemExit(f"\nmodel.centered: no stage-1 mean model at {mpath}.\n-> train it first: "
                             f"`make mean-holdout HOLDOUT={val_names[0] if val_names else '...'}` "
                             f"(or `make mean-full` for a full run), or set mean.checkpoint.")
        mean_net, mean_info = mm.load(mpath, device)
        leak = set(mean_info.get("train_contexts", [])) & set(val_names)
        if leak:
            raise SystemExit(f"{mpath} was trained ON the held-out line(s) {sorted(leak)}: it would leak the answer")
        if set(mean_info.get("train_contexts", [])) != set(train_names_):
            print(f"NOTE: the mean model was trained on {len(mean_info.get('train_contexts', []))} contexts, this flow "
                  f"on {len(train_names_)} -- fine, but they are not the same set")
        shutil.copy(mpath, run_dir / "mean.pt")              # prediction reads it from the flow's folder
        import pandas as pd

        gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
        print(f"TWO-STAGE model: frozen mean model {mpath} (best epoch {mean_info.get('best_epoch')}) owns the "
              f"average; this flow is CENTERED (target = perturbed cell - its knockdown's real average)")
    if centered:
        _say("          centering targets: every knockdown's real average in its own context...")
        for c in all_ctx:
            own = _own_averages(art, c, space, pcfg, held_out=c in val_ctx)
            c.center_rows = np.zeros((len(c.fp_rows), K), dtype=np.float32)
            c.center_ok = np.zeros(len(c.fp_rows), dtype=bool)
            c.center_ok[0] = True                                # controls: nothing to subtract
            for p in np.flatnonzero(c.fp_lookup):
                if int(p) in own:
                    c.center_rows[c.fp_lookup[p]] = own[int(p)]
                    c.center_ok[c.fp_lookup[p]] = True
    flow_method = "gene_flow" if centered and stage1 == "gene" else "flow_linear" if other_genes == "linear" else "flow"
    print(f"PCA space: {len(space.gene_idx):,} model genes, {K} PCs "
          f"({art.manifest['var_explained']:.1%} of their variance), fit without "
          f"{art.manifest['holdout'] or 'nothing'}; fingerprint scales delta {sc['delta']:.3g}, cov {sc['cov']:.3g}")
    _fingerprint_report(train_ctx, val_ctx, K)

    trained_perts = sorted({int(p) for c in train_ctx for p in np.unique(c.pert) if p != 0})
    train_ds = FlowDataset(train_ctx, seed=seed, nodelta_cap=tcfg.get("nodelta_cap"),
                           pert_cap=tcfg.get("pert_cap"))
    ctx_w = _context_weights(train_ds, tcfg.get("context_balance", False))
    if train_ds.pert_cap or ctx_w is not None:
        _print_balance(train_ds, ctx_w, tcfg.get("context_balance", False))
    dropped = {n: k for n, k in train_ds.n_dropped.items() if k}
    if dropped:
        print(f"dropped {sum(dropped.values()):,} training cells with NEITHER fingerprint: "
              + ", ".join(f"{n} {k:,}" for n, k in dropped.items()))
    if train_ds.keep_frac < 1.0:
        print(f"nodelta_cap {tcfg.get('nodelta_cap')}: {train_ds.n_nodelta:,} training cells have only the "
              f"covariance fingerprint; a fresh {train_ds.keep_frac:.0%} of them each epoch "
              f"(~{round(train_ds.keep_frac * train_ds.n_nodelta):,} of {len(train_ds):,} cells per epoch)")

    val_ds = None
    if val_ctx:
        # the held-out line is judged on the perturbations the training lines carry (controls always count)
        val_ds = FlowDataset(val_ctx, seed=seed + 1, keep_perts=np.array(trained_perts, dtype=np.int64))
        n_pert_cells = sum(int((~c.is_control[val_ds.rows[i]]).sum()) for i, c in enumerate(val_ctx))
        if n_pert_cells == 0:
            print("WARNING: no perturbation of the held-out line appears in the training lines; "
                  "falling back to ALL held-out cells so the run can proceed.")
            val_ds = FlowDataset(val_ctx, seed=seed + 1)

    ldr = dict(num_workers=int(tcfg.get("num_workers", 4)),
               block_cells=int(tcfg.get("block_cells", 1024)),
               blocks_per_task=int(tcfg.get("blocks_per_task", 4)),
               prefetch_factor=int(tcfg.get("prefetch_factor", 2)),
               pin_memory=device.type == "cuda",
               persistent_workers=bool(tcfg.get("persistent_workers", False)))
    _say(f"          starting {ldr['num_workers']} loader worker process(es), batch size {bs} ...")
    train_dl = make_loader(train_ds, bs, seed, **ldr)
    val_dl = (make_loader(val_ds, bs, seed, shuffle=False, **{**ldr, "num_workers": int(tcfg.get("val_workers", 2))})
              if val_ds is not None else None)
    tspace = pca_space.TorchSpace(space, device)

    model = build_model(K, space.pc_sd, mcfg, state=state_k > 0, anchor=anchor_cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    # train.ema_decay: a slowly-updated AVERAGE of the weights, which is what is validated, scored and
    # saved ("model" in every checkpoint). The raw weights jump with every single-context batch
    # (runs/v1_0_2: held-out loss 2.06 -> 2.40 -> 2.09 on neighbouring epochs); the average does not.
    ema_decay = float(tcfg.get("ema_decay", 0.0) or 0.0)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay > 0 else None
    ev = ema if ema is not None else model                    # the model everything is judged on
    ema_p, model_p = (list(ema.parameters()), list(model.parameters())) if ema is not None else ([], [])
    ema_step = 0
    print(
        f"device={device}  model genes={len(space.gene_idx)}  PCs={K}  contexts: {len(train_ctx)} train"
        f"{'' if val_ds is None else f' / {len(val_ctx)} held out'}\n"
        f"train cells/epoch={len(train_ds)}" + ("" if val_ds is None else f"  held-out cells={len(val_ds)}")
        + f"  params={n_par/1e6:.2f}M  max epochs={epochs}  [{time.time() - t0:.0f}s]\n"
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    # Linear warmup (peak lr is tcfg["lr"], reached at warmup_steps) into cosine decay down to
    # lr_min over the rest of the run. Warmup matters more here than in a generic setup: several
    # pieces of the model start deliberately zero (FiLM's gamma/beta, the zero-inited
    # output head) and only grow a nonzero gradient signal once the rest of the network gives them
    # one -- taking a full-size AdamW step against that on step 1, with second-moment estimates
    # still noisy, is exactly the kind of instability the config's own lr note describes (1e-3
    # overshot badly). warmup_steps=0 recovers the old bare-cosine schedule exactly.
    total_steps = max(epochs * len(train_dl), 1)
    warmup_steps = min(int(tcfg.get("warmup_steps", 500)), max(total_steps - 1, 0))
    lr_min = float(tcfg.get("lr_min", 1e-6))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps - warmup_steps, 1), eta_min=lr_min
    )
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        sched = cosine

    run_info = {
        "mode": mode, "n_pcs": K, "n_model_genes": int(len(space.gene_idx)),
        "pca_space": str(art_dir), "pca_holdout": art.manifest["holdout"],
        "fingerprint_scales": sc, "nodelta_keep_frac": train_ds.keep_frac, "state_knn": state_k,
        "pert_cap": train_ds.pert_cap, "context_weights": None if ctx_w is None else
        {c.name: w for c, w in zip(train_ctx, ctx_w)},
        "other_genes": other_genes, "linear_coef": None if lin_coef is None else [float(x) for x in lin_coef],
        "anchor": anchor_cfg, "mag_max": model.mag_max, "ema_decay": ema_decay,
        "centered": centered, "stage1": stage1 if centered else None,
        "mean_checkpoint": str(run_dir / "mean.pt") if centered and stage1 == "mean" else None,
        "gene_model": str(gene_dir) if gene_dir is not None else None,
        "coupling": ccfg,
        "train_contexts": [c.name for c in train_ctx], "held_out": [c.name for c in val_ctx],
        "trained_perts": [pert_names[p] for p in trained_perts],
        "max_epochs": epochs,
        "stop": stop,                      # the exact stopping settings this run used
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))

    # ---- what to watch, and whether we can ---------------------------------------------
    every = int(tcfg.get("cell_eval_every", 0))
    metric = stop["metric"] if stop else None
    scorer, ref, mean_dz, gene_pred = None, None, None, None
    if val_ds is not None and every > 0:
        from ..evaluation.cell_eval import ValScorer, format_reference, summarize

        _say("SETUP 5/5  cell-eval: reading the held-out cells and the cross-line shifts, then scoring the "
             "two reference baselines (identity, mean_shift) ONCE with cell-eval2...")
        print("cell-eval on the held-out line:")
        scorer = ValScorer(
            cfg, train_ctx, val_ds, device, space,
            n_steps=int(tcfg.get("cell_eval_steps", 20)),
            min_cells=int(tcfg.get("cell_eval_min_cells", 5)),
            max_cells=int(tcfg.get("cell_eval_max_cells", 0)),
            log=to_log,
        )
        if scorer.ctx:
            print(f"  {scorer.n_contexts} context(s), {scorer.n_perts} perturbations, "
                  f"{scorer.n_cells:,} reference cells   "
                  f"(detail and cell-eval2's notices go to {run_dir / 'train.log'} only)")
            _say("          scorer ready; scoring the reference baselines (about a minute)...")
            ref_df = scorer.score(model, methods=("identity", "mean_shift"), tag="reference | ")
            ref = {m: summarize(ref_df, m) for m in ("identity", "mean_shift")}
            if centered and stage1 == "gene":
                # the gene model's saved predictions for the held-out line, scored ALONE once here
                # (move + stretch, and move only): the bars the flow has to clear
                gene_pred = {}
                for vc in val_ctx:
                    z = np.load(gene_dir / f"heldout_pred_{vc.name}.npz")
                    if not np.array_equal(z["gene_idx"], vc.gene_idx):
                        raise SystemExit(f"{gene_dir}: its {vc.name} predictions are over a different gene panel")
                    at = {int(p): i for i, p in enumerate(z["perts"])}
                    ps = scorer.perts().get(vc.name, [])
                    gone = [p for p in ps if p not in at]
                    if gone:
                        raise SystemExit(f"{gene_dir}: no prediction for {len(gone)} scored knockdowns of {vc.name}; "
                                         f"re-run `make gene-holdout`")
                    gp = {"mu": z["mu"]}
                    gp.update({p: (z["pred"][at[p], :, 0].astype(np.float32), z["pred"][at[p], :, 1].astype(np.float32))
                               for p in ps})
                    gene_pred[vc.name] = gp
                for gmeth in ("gene", "gene_mean"):
                    ref[gmeth] = summarize(scorer.score(None, methods=(gmeth,), tag="reference | ",
                                                        gene_pred=gene_pred), gmeth)
            elif centered:
                # the frozen mean model's averages for the held-out knockdowns, from the held-out line's
                # own controls; scored ALONE once here: the bar the flow has to clear
                mean_dz = {}
                for vc in val_ctx:
                    ps = scorer.perts().get(vc.name, [])
                    if ps:
                        inp = mm.context_inputs(art, vc, ps, train_names_, pcfg, pert_names, gene_names,
                                                int(mm.mean_cfg(mean_info["config"])["ctrl_cells"]), seed)
                        pr = mm.predict(mean_net, inp, device)
                        mean_dz[vc.name] = {p: pr[i] for i, p in enumerate(ps)}
                mean_method = "mean_linear" if other_genes == "linear" else "mean"
                ref[mean_method] = summarize(scorer.score(None, methods=(mean_method,), tag="reference | ",
                                                          other_shift=other_shift, mean_dz=mean_dz), mean_method)
            print(format_reference(ref) + "\n")
        else:
            print("  the held-out line has too few cells per (seen) perturbation for cell-eval; it is off\n")
            scorer = None
    watch_vcc = scorer is not None and metric == "vcc"
    if val_ds is not None and metric == "vcc" and not watch_vcc:
        print("NOTE: early stopping falls back to the held-out LOSS, because cell-eval is unavailable.\n")

    stopper = PlateauStopper(
        patience=stop["patience"] if stop else 0,
        min_delta=stop["min_delta"] if stop else 0.0,
        window=stop["window"] if stop else 1,
    )

    tb = RunLog(run_dir)
    history, best_loss = [], float("inf")
    best_vcc, best_vcc_ep, ep = float("-inf"), 0, 0
    vcc_track: list[tuple[int, float]] = []
    stopped_early, diverged = False, False

    # Ctrl-C is safe: everything is written after EVERY epoch.
    try:
        from ..evaluation.cell_eval import format_overview, format_report, summarize  # noqa: F811  (no-op if cell-eval is off)
    except Exception:  # pragma: no cover
        pass
    n_steps_ep = len(train_dl)
    print("\n" + "=" * 78)
    if ema is not None:
        print(f"EMA weights (train.ema_decay {ema_decay}): held-out loss, cell-eval and every checkpoint use the "
              f"averaged weights")
    _say(f"START TRAINING   up to {epochs} epochs x {n_steps_ep:,} steps of {bs} cells "
         f"({len(train_ds):,} training cells), lr {tcfg.get('lr')} "
         f"(warmup {warmup_steps} steps, cosine to {lr_min:.1e}), device {device}")
    print("  every epoch shows a progress bar; a summary line follows it, and a cell-eval overview every "
          f"{every if every else '(off)'} epochs")
    print("=" * 78, flush=True)
    try:
        for ep in range(1, epochs + 1):
            t0 = time.time()
            # the running loss stays ON the GPU: a .item() per step would make the CPU wait for the
            # GPU every step (measured: up to 25% slower); it is read back only when printed
            run_t, n = torch.zeros((), device=device), 0
            gain_t, n_g = torch.zeros(3, device=device), 0      # anchored model: mean gains this epoch
            mag_t, n_m = torch.zeros((), device=device), 0      # mean ||r|| magnitude this epoch (model.mag_max cap)
            bar = _EpochBar(ep, n_steps_ep, t0)
            train_dl.dataset.set_epoch(ep - 1)      # workers are fresh each epoch: tell them which one it is
            for step_i, batch in enumerate(train_dl, 1):
                batch = to_device(batch, device, tspace)
                loss, _ = cfm_loss(model, batch, sigma)
                opt.zero_grad(set_to_none=True)
                # train.context_balance: a batch is one context, weighted by that context's
                # weight; the clip threshold scales with it, so clipping does not undo the weight
                w = 1.0 if ctx_w is None else ctx_w[int(batch["ctx"][0])]
                (loss * w).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), w)
                opt.step()
                sched.step()
                if ema is not None:
                    ema_step += 1
                    d = min(ema_decay, (1 + ema_step) / (10 + ema_step))    # short warm-up of the average
                    with torch.no_grad():
                        torch._foreach_lerp_(ema_p, model_p, 1.0 - d)
                b = batch["z0"].shape[0]
                run_t += loss.detach() * b
                n += b
                if anchor_cfg is not None:
                    gain_t += model.last_gain
                    n_g += 1
                mag_t += model.last_mag
                n_m += 1
                bar.update(step_i, n, run_t)
            bar.close()

            tr = float(run_t) / max(n, 1)
            entry = {"epoch": ep, "train_loss": tr}
            if n_g:
                gd, gs, gm = (gain_t / n_g).tolist()
                entry.update({"gain_d": gd, "gain_s": gs, "gain_m": gm})
                print(f"          mean anchor gains this epoch: g_d {gd:+.3f}  g_s {gs:+.3f}  g_m {gm:+.3f}")
            if n_m:
                mag = float(mag_t / n_m)
                entry["mag_mean"] = mag
                print(f"          mean residual magnitude ||r|| this epoch: {mag:.2f}  (cap {model.mag_max:.0f})")
            if not np.isfinite(tr):
                print(f"\nepoch {ep}: the training loss is {tr}. Stopping: the run has diverged.")
                diverged = True
                break

            if val_dl is not None:
                va = run_val(ev, val_dl, sigma, device, tspace)
                entry.update({f"val_{k}": v for k, v in va.items()})
                print(
                    f"epoch {ep:3d}  train {tr:.4f}  held-out {va['loss']:.4f}  "
                    f"(identity {va['identity_loss']:.4f}, "
                    f"explained {va['frac_var_explained']*100:5.1f}%)  "
                    f"{time.time()-t0:.1f}s"
                )
                if va["loss"] < best_loss:
                    best_loss = va["loss"]
                    _save(run_dir / "best.pt", ev, cfg, ep, raw=model if ema is not None else None, val=va)
                # the plateau rule watches the score when cell-eval drives it (measured only
                # on scoring epochs), otherwise the held-out loss (every epoch)
                if not watch_vcc:
                    stopper.update(-va["loss"], step=ep)
            else:
                print(f"epoch {ep:3d}  train {tr:.4f}  {time.time()-t0:.1f}s")

            if scorer is not None and (ep % every == 0 or ep == epochs):
                t1 = time.time()
                s = summarize(scorer.score(ev, methods=(flow_method,), tag=f"epoch {ep} | ",
                                           other_shift=other_shift, mean_dz=mean_dz, gene_pred=gene_pred), flow_method)
                entry.update({f"vcc_{k}": v for k, v in s.items()})
                new_best = bool(np.isfinite(s["avg_score"]) and s["avg_score"] > best_vcc)
                if new_best:
                    best_vcc, best_vcc_ep = s["avg_score"], ep
                    _save(run_dir / "best_vcc.pt", ev, cfg, ep, raw=model if ema is not None else None, vcc=s)
                if watch_vcc:
                    stopper.update(s["avg_score"], step=ep)
                    entry["vcc_avg_smooth"] = stopper.smoothed
                note = ("new best -> best_vcc.pt" if new_best
                        else f"best {best_vcc:+.3f} at epoch {best_vcc_ep}")
                if stopper.patience and watch_vcc:
                    note += (f"   plateau watch {stopper.bad}/{stopper.patience}"
                             f" (smoothed {stopper.smoothed:+.3f}, best smoothed {stopper.top:+.3f}"
                             f" at epoch {stopper.top_step}, must exceed {stopper.needs:+.3f} to count as progress)")
                vcc_track.append((ep, s["avg_score"]))
                print(format_overview(s, ref, f"cell-eval (held out) after epoch {ep}   [{time.time()-t1:.0f}s]", note=note))
                to_log(format_report(          # the detailed table (raw + scaled + contexts): log file only
                    s, f"cell-eval (held out) after epoch {ep}", ref_avg=ref["mean_shift"]["avg_score"], note=note))
            history.append(entry)
            tb.epoch(entry, lr=opt.param_groups[0]["lr"])
            if "vcc_avg_score" in entry:
                tb.baselines(ref, ep)

            _save(run_dir / "last.pt", ev, cfg, ep, raw=model if ema is not None else None)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

            if stopper.should_stop:
                what, sign = ("cell-eval score", 1) if watch_vcc else ("held-out loss", -1)
                print(f"\nplateau: the {what}, smoothed over the last {stopper.window} measurement(s), has not "
                      f"improved by at least {stopper.min_delta:g} for {stopper.bad} in a row. Best smoothed "
                      f"{sign * stopper.top:.4g} at epoch {stopper.top_step}. Stopping "
                      f"(early_stop_patience={stopper.patience}).")
                stopped_early = True
                break
    except KeyboardInterrupt:
        print(f"\ninterrupted during epoch {max(ep, 1)}; nothing from that epoch is kept")

    tb.close()
    epochs_run = history[-1]["epoch"] if history else 0
    summary = {
        **run_info,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "diverged": diverged,
        "metric": ("vcc" if watch_vcc else "val_loss") if val_ds is not None else None,
        "best_epoch": stopper.top_step if val_ds is not None and stopper.top_step else None,
        "best_score_smoothed": (stopper.top if watch_vcc else -stopper.top) if val_ds is not None and stopper.top_step else None,
        "hit_max_epochs": bool(val_ds is not None and stopper.best_step == epochs and not stopped_early),
    }
    (run_dir / "run.json").write_text(json.dumps(summary, indent=2))

    if not history:
        print("no epoch completed, so no checkpoint was written")
        return summary
    if vcc_track:
        print(f"\ncell-eval average by epoch (scaled; mean_shift {ref['mean_shift']['avg_score']:+.3f}):")
        print("  " + "   ".join(f"epoch {e}: {a:+.3f}" for e, a in vcc_track))
    print(f"\nstopped after epoch {epochs_run} of at most {epochs}"
          f"{' (plateau)' if stopped_early else ''}. Written to {run_dir}/:")
    print("  last.pt      the last completed epoch" + ("   <- the model to use" if val_ds is None else ""))
    if val_ds is not None:
        print(f"  best.pt      lowest held-out loss {best_loss:.4f}")
        if best_vcc > float("-inf"):
            print(f"  best_vcc.pt  highest cell-eval avg score {best_vcc:+.4f}  (pass --checkpoint to use it)")
    print("  pca.npz, model_genes.csv, fingerprints.npz (what prediction needs), run.json, history.json, train.log")
    return summary


def _summarise_crossval(folds: list[dict], out: Path, max_epochs: int) -> dict:
    """Aggregate the per-fold runs and recommend an epoch count for `full`."""
    have = [f for f in folds if f.get("best_epoch")]
    if not have:
        raise SystemExit("no fold produced a held-out score; nothing to summarise")
    best_epochs = [f["best_epoch"] for f in have]
    scores = [f["best_score_smoothed"] for f in have]
    recommended = int(np.ceil(np.median(best_epochs)))
    metric = have[0]["metric"]
    capped = [f["held_out"][0] for f in have if f["hit_max_epochs"]]
    summary = {
        "metric": metric,
        "max_epochs": max_epochs,
        "recommended_epochs": recommended,
        "best_epochs": {f["held_out"][0]: f["best_epoch"] for f in have},
        "best_score_mean": float(np.mean(scores)),
        "best_score_std": float(np.std(scores)),
        "per_line": {f["held_out"][0]: {
            "best_epoch": f["best_epoch"], "best_score": f["best_score_smoothed"],
            "epochs_run": f["epochs_run"], "stopped_early": f["stopped_early"],
            "hit_max_epochs": f["hit_max_epochs"]} for f in have},
        "folds_still_improving_at_max": capped,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    what = "cell-eval average (scaled; 1 = perfect)" if metric == "vcc" else "held-out loss (lower is better)"
    print(f"\n=== cross-validation summary ({len(have)} folds) — watching the {what} ===")
    print(f"  {'held-out line':44s}{'best epoch':>11s}{'best score':>12s}{'epochs run':>12s}")
    for f in have:
        flag = "  <- still improving at the max" if f["hit_max_epochs"] else ""
        print(f"  {f['held_out'][0]:44s}{f['best_epoch']:11d}{f['best_score_smoothed']:12.4f}{f['epochs_run']:12d}{flag}")
    print(f"  {'mean +- std':44s}{'':11s}{np.mean(scores):8.4f} +- {np.std(scores):.4f}")
    print(f"\n  recommended epochs for the final model (median best epoch): {recommended}")
    if capped:
        print(f"  WARNING: {len(capped)} fold(s) were still improving at the maximum of {max_epochs} epochs "
              f"({', '.join(capped)}). Raise train.epochs and re-run before trusting the recommendation.")
    print(f"  written: {out / 'summary.json'}")
    print("  next:  set train.full_epochs: " + str(recommended) + " in the config, then make train-full")
    return summary


def _need(tcfg: dict, key: str, cfg_path: str):
    if key not in tcfg:
        raise SystemExit(f"{cfg_path}: train.{key} is missing. It is a run choice and is never guessed "
                         f"-- set it in the config (see configs/prototype.yaml).")
    return tcfg[key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("holdout", "crossval", "full"), required=True)
    ap.add_argument("--vcc25", action="store_true",
                    help="holdout mode: validate on ALL the VCC25__* lines together (instead of --holdout)")
    ap.add_argument("--holdout", default=None, metavar="CONTEXT",
                    help="holdout mode: the ONE context to validate on (required in holdout mode), "
                         "e.g. VCC25__adata_Validation")
    ap.add_argument("--eval-cells", type=int, default=None, metavar="N",
                    help="cell-eval draws at most N random cells per (line, perturbation) "
                         "(overrides train.cell_eval_max_cells; 0 = no cap)")
    ap.add_argument("--exclude", nargs="+", default=None, metavar="CONTEXT",
                    help="contexts to leave out of the run entirely (neither trained on nor "
                         "validated on). Default: none. `--exclude none` = exclude nothing")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["train"]
    if args.eval_cells is not None:
        tcfg["cell_eval_max_cells"] = args.eval_cells
    if args.holdout and args.mode != "holdout":
        raise SystemExit("--holdout only applies to --mode holdout")

    # every run choice comes from the YAML and only from there; --mode just says which run to do
    stop = None
    if args.mode == "full":
        epochs = int(_need(tcfg, "full_epochs", args.config))
    else:
        epochs = int(_need(tcfg, "max_epochs", args.config))
        es = _need(tcfg, "early_stop", args.config)
        for k in ("metric", "patience", "window", "min_delta"):
            if k not in es:
                raise SystemExit(f"{args.config}: train.early_stop.{k} is missing (never guessed).")
        if es["metric"] not in ("vcc", "val_loss"):
            raise SystemExit("train.early_stop.metric must be 'vcc' or 'val_loss'")
        if es["patience"] < 0 or es["window"] < 1:
            raise SystemExit("train.early_stop.patience must be >= 0 and window >= 1")
        stop = {"metric": es["metric"], "patience": int(es["patience"]),
                "window": int(es["window"]), "min_delta": float(es["min_delta"])}
    if epochs < 1:
        raise SystemExit("the epoch count must be at least 1")

    # one version string: the run folder is runs/<config file name> (configs/v1_0_1.yaml ->
    # runs/v1_0_1), the same name the Makefile derives everything else from. A config may
    # still pin train.out_dir explicitly.
    base = Path(tcfg.get("out_dir") or Path("runs") / Path(args.config).stem)

    # contexts to leave out entirely: a run parameter only (--exclude), nothing is excluded by default
    exclude = [] if args.exclude in (None, ["none"]) else list(args.exclude)
    _say("SETUP 1/5  opening the processed contexts (reading metadata; the biggest small ones go into RAM)...")
    meta, contexts = load_contexts(cfg["data"]["processed_dir"], exclude=exclude)
    names = [c.name for c in contexts]
    _say(f"          {len(contexts)} contexts opened: "
         + ", ".join(f"{c.name} ({'RAM' if c.X is not None else 'disk'})" for c in contexts))
    if exclude:
        print(f"excluded contexts: {', '.join(exclude)}")
    held = args.holdout or (tcfg.get("holdout") if args.mode == "holdout" and not args.vcc25 else None)
    if held in exclude:
        raise SystemExit(f"the held-out context {held!r} is in the exclude list; pick another "
                         f"(--holdout) or drop it from --exclude")

    if args.mode == "full":
        train_run(cfg, meta, contexts, names, [], base / "full", epochs, "full", None, args.device)
        return

    if args.vcc25 and args.mode != "holdout":
        raise SystemExit("--vcc25 only applies to --mode holdout")
    if args.mode == "holdout" and args.vcc25:
        val = [n for n in names if n.startswith("VCC25__")]
        if not val:
            raise SystemExit("--vcc25: no context named VCC25__* in the processed data")
        train_run(cfg, meta, contexts, [n for n in names if n not in val], val,
                  base / "holdout_vcc25", epochs, "holdout", stop, args.device)
        return

    if args.mode == "holdout":
        line = args.holdout
        if not line:
            raise SystemExit("holdout mode needs the context to validate on: pass --holdout <context> "
                             "(make holdout HOLDOUT=<context>), or use `make holdout-vcc25`. Contexts:\n  "
                             + "\n  ".join(names))
        if line not in names:
            raise SystemExit("the held-out context (--holdout) must be one of the cell lines:\n  " + "\n  ".join(names)
                             + f"\n(got {line!r})")
        train_run(cfg, meta, contexts, [n for n in names if n != line], [line],
                  base / f"holdout_{line}", epochs, "holdout", stop, args.device)
        return

    # crossval: every line (or `train.crossval_lines`) is held out once
    lines = tcfg.get("crossval_lines") or names
    bad = [n for n in lines if n not in names]
    if bad:
        raise SystemExit(f"train.crossval_lines has unknown cell line(s): {bad}")
    out = base / "crossval"
    out.mkdir(parents=True, exist_ok=True)
    folds = []
    for k, line in enumerate(lines, 1):
        print(f"\n########## fold {k}/{len(lines)}: holding out {line} ##########")
        folds.append(train_run(cfg, meta, contexts, [n for n in names if n != line], [line],
                               out / f"fold_{line}", epochs, "crossval", stop, args.device))
    _summarise_crossval(folds, out, epochs)


if __name__ == "__main__":
    main()
