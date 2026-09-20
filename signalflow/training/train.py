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

There is no train/val/test split of CELLS any more, and `prepare` does no splitting. A
cell line is either training data or held out, decided here per run -- because the
shared PCA basis has to be fit without the held-out line, so it is fit here too, once
per run, and saved next to the checkpoint (`prediction/predict.py` reads it from there).

WHERE THE RUN CHOICES ARE SET: IN THE YAML, AND ONLY THERE
    `train.holdout`, `max_epochs`, `full_epochs` and the `early_stop` block (metric, patience,
    window, min_delta) say what a run does. The only flag is `--mode`, which the Makefile
    targets pass. There are no code or Makefile defaults: a missing key is an error, so
    there is no hidden second value.

EPOCHS
    `train.max_epochs` is the MAXIMUM in holdout/crossval: early stopping only ever stops sooner.
    `train.full_epochs` is the exact count in `full` mode. Treat it as a hyperparameter: crossval finds it,
    full uses it.

WHAT EARLY STOPPING WATCHES  (`train.early_stop.metric`)
    vcc        the cell-eval average on the held-out line: what the challenge scores, but
               noisy and slower, so it is measured every `train.cell_eval_every` epochs and
               smoothed (see `stopping.py`).
    val_loss   the flow-matching loss on the held-out line: cheap, every epoch, but only a
               proxy (mostly irreducible noise), and it need not agree with the metrics.
    The TRAINING loss is deliberately not an option: it falls every epoch whether or not
    the model generalises, so it cannot detect overfitting. It is only used as a guard
    (a non-finite training loss aborts the run).

WHAT A RUN WRITES  (into `<out_dir>/holdout_<line>/`, `.../crossval/fold_<line>/` or `.../full/`)
    last.pt       the most recent completed epoch (the model to use for `full`)
    best.pt       lowest held-out loss so far           (holdout / crossval)
    best_vcc.pt   highest held-out cell-eval score      (holdout / crossval, cell-eval on)
    pca_shared.npz  the PCA basis this model was trained with -- it belongs to the model
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
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data import shared_pca, state
from ..data.dataset import FlowDataset, load_contexts, make_loader
from ..data.vocab import CONTROL_LABEL
from ..models.build import build_model, pick_device
from ..models.flow import cfm_loss
from .stopping import PlateauStopper
from .tb import RunLog


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def run_val(model, loader, sigma, device) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "identity_loss": 0.0}
    n = 0
    for batch in loader:
        _, s = cfm_loss(model, to_device(batch, device), sigma)
        b = batch["x0"].shape[0]
        for k in tot:
            tot[k] += s[k] * b
        n += b
    model.train()
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out["frac_var_explained"] = 1.0 - out["loss"] / max(out["identity_loss"], 1e-12)
    return out


def _save(path: Path, model, cfg, epoch: int, **extra) -> None:
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, **extra}, path)


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
    device = pick_device(device_arg or tcfg.get("device", "auto"))
    bs = int(tcfg.get("batch_size", 256))
    sigma = float(tcfg.get("sigma", 0.0))
    n_genes, n_perts = meta["n_genes"], meta["n_perts"]
    n_pcs = int(cfg["data"].get("n_pcs", 32))

    # ---- the PCA basis: fit on the TRAINING lines only, then everyone is projected onto it
    t0 = time.time()
    loadings, mu = state.fit_basis(train_ctx, n_genes, n_pcs, seed)
    shared_pca.save(run_dir / "pca_shared.npz", loadings, mu)
    for c in train_ctx + val_ctx:
        state.attach_state(c, loadings, mu)
    print(f"shared PCA fit on the control cells of {len(train_ctx)} training context(s)"
          f"{'' if not val_ctx else ', held-out: ' + ', '.join(c.name for c in val_ctx)}  "
          f"[{time.time() - t0:.0f}s]")

    train_ds = FlowDataset(train_ctx, n_genes, n_perts, seed=seed)
    trained_perts = sorted({int(p) for c in train_ctx for p in np.unique(c.pert) if p != 0})

    val_ds = None
    if val_ctx:
        # the model cannot know a perturbation it never saw, so the held-out line is
        # judged on the perturbations the training lines carry (controls always count)
        val_ds = FlowDataset(val_ctx, n_genes, n_perts, seed=seed + 1, keep_perts=np.array(trained_perts, dtype=np.int64))
        n_pert_cells = sum(int((~c.is_control[val_ds.rows[i]]).sum()) for i, c in enumerate(val_ctx))
        if n_pert_cells == 0:
            print("WARNING: no perturbation of the held-out line appears in the training lines, so "
                  "the model has nothing to go on for any of them. The hold-out score below measures "
                  "noise, not generalisation. Falling back to ALL held-out cells so the run can proceed.")
            val_ds = FlowDataset(val_ctx, n_genes, n_perts, seed=seed + 1)

    train_dl = make_loader(train_ds, bs, seed)
    val_dl = make_loader(val_ds, bs, seed, shuffle=False) if val_ds is not None else None

    model = build_model(n_genes, n_perts, train_ds.n_state, mcfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  genes={n_genes}  perts={n_perts}  contexts: {len(train_ctx)} train"
        f"{'' if val_ds is None else f' / {len(val_ctx)} held out'}  state={train_ds.n_state}\n"
        f"train cells={len(train_ds)}" + ("" if val_ds is None else f"  held-out cells={len(val_ds)}")
        + f"  params={n_par/1e6:.2f}M  head={mcfg.get('head','plain')}  max epochs={epochs}\n"
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * len(train_dl), 1))

    pert_names = [str(x) for x in np.loadtxt(
        Path(cfg["data"]["processed_dir"]) / "pert_vocab.csv", dtype=str, delimiter="\t", skiprows=1)]
    run_info = {
        "mode": mode, "n_state": int(train_ds.n_state), "n_pcs": n_pcs,
        "train_contexts": [c.name for c in train_ctx], "held_out": [c.name for c in val_ctx],
        "trained_perts": [pert_names[p] for p in trained_perts],
        "max_epochs": epochs,
        "stop": stop,                      # the exact stopping settings this run used
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))

    # ---- what to watch, and whether we can ---------------------------------------------
    every = int(tcfg.get("cell_eval_every", 0))
    metric = stop["metric"] if stop else None
    scorer, ref = None, None
    if val_ds is not None and every > 0:
        from ..evaluation.cell_eval import ValScorer, format_reference, summarize

        print("cell-eval on the held-out line:")
        scorer = ValScorer(
            cfg, train_ctx, val_ds, device,
            n_steps=int(tcfg.get("cell_eval_steps", 20)),
            min_cells=int(tcfg.get("cell_eval_min_cells", 5)),
            log=to_log,
        )
        if scorer.ctx:
            print(f"  {scorer.n_contexts} context(s), {scorer.n_perts} perturbations, "
                  f"{scorer.n_cells:,} reference cells   "
                  f"(detail and cell-eval2's notices go to {run_dir / 'train.log'} only)")
            ref_df = scorer.score(model, methods=("identity", "mean_shift"), tag="reference | ")
            ref = {m: summarize(ref_df, m) for m in ("identity", "mean_shift")}
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
        from ..evaluation.cell_eval import format_report, summarize  # noqa: F811  (no-op if cell-eval is off)
    except Exception:  # pragma: no cover
        pass
    try:
        for ep in range(1, epochs + 1):
            t0 = time.time()
            run, n = 0.0, 0
            for batch in train_dl:
                batch = to_device(batch, device)
                loss, _ = cfm_loss(model, batch, sigma)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                b = batch["x0"].shape[0]
                run += loss.item() * b
                n += b

            tr = run / max(n, 1)
            entry = {"epoch": ep, "train_loss": tr}
            if not np.isfinite(tr):
                print(f"\nepoch {ep}: the training loss is {tr}. Stopping: the run has diverged.")
                diverged = True
                break

            if val_dl is not None:
                va = run_val(model, val_dl, sigma, device)
                entry.update({f"val_{k}": v for k, v in va.items()})
                print(
                    f"epoch {ep:3d}  train {tr:.4f}  held-out {va['loss']:.4f}  "
                    f"(identity {va['identity_loss']:.4f}, "
                    f"explained {va['frac_var_explained']*100:5.1f}%)  "
                    f"{time.time()-t0:.1f}s"
                )
                if va["loss"] < best_loss:
                    best_loss = va["loss"]
                    _save(run_dir / "best.pt", model, cfg, ep, val=va)
                # the plateau rule watches the score when cell-eval drives it (measured only
                # on scoring epochs), otherwise the held-out loss (every epoch)
                if not watch_vcc:
                    stopper.update(-va["loss"], step=ep)
            else:
                print(f"epoch {ep:3d}  train {tr:.4f}  {time.time()-t0:.1f}s")

            if scorer is not None and (ep % every == 0 or ep == epochs):
                t1 = time.time()
                s = summarize(scorer.score(model, methods=("flow",), tag=f"epoch {ep} | "), "flow")
                entry.update({f"vcc_{k}": v for k, v in s.items()})
                new_best = bool(np.isfinite(s["avg_score"]) and s["avg_score"] > best_vcc)
                if new_best:
                    best_vcc, best_vcc_ep = s["avg_score"], ep
                    _save(run_dir / "best_vcc.pt", model, cfg, ep, vcc=s)
                if watch_vcc:
                    stopper.update(s["avg_score"], step=ep)
                    entry["vcc_avg_smooth"] = stopper.smoothed
                note = ("new best -> best_vcc.pt" if new_best
                        else f"best {best_vcc:+.3f} at epoch {best_vcc_ep}")
                if stopper.patience and watch_vcc:
                    note += (f"   plateau watch {stopper.bad}/{stopper.patience}"
                             f" (smoothed {stopper.smoothed:+.3f}, best {stopper.best:+.3f})")
                vcc_track.append((ep, s["avg_score"]))
                print(format_report(
                    s, f"cell-eval (held out) after epoch {ep}   [{time.time()-t1:.0f}s]",
                    ref_avg=ref["mean_shift"]["avg_score"], note=note))
            history.append(entry)
            tb.epoch(entry, lr=opt.param_groups[0]["lr"])
            if "vcc_avg_score" in entry:
                tb.baselines(ref, ep)

            _save(run_dir / "last.pt", model, cfg, ep)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

            if stopper.should_stop:
                what, sign = ("cell-eval score", 1) if watch_vcc else ("held-out loss", -1)
                print(f"\nplateau: the {what}, smoothed over the last {stopper.window} measurement(s), has not "
                      f"improved by at least {stopper.min_delta:g} for {stopper.bad} in a row. Best smoothed "
                      f"{sign * stopper.best:.4g} at epoch {stopper.best_step}. Stopping "
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
        "best_epoch": stopper.best_step if val_ds is not None and stopper.best_step else None,
        "best_score_smoothed": (stopper.best if watch_vcc else -stopper.best) if val_ds is not None and stopper.best_step else None,
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
    print("  pca_shared.npz, run.json, history.json, train.log")
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
                    help="holdout mode: validate on ALL the VCC25__* lines together (instead of train.holdout)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["train"]

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

    base = Path(tcfg.get("out_dir", "runs/prototype"))

    meta, contexts = load_contexts(cfg["data"]["processed_dir"])
    names = [c.name for c in contexts]

    # model.pert_corr needs the per-cell-line correlation rows `prepare` writes
    if bool(cfg["model"].get("pert_corr", True)) and not any(c.corr_at for c in contexts):
        raise SystemExit(
            f"{cfg['data']['processed_dir']}: no gene-gene correlation rows, but "
            f"model.pert_corr is on. They are written by `prepare` -- re-run "
            f"`make prepare` (or set model.pert_corr: false for the one-hot-only model)."
        )

    if args.mode == "full":
        train_run(cfg, meta, contexts, names, [], base / "full", epochs, "full", None, args.device)
        return

    if args.vcc25 and args.mode != "holdout":
        raise SystemExit("--vcc25 only applies to --mode holdout")
    if args.mode == "holdout" and args.vcc25:
        val = [n for n in names if n.startswith("VCC25__")]
        if not val:
            raise SystemExit("--vcc25: no context named VCC25__* in the processed data")
        if "holdout" in tcfg:
            print(f"NOTE: --vcc25 is set, so train.holdout ({tcfg['holdout']}) is ignored.")
        train_run(cfg, meta, contexts, [n for n in names if n not in val], val,
                  base / "holdout_vcc25", epochs, "holdout", stop, args.device)
        return

    if args.mode == "holdout":
        line = _need(tcfg, "holdout", args.config)
        if line not in names:
            raise SystemExit("train.holdout must be one of the cell lines:\n  " + "\n  ".join(names)
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
