"""STAGE 1 of the two-stage model: train the MEAN MODEL (models/mean_model.py).

    python -m signalflow.training.train_mean --config configs/v2_0_0.yaml --mode holdout|crossval|full

It learns each (cell line, knockdown)'s AVERAGE shift from a TABLE of averages -- one row per
(training line, knockdown), ~53k rows, the target being that knockdown's mean PC-score shift
over all its cells in that line (`make prepare` already stores it: pca_space delta tables).
No single cells, no flow: an epoch takes seconds, a run minutes. The flow (stage 2,
training/train.py with model.centered: true) is trained AFTERWARDS on top of the frozen result.

MODES
    holdout    hold one context out (default data.pca.holdout = VCC25__adata_Validation): train on the
               rest, keep the epoch with the lowest error on the held-out line's real averages, then
               score it with cell-eval2 next to identity / mean_shift / the linear model alone.
    crossval   LEAVE ONE CELL LINE OUT over the training lines of that same artifact (7 lines ->
               7 folds): the honest test for a line the model never saw. Averages only (no
               cell-eval), minutes in total. Recommends `mean.full_epochs`.
    full       every line, exactly `mean.full_epochs`: the mean model for the final prediction
               (needs `make prepare HOLDOUT=none`).

NOTHING PER CELL LINE IS LEARNED OR STORED. A line enters only through features computed from its
OWN control cells (average control cell, slope on the target, target expression) -- see the
mean_model docstring. That is what lets the final model predict a line it has never seen.

WHAT A RUN WRITES  runs/<config name>/mean/<holdout_<line> | full | crossval>/
    mean.pt              the model (EMA weights of the best / last epoch)   <- stage 2 loads this
    pca.npz, model_genes.csv, fingerprints.npz, linear_genes.npz   what prediction needs
    run.json, history.json, train.log, scores.csv (holdout)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..data import pca_space
from ..data.dataset import FlowDataset, load_contexts
from ..models import linear_genes
from ..models import mean_model as mm
from ..models.build import pick_device
from .train import _logged, _say


# ---- the table ------------------------------------------------------------------------------

def own_deltas(art: pca_space.Artifact, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(perts, n_cells, mean shift [n, K]) of a context `prepare` computed them for (training lines)."""
    z = np.load(art.dir / "delta" / f"{name}.npz")
    return z["perts"].astype(np.int64), z["n_cells"].astype(np.float32), z["fp"].astype(np.float32)


class Table:
    """Rows of (context, knockdown): inputs, target Y (real average shift), weight. On `device`."""

    def __init__(self, art, stores, own: dict, feat_names, pcfg, pert_names, gene_names, mc, seed, device,
                 cache: dict) -> None:
        cols = {k: [] for k in ("D", "d_ok", "S", "s_ok", "e", "Y", "n", "ctx", "pert")}
        Ms, Cs, self.names = [], [], []
        for ci, c in enumerate(stores):
            perts, n, Y = own[c.name]
            inp = mm.context_inputs(art, c, perts, feat_names, pcfg, pert_names, gene_names,
                                    int(mc["ctrl_cells"]), seed, cache)
            keep = (inp["d_ok"] > 0) | (inp["s_ok"] > 0)       # neither: nothing says which knockdown it is
            for k in ("D", "d_ok", "S", "s_ok", "e"):
                cols[k].append(inp[k][keep])
            cols["Y"].append(Y[keep]); cols["n"].append(n[keep]); cols["pert"].append(perts[keep])
            cols["ctx"].append(np.full(int(keep.sum()), ci, dtype=np.int64))
            Ms.append(inp["M"]); Cs.append(inp["C"]); self.names.append(c.name)
        t = lambda a: torch.as_tensor(np.concatenate(a), device=device)
        for k, v in cols.items():
            setattr(self, k, t(v))
        self.M = torch.as_tensor(np.stack(Ms), device=device)
        self.C = torch.as_tensor(np.stack(Cs), device=device)
        self.rows_per_ctx = np.bincount(self.ctx.cpu().numpy(), minlength=len(stores))

    def __len__(self) -> int:
        return int(self.Y.shape[0])

    def weights(self, mc) -> torch.Tensor:
        """n / (n + n0) per row, times a per-context balance n_rows ** -alpha; mean 1."""
        w = self.n / (self.n + float(mc["n0"]))
        alpha = float(mc["context_balance"] or 0.0)
        if alpha > 0:
            cw = torch.as_tensor(np.maximum(self.rows_per_ctx, 1) ** -alpha, dtype=torch.float32, device=w.device)
            w = w * cw[self.ctx]
        return w / w.mean()

    def batch(self, idx):
        return (self.D[idx], self.d_ok[idx], self.S[idx], self.s_ok[idx], self.M[self.ctx[idx]],
                self.C[self.ctx[idx]], self.e[idx])


# ---- training ---------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tab: Table, gains0) -> dict:
    """Error of the predicted averages on `tab`'s rows (unweighted, per knockdown):
    mse = mean ||pred - real||^2, next to predicting zero and the linear start (gains0)."""
    model.eval()
    P = torch.cat([model(*tab.batch(torch.arange(i, min(i + 4096, len(tab)), device=tab.Y.device)))
                   for i in range(0, len(tab), 4096)])
    Y = tab.Y
    lin = gains0[0] * tab.D + gains0[1] * tab.S + gains0[2] * tab.M[tab.ctx]
    cs = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1)
    Pc, Yc = P - P.mean(0), Y - Y.mean(0)
    out = {"mse": float(((P - Y) ** 2).sum(1).mean()), "mse_zero": float((Y ** 2).sum(1).mean()),
           "mse_linear": float(((lin - Y) ** 2).sum(1).mean()),
           "cos": float(cs(P, Y).median()), "cos_linear": float(cs(lin, Y).median()),
           "cos_centered": float(cs(Pc, Yc).median()),
           "cos_centered_linear": float(cs(lin - lin.mean(0), Yc).median())}
    out["skill_vs_linear"] = 1.0 - out["mse"] / max(out["mse_linear"], 1e-12)
    return out


def fit(train: Table, val: Table | None, mc: dict, space, device, epochs: int, seed: int, say=print):
    """Train on `train`, keep the EMA weights of the epoch with the lowest `val` error (or the
    last epoch without `val`). Returns (model, gains0, history, best_epoch)."""
    torch.manual_seed(seed)
    w = train.weights(mc)
    gains0 = mm.fit_gains(train.D, train.S, train.M[train.ctx], train.Y, w)
    rms = lambda x, ok: float(torch.sqrt((x[ok > 0] ** 2).mean())) if bool((ok > 0).any()) else 1.0
    out_sd = torch.sqrt((w[:, None] * train.Y ** 2).sum(0) / w.sum()).cpu().numpy()
    model = mm.MeanModel(space.n_pcs, space.pc_sd, gain0=gains0, scale_d=rms(train.D, train.d_ok),
                         scale_s=rms(train.S, train.s_ok), out_sd=out_sd, n_c=mc["n_c"], enc=mc["enc"],
                         hidden=mc["hidden"], dropout=mc["dropout"], c_noise=mc["c_noise"],
                         c_drop=mc["c_drop"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay = float(mc["ema_decay"])
    opt = torch.optim.AdamW([{"params": [model.gain], "weight_decay": 0.0},
                             {"params": [p for n, p in model.named_parameters() if n != "gain"]}],
                            lr=float(mc["lr"]), weight_decay=float(mc["weight_decay"]))
    bs = int(mc["batch_size"])
    n_steps = max(epochs * (len(train) // bs), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(mc["lr"]), total_steps=n_steps, pct_start=0.05)
    say(f"  {len(train):,} training rows; linear start g_d={gains0[0]:+.3f} g_s={gains0[1]:+.3f} "
        f"g_m={gains0[2]:+.3f}; {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    if val is not None:
        v0 = evaluate(ema, val, gains0)
        say(f"  held-out rows {len(val):,}: ||real||^2 {v0['mse_zero']:.3f}   linear start {v0['mse_linear']:.3f} "
            f"(cos {v0['cos_linear']:+.3f}, pert-specific cos {v0['cos_centered_linear']:+.3f})")
    gen = torch.Generator(device=device).manual_seed(seed)
    history, best, best_ep, best_state, bad, step = [], float("inf"), 0, None, 0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train), device=device, generator=gen)
        tot, tw = 0.0, 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i : i + bs]
            pred = model(*train.batch(idx))
            wb = w[idx]
            loss = (wb * ((pred - train.Y[idx]) ** 2).mean(1)).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < n_steps - 1:
                sched.step()
            step += 1
            d = min(decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - d)
            tot += float(loss) * float(wb.sum()); tw += float(wb.sum())
        entry = {"epoch": ep, "train_loss": tot / max(tw, 1e-12), "gain": ema.gain.detach().cpu().tolist()}
        line = f"  epoch {ep:3d}  train {entry['train_loss']:.5f}"
        if val is not None:
            v = evaluate(ema, val, gains0)
            entry.update({f"val_{k}": x for k, x in v.items()})
            line += (f"  held-out ||err||^2 {v['mse']:.3f} (linear {v['mse_linear']:.3f}, zero {v['mse_zero']:.3f}; "
                     f"skill vs linear {v['skill_vs_linear']:+.1%})  cos {v['cos']:+.3f}  pert-specific cos "
                     f"{v['cos_centered']:+.3f}")
            if v["mse"] < best:
                best, best_ep, bad = v["mse"], ep, 0
                best_state = copy.deepcopy(ema.state_dict())
                line += "  *"
            else:
                bad += 1
        say(line + f"  [{time.time() - t0:.1f}s]")
        history.append(entry)
        if val is not None and bad >= int(mc["patience"]):
            say(f"  no better held-out error for {bad} epochs: stopping (best epoch {best_ep})")
            break
    if best_state is not None:
        ema.load_state_dict(best_state)
    return ema, gains0, history, (best_ep if val is not None else history[-1]["epoch"])


# ---- modes ------------------------------------------------------------------------------------

def _names(root: Path):
    gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    pert_names = pd.read_csv(root / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    return gene_names, pert_names


def _write_run(run_dir, art, train_ctx, pert_names, cfg, model, gains0, history, best_ep, mode, held, extra=None):
    names = [c.name for c in train_ctx]
    sc = pca_space.scales(art, names)
    pca_space.save_for_run(run_dir, art, names, sc, pert_names)
    trained = sorted({int(p) for c in train_ctx for p in np.unique(c.pert) if p != 0})
    mm.save(run_dir / "mean.pt", model, cfg, train_contexts=names, held_out=held, pca_space=str(art.dir),
            best_epoch=best_ep, gains0=[float(g) for g in gains0])
    info = {"kind": mm.KIND, "mode": mode, "train_contexts": names, "held_out": held, "pca_space": str(art.dir),
            "best_epoch": best_ep, "epochs_run": history[-1]["epoch"] if history else 0,
            "gains0": [float(g) for g in gains0], "gains": model.gain.detach().cpu().tolist(),
            "mean": mm.mean_cfg(cfg), "trained_perts": [str(pert_names[p]) for p in trained], **(extra or {})}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))


def _other_genes(cfg, run_dir, train_ctx, all_ctx, val_ctx, G, device, pert_names):
    """The linear model on the non-model genes: saved for prediction, and the held-out shifts."""
    if linear_genes.mode(cfg) != "linear":
        return None
    _say("          fitting the linear population-mean model for the non-model genes...")
    coef, D_l, S_l = linear_genes.fit_for_run(cfg, train_ctx, all_ctx, G, device)
    linear_genes.save_for_run(run_dir, train_ctx, D_l, coef, G, pert_names)
    out = {c.name: linear_genes.heldout_shifts(c, train_ctx, coef, D_l, S_l, G)[0] for c in val_ctx}
    del D_l, S_l
    return out


def run_holdout(cfg, meta, contexts, held: str, run_dir: Path, device, eval_cells=None) -> None:
    mc, pcfg, seed = mm.mean_cfg(cfg), pca_space.pca_cfg(cfg), int(cfg.get("seed", 0))
    root = Path(cfg["data"]["processed_dir"])
    gene_names, pert_names = _names(root)
    train_ctx = [c for c in contexts if c.name != held]
    V = next(c for c in contexts if c.name == held)
    art = pca_space.Artifact(pca_space.find(root, [held], cfg, contexts))
    space = art.space
    names = [c.name for c in train_ctx]
    _say(f"SETUP  PCA space {art.dir.name}: {space.n_pcs} PCs; training lines: {len(train_ctx)}, held out: {held}")
    for c in contexts:
        c.model_col = pca_space.model_cols(c.gene_idx, space.gene_idx, c.name)
    own = {c.name: own_deltas(art, c.name) for c in train_ctx if (art.dir / "delta" / f"{c.name}.npz").exists()}
    train_ctx = [c for c in train_ctx if c.name in own]
    _say(f"          the held-out line's REAL averages (for measuring only, never trained on)...")
    d = pca_space.delta_table(V, space, V.model_col, int(pcfg["min_cells"]))
    carried = {int(p) for c in train_ctx for p in own[c.name][0]}
    vk = np.array([int(p) in carried for p in d["perts"]])
    own[V.name] = (d["perts"][vk].astype(np.int64), d["n_cells"][vk].astype(np.float32), d["fp"][vk])
    _say("          building the tables (inputs from each line's own controls)...")
    cache: dict = {}
    tr = Table(art, train_ctx, own, names, pcfg, pert_names, gene_names, mc, seed, device, cache)
    va = Table(art, [V], own, names, pcfg, pert_names, gene_names, mc, seed, device, cache)
    _say("TRAIN  mean model")
    model, gains0, history, best_ep = fit(tr, va, mc, space, device, int(mc["epochs"]), seed, say=print)
    _write_run(run_dir, art, train_ctx, pert_names, cfg, model, gains0, history, best_ep, "holdout", [held])
    print(f"\nbest epoch {best_ep}: {json.dumps({k: round(v, 4) for k, v in history[best_ep - 1].items() if k.startswith('val_')})}")

    if not int(cfg["train"].get("cell_eval_every", 1)):
        return
    # ---- cell-eval2 on the held-out line: mean model alone vs the linear start vs baselines
    from ..evaluation.cell_eval import ValScorer, format_reference, summarize

    tcfg = cfg["train"]
    sc = pca_space.scales(art, names)
    for c in [V] + train_ctx:
        c.fp_lookup, c.fp_rows = pca_space.context_fingerprints(art, c.name, np.unique(c.pert), names,
                                                                meta["n_perts"], sc, pcfg)
    trained = sorted({int(p) for c in train_ctx for p in np.unique(c.pert) if p != 0})
    val_ds = FlowDataset([V], seed=seed + 1, keep_perts=np.array(trained, dtype=np.int64))
    other = _other_genes(cfg, run_dir, train_ctx, contexts, [V], meta["n_genes"], device, pert_names)
    _say("EVAL   cell-eval2 on the held-out line (identity, mean_shift, the linear start, the mean model)...")
    scorer = ValScorer(cfg, train_ctx, val_ds, device, space, n_steps=1,
                       min_cells=int(tcfg.get("cell_eval_min_cells", 5)),
                       max_cells=int(eval_cells if eval_cells is not None else tcfg.get("cell_eval_max_cells", 0)),
                       log=lambda s: None)
    ps = scorer.perts()[V.name]
    inp = mm.context_inputs(art, V, ps, names, pcfg, pert_names, gene_names, int(mc["ctrl_cells"]), seed, cache)
    pred = mm.predict(model, inp, device)
    lin = gains0[0] * inp["D"] + gains0[1] * inp["S"] + gains0[2] * inp["M"][None, :]
    meth = "mean_linear" if other is not None else "mean"
    df = scorer.score(None, methods=("identity", "mean_shift"), tag="reference | ")
    df_lin = scorer.score(None, methods=(meth,), other_shift=other,
                          mean_dz={V.name: {p: lin[i].astype(np.float32) for i, p in enumerate(ps)}})
    df_lin["method"] = "linear"
    df_mm = scorer.score(None, methods=(meth,), other_shift=other,
                         mean_dz={V.name: {p: pred[i] for i, p in enumerate(ps)}})
    df_mm["method"] = "mean_model"
    df = pd.concat([df, df_lin, df_mm], ignore_index=True)
    df.to_csv(run_dir / "scores.csv", index=False)
    ref = {m: summarize(df, m) for m in ("identity", "mean_shift", "linear", "mean_model")}
    print("\n" + format_reference(ref).split("\n  AVERAGE =")[0])
    info = json.loads((run_dir / "run.json").read_text())
    info["cell_eval"] = {m: ref[m]["avg_score"] for m in ref}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    print(f"\nwritten to {run_dir}/  (mean.pt = the model stage 2 builds on)")


def run_crossval(cfg, contexts, held: str, out: Path, device) -> None:
    """Leave one CELL LINE out over the training lines of the `held` artifact."""
    mc, pcfg, seed = mm.mean_cfg(cfg), pca_space.pca_cfg(cfg), int(cfg.get("seed", 0))
    root = Path(cfg["data"]["processed_dir"])
    gene_names, pert_names = _names(root)
    art = pca_space.Artifact(pca_space.find(root, [held], cfg, contexts))
    space = art.space
    pool = [c for c in contexts if c.name != held and (art.dir / "delta" / f"{c.name}.npz").exists()]
    for c in pool:
        c.model_col = pca_space.model_cols(c.gene_idx, space.gene_idx, c.name)
    own = {c.name: own_deltas(art, c.name) for c in pool}
    lines: dict[str, list] = {}
    for c in pool:
        lines.setdefault(pca_space.cell_line(c.name, pcfg), []).append(c)
    _say(f"CROSSVAL  leave one cell line out: {len(lines)} lines ({', '.join(lines)}); {held} is never used")
    cache: dict = {}
    folds = {}
    for k, (line, held_ctx) in enumerate(lines.items(), 1):
        tr_ctx = [c for c in pool if c not in held_ctx]
        names = [c.name for c in tr_ctx]
        print(f"\n########## fold {k}/{len(lines)}: cell line {line} ({', '.join(c.name for c in held_ctx)}) unseen ##########")
        tr = Table(art, tr_ctx, own, names, pcfg, pert_names, gene_names, mc, seed, device, cache)
        va = Table(art, held_ctx, own, names, pcfg, pert_names, gene_names, mc, seed, device, cache)
        model, gains0, history, best_ep = fit(tr, va, mc, space, device, int(mc["epochs"]), seed, say=print)
        b = history[best_ep - 1]
        folds[line] = {"best_epoch": best_ep, "rows": len(va), **{k2[4:]: v for k2, v in b.items() if k2.startswith("val_")}}
        del tr, va, model
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    rec = int(np.ceil(np.median([f["best_epoch"] for f in folds.values()])))
    (out / "summary.json").write_text(json.dumps({"folds": folds, "recommended_full_epochs": rec}, indent=2))
    print("\n=== leave-one-cell-line-out: the mean model on lines it never saw ===")
    print(f"  {'unseen line':12s}{'rows':>8s}{'best ep':>9s}{'err':>9s}{'linear':>9s}{'zero':>9s}"
          f"{'skill vs lin':>14s}{'cos':>8s}{'pert cos':>10s}")
    for line, f in folds.items():
        print(f"  {line[:12]:12s}{f['rows']:8d}{f['best_epoch']:9d}{f['mse']:9.3f}{f['mse_linear']:9.3f}"
              f"{f['mse_zero']:9.3f}{f['skill_vs_linear']:+14.1%}{f['cos']:+8.3f}{f['cos_centered']:+10.3f}")
    print(f"  mean skill vs the linear model: {np.mean([f['skill_vs_linear'] for f in folds.values()]):+.1%}"
          f"   (positive = the network beats the linear formula on an UNSEEN line)")
    print(f"\n  recommended mean.full_epochs (median best epoch): {rec}   written: {out / 'summary.json'}")


def run_full(cfg, meta, contexts, run_dir: Path, device) -> None:
    mc, pcfg, seed = mm.mean_cfg(cfg), pca_space.pca_cfg(cfg), int(cfg.get("seed", 0))
    root = Path(cfg["data"]["processed_dir"])
    gene_names, pert_names = _names(root)
    art = pca_space.Artifact(pca_space.find(root, [], cfg, contexts, full=True))
    space = art.space
    train_ctx = [c for c in contexts if (art.dir / "delta" / f"{c.name}.npz").exists()]
    names = [c.name for c in train_ctx]
    for c in train_ctx:
        c.model_col = pca_space.model_cols(c.gene_idx, space.gene_idx, c.name)
    own = {c.name: own_deltas(art, c.name) for c in train_ctx}
    tr = Table(art, train_ctx, own, names, pcfg, pert_names, gene_names, mc, seed, device, {})
    epochs = int(mc["full_epochs"])
    _say(f"TRAIN  mean model on every line, exactly {epochs} epochs (mean.full_epochs)")
    model, gains0, history, best_ep = fit(tr, None, mc, space, device, epochs, seed, say=print)
    _write_run(run_dir, art, train_ctx, pert_names, cfg, model, gains0, history, best_ep, "full", [])
    _other_genes(cfg, run_dir, train_ctx, contexts, [], meta["n_genes"], device, pert_names)
    print(f"\nwritten to {run_dir}/  (mean.pt = the final mean model; `make flow-full` builds on it)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("holdout", "crossval", "full"), required=True)
    ap.add_argument("--holdout", default=None, metavar="CONTEXT",
                    help="holdout: the context to validate on; crossval: the artifact's held-out context "
                         "(never used). Default: data.pca.holdout")
    ap.add_argument("--exclude", nargs="+", default=None, metavar="CONTEXT",
                    help="contexts left out of the run entirely")
    ap.add_argument("--eval-cells", type=int, default=None, metavar="N")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = pick_device(args.device or cfg["train"].get("device", "auto"))
    held = args.holdout or (pca_space.pca_cfg(cfg)["holdout"] or [None])[0]
    exclude = [] if args.exclude in (None, ["none"]) else list(args.exclude)
    base = Path(cfg["train"].get("out_dir") or Path("runs") / Path(args.config).stem) / "mean"
    _say("SETUP  opening the processed contexts...")
    meta, contexts = load_contexts(cfg["data"]["processed_dir"], exclude=exclude)
    if args.mode == "full":
        run_dir = base / "full"
    elif args.mode == "holdout":
        if held not in [c.name for c in contexts]:
            raise SystemExit(f"--holdout {held!r} is not a (non-excluded) context")
        run_dir = base / f"holdout_{held}"
    else:
        run_dir = base / "crossval"
    run_dir.mkdir(parents=True, exist_ok=True)
    with _logged(run_dir, f"mean model {args.mode}" + (f" holding out {held}" if args.mode != "full" else "")):
        if exclude:
            print(f"excluded contexts: {', '.join(exclude)}")
        if args.mode == "full":
            run_full(cfg, meta, contexts, run_dir, device)
        elif args.mode == "holdout":
            run_holdout(cfg, meta, contexts, held, run_dir, device, args.eval_cells)
        else:
            run_crossval(cfg, contexts, held, run_dir, device)


if __name__ == "__main__":
    main()
