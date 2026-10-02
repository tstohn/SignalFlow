"""Train the GENE MODEL (models/gene_model.py): a per-gene density shift, on ALL genes, no PCA.

    python -m signalflow.training.train_gene --config configs/v3_0_0.yaml --mode holdout|crossval|full

Learns, per (cell line, knockdown, gene), how far the gene's distribution moves (mean shift) and how
much it widens or narrows (log sd ratio), from STATISTICS over all of the knockdown's cells --
never from single cells. Every gene a dataset measures is used (Orion / VCC25 ~18k, Replogle /
Nadig ~8-9k of the 18,533 readout genes).

FIRST RUN: the per-(context, knockdown, gene) statistics are computed once, one pass over every
processed file, and cached in <processed_dir>/gene_stats/ (Orion takes the longest). The slopes
come from the linear model's cached tables (<processed_dir>/linear_genes/, `make prepare`).

MODES
    holdout    hold one context out (default data.pca.holdout = VCC25__adata_Validation), train on the
               rest, keep the epoch with the lowest held-out error, then cell-eval2 on it next to
               identity / mean_shift / the gene model's linear start / move-only.
    crossval   LEAVE ONE CELL LINE OUT over the training lines (the honest test for an unseen line);
               errors only, no cell-eval. Recommends `gene.full_epochs`.
    full       every line, exactly `gene.full_epochs`: the gene model for the final prediction.

A line's own knockdown statistics are never an input for it: d / v / m / mv come from OTHER cell
lines only (data.pca.cell_line_groups), for training rows and the held-out line alike.

WHAT A RUN WRITES  runs/<config name>/gene/<holdout_<line> | full | crossval>/
    gene.pt             the model                                   <- predict / stage 2 use it
    gene_tables.npz     the training lines' per-knockdown statistics over all genes (for predict)
    heldout_pred_<line>.npz   (holdout) its predictions for the held-out line (stage 2 scores with them)
    run.json, history.json, train.log, scores.csv (holdout)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..data import dataset as dsmod
from ..data import pca_space
from ..models import gene_model as gm
from ..models import linear_genes
from ..models.build import pick_device
from .train import _logged, _say


# ---- data ---------------------------------------------------------------------------------------

def _names(root: Path):
    gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    pert_names = pd.read_csv(root / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    return gene_names, pert_names


class Data:
    """Every context's statistics (cached), its Line (controls-only view), and the Sources table."""

    def __init__(self, cfg, contexts, device, say=print) -> None:
        root = Path(cfg["data"]["processed_dir"])
        self.gene_names, self.pert_names = _names(root)
        self.pcfg = pca_space.pca_cfg(cfg)
        self.G = len(self.gene_names)
        say(f"          per-gene statistics of {len(contexts)} contexts (cached in {root / gm.DIR})...")
        self.stats, self.lines, self.gene_idx = {}, {}, {}
        for c in contexts:
            st = gm.context_stats(c, root / gm.DIR, say=say)
            sl = linear_genes.slopes(c, linear_genes.table_dir(cfg), self.gene_names, self.pert_names, str(device))
            self.stats[c.name] = st
            self.gene_idx[c.name] = np.asarray(c.gene_idx, dtype=np.int64)
            self.lines[c.name] = gm.Line(c.name, c.gene_idx, st["ctrl_mean"], st["ctrl_sd"], sl["perts"],
                                         sl["slope"].astype(np.float16), self.gene_names)
            say(f"            {c.name}: {len(st['perts']):,} knockdowns x {len(c.gene_idx):,} genes")
        self.sources = gm.Sources(self.G, len(self.pert_names))
        for name, st in self.stats.items():
            self.sources.add(name, st["perts"], st["delta"], st["lsr"], self.gene_idx[name])

    def line_of(self, name: str) -> str:
        return pca_space.cell_line(name, self.pcfg)

    def others(self, name: str, pool: list[str]) -> list[str]:
        """The source contexts for `name`: the ones in `pool` of a DIFFERENT cell line."""
        return [n for n in pool if self.line_of(n) != self.line_of(name)]


class Rows:
    """One context's rows (knockdowns with statistics), the sources its features come from."""

    def __init__(self, data: Data, name: str, src_names: list[str], perts=None) -> None:
        st = data.stats[name]
        self.name, self.line, self.src = name, data.lines[name], src_names
        keep = np.ones(len(st["perts"]), bool) if perts is None else np.isin(st["perts"], perts)
        self.idx = np.flatnonzero(keep)
        self.perts = st["perts"][self.idx].astype(np.int64)
        self.n = st["n"][self.idx]
        self.st = st

    def __len__(self) -> int:
        return len(self.perts)

    def targets(self, ridx, cols) -> np.ndarray:
        r = self.idx[ridx]
        return np.stack([self.st["delta"][r][:, cols].astype(np.float32),
                         self.st["lsr"][r][:, cols].astype(np.float32)], -1)


# ---- training ----------------------------------------------------------------------------------

def _batch(data: Data, R: Rows, gc: dict, rng):
    B = min(int(gc["rows_per_step"]), len(R))
    ridx = rng.choice(len(R), B, replace=False)
    n_loc = len(R.line.gene_idx)
    cols = np.sort(rng.choice(n_loc, min(int(gc["genes_per_step"]), n_loc), replace=False))
    X = R.line.features(data.sources, R.src, R.perts[ridx], data.pert_names, cols)
    Y = R.targets(ridx, cols)
    w = (R.n[ridx] / (R.n[ridx] + float(gc["n0"]))).astype(np.float32)
    return X, Y, w


@torch.no_grad()
def evaluate(model, data: Data, R: Rows, device, rows_chunk: int = 16) -> tuple[dict, dict]:
    """Held-out errors over ALL of the line's genes; returns (metrics, predictions {pert: [n_loc, 2]})."""
    preds, se_m, se_v, z_m, z_v, cs, P_all, Y_all = {}, 0.0, 0.0, 0.0, 0.0, [], [], []
    N = 0
    for lo in range(0, len(R), rows_chunk):
        ridx = np.arange(lo, min(lo + rows_chunk, len(R)))
        X = R.line.features(data.sources, R.src, R.perts[ridx], data.pert_names)
        P = gm.predict(model, X, device)
        Y = R.targets(ridx, np.arange(len(R.line.gene_idx)))
        se_m += float(((P[..., 0] - Y[..., 0]) ** 2).sum()); z_m += float((Y[..., 0] ** 2).sum())
        se_v += float(((P[..., 1] - Y[..., 1]) ** 2).sum()); z_v += float((Y[..., 1] ** 2).sum())
        N += Y[..., 0].size
        for b, i in enumerate(ridx):
            preds[int(R.perts[i])] = P[b]
            P_all.append(P[b, :, 0]); Y_all.append(Y[b, :, 0])
    P_all, Y_all = np.stack(P_all), np.stack(Y_all)
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    out = {"mse": se_m / N, "mse_zero": z_m / N, "spread_mse": se_v / N, "spread_mse_zero": z_v / N,
           "cos": float(np.median(cos(P_all, Y_all))),
           "cos_centered": float(np.median(cos(P_all - P_all.mean(0), Y_all - Y_all.mean(0))))}
    out["skill_vs_zero"] = 1.0 - out["mse"] / max(out["mse_zero"], 1e-12)
    return out, preds


def fit(data: Data, train: list[Rows], val: list[Rows] | None, gc: dict, device, epochs: int, seed: int, say=print):
    """Returns (best EMA model, linear-start model, history, best epoch)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sizes = np.array([len(R) for R in train], dtype=np.float64)
    prob = sizes ** (1.0 - float(gc["context_balance"] or 0.0))
    prob /= prob.sum()
    lam = float(gc["spread_weight"]) if gc["spread"] else 0.0

    # feature scaling + least-squares start from a sample of batches
    t0 = time.time()
    Xs, Ys, Ws = [], [], []
    for _ in range(60):
        X, Y, w = _batch(data, train[rng.choice(len(train), p=prob)], gc, rng)
        k = rng.choice(X.shape[1], min(512, X.shape[1]), replace=False)
        Xs.append(X[:, k].reshape(-1, gm.N_FEAT)); Ys.append(Y[:, k].reshape(-1, 2))
        Ws.append(np.repeat(w, len(k)))
    Xs, Ys, Ws = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)
    fm, fs = Xs.mean(0), Xs.std(0)
    fs[fs < 1e-6] = 1.0
    model = gm.GeneModel(gc["hidden"], gc["layers"], gc["dropout"], fm, fs).to(device)
    H = np.concatenate([(Xs - fm) / fs, np.ones((len(Xs), 1), np.float32)], 1).astype(np.float64)
    A = (H * Ws[:, None]).T @ H + 1e-3 * np.eye(H.shape[1])
    coef = np.linalg.solve(A, (H * Ws[:, None]).T @ Ys.astype(np.float64))        # [F + 1, 2]
    with torch.no_grad():
        model.lin.weight.copy_(torch.from_numpy(coef[:-1].T.astype(np.float32)))
        model.lin.bias.copy_(torch.from_numpy(coef[-1].astype(np.float32)))
    start = copy.deepcopy(model).eval()
    say(f"  linear start fitted on {len(Xs):,} sampled entries  [{time.time() - t0:.0f}s]; "
        f"{sum(p.numel() for p in model.parameters()):,} parameters; spread {'on' if gc['spread'] else 'off'}")
    if val:
        for R in val:
            v0, _ = evaluate(start, data, R, device)
            say(f"  held-out {R.name} ({len(R)} knockdowns): linear start mse {v0['mse']:.5f} "
                f"(zero {v0['mse_zero']:.5f}, skill {v0['skill_vs_zero']:+.1%})  spread mse {v0['spread_mse']:.5f} "
                f"(zero {v0['spread_mse_zero']:.5f})  cos {v0['cos']:+.3f}  pert-specific cos {v0['cos_centered']:+.3f}")

    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay = float(gc["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(gc["lr"]), weight_decay=float(gc["weight_decay"]))
    steps = int(gc["steps_per_epoch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(gc["lr"]), total_steps=max(epochs * steps, 1),
                                                pct_start=0.05)
    history, best, best_ep, best_state, bad, k = [], float("inf"), 0, None, 0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for _ in range(steps):
            X, Y, w = _batch(data, train[rng.choice(len(train), p=prob)], gc, rng)
            X, Y, w = (torch.from_numpy(a).to(device, non_blocking=True) for a in (X, Y, w))
            P = model(X)
            err = (P[..., 0] - Y[..., 0]) ** 2 + lam * (P[..., 1] - Y[..., 1]) ** 2
            loss = (w[:, None] * err).mean() / w.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if k < epochs * steps - 1:
                sched.step()
            k += 1
            d = min(decay, (1 + k) / (10 + k))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - d)
            tot += float(loss)
        entry = {"epoch": ep, "train_loss": tot / steps}
        line = f"  epoch {ep:3d}  train {entry['train_loss']:.5f}"
        if val:
            score = 0.0
            for R in val:
                v, _ = evaluate(ema, data, R, device)
                entry.update({f"val_{R.name}_{a}": b for a, b in v.items()})
                score += v["mse"] + lam * v["spread_mse"]
                line += (f"  | {R.name[:18]}: mse {v['mse']:.5f} (skill vs zero {v['skill_vs_zero']:+.1%}) spread "
                         f"{v['spread_mse']:.5f} cos {v['cos']:+.3f} pert cos {v['cos_centered']:+.3f}")
            entry["val_score"] = score
            if score < best:
                best, best_ep, bad = score, ep, 0
                best_state = copy.deepcopy(ema.state_dict())
                line += "  *"
            else:
                bad += 1
        say(line + f"  [{time.time() - t0:.0f}s]")
        history.append(entry)
        if val and bad >= int(gc["patience"]):
            say(f"  no better held-out error for {bad} epochs: stopping (best epoch {best_ep})")
            break
    if best_state is not None:
        ema.load_state_dict(best_state)
    return ema, start, history, (best_ep if val else history[-1]["epoch"])


def predict_perts(model, data: Data, name: str, src: list[str], perts, device, chunk: int = 16) -> dict:
    """{pert: [n_local, 2] (shift, lsr)} for knockdowns `perts` of line `name` over all its genes --
    from features only (no statistics of `name`'s knockdowns are needed)."""
    out = {}
    perts = np.asarray(perts, dtype=np.int64)
    for lo in range(0, len(perts), chunk):
        pp = perts[lo:lo + chunk]
        P = gm.predict(model, data.lines[name].features(data.sources, src, pp, data.pert_names), device)
        out.update({int(p): P[i] for i, p in enumerate(pp)})
    return out


# ---- modes -------------------------------------------------------------------------------------

def _link(src: Path, dst: Path) -> None:
    """Hard link (no second copy of a big file), copy where that is impossible."""
    import shutil

    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _write(run_dir, data: Data, train_names, cfg, model, history, best_ep, mode, held, extra=None) -> None:
    gm.save(run_dir / "gene.pt", model, cfg, train_contexts=train_names, held_out=held, best_epoch=best_ep)
    _say("          saving the training lines' per-knockdown statistics over all genes (for predict)...")
    tab = data.sources.merged(train_names, data.pert_names)
    np.savez(run_dir / "gene_tables.npz", **tab)
    trained = sorted({int(p) for n in train_names for p in data.stats[n]["perts"]})
    info = {"kind": gm.KIND, "mode": mode, "train_contexts": train_names, "held_out": held, "best_epoch": best_ep,
            "epochs_run": history[-1]["epoch"] if history else 0, "gene": gm.gene_cfg(cfg),
            "trained_perts": [str(data.pert_names[p]) for p in trained], **(extra or {})}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))


def run_holdout(cfg, meta, contexts, held, run_dir: Path, device, eval_cells=None) -> None:
    gc, seed = gm.gene_cfg(cfg), int(cfg.get("seed", 0))
    data = Data(cfg, contexts, device, say=_say)
    train_names = [c.name for c in contexts if c.name != held]
    train = [Rows(data, n, data.others(n, train_names)) for n in train_names]
    carried = np.unique(np.concatenate([R.perts for R in train]))
    V = Rows(data, held, data.others(held, train_names), perts=carried)
    _say(f"TRAIN  gene model on {len(train_names)} contexts ({sum(len(R) for R in train):,} knockdown rows); "
         f"held out {held}: {len(V)} knockdowns; its sources: {', '.join(V.src)}")
    model, start, history, best_ep = fit(data, train, [V], gc, device, int(gc["epochs"]), seed, say=print)
    _write(run_dir, data, train_names, cfg, model, history, best_ep, "holdout", [held])
    # the held-out line's predictions for every knockdown of it the training lines carry: stage 2
    # (the flow) scores with these, so it never has to load the statistics itself
    in_train = {int(p) for c in contexts if c.name != held for p in np.unique(c.pert) if p != 0}
    held_perts = np.array(sorted(int(p) for p in np.unique(next(c for c in contexts if c.name == held).pert)
                                 if int(p) in in_train), dtype=np.int64)
    preds = predict_perts(model, data, held, V.src, held_perts, device)
    np.savez(run_dir / f"heldout_pred_{held}.npz", perts=held_perts,
             pred=np.stack([preds[int(p)] for p in held_perts]).astype(np.float16), mu=data.lines[held].mu,
             gene_idx=data.lines[held].gene_idx)
    print(f"\nbest epoch {best_ep}")

    if not int(cfg["train"].get("cell_eval_every", 1)):
        return
    # ---- cell-eval2 (the PCA artifact is only used for the scorer's bookkeeping, not by the model)
    from ..data.dataset import FlowDataset
    from ..evaluation.cell_eval import ValScorer, format_reference, summarize

    root = Path(cfg["data"]["processed_dir"])
    art = pca_space.Artifact(pca_space.find(root, [held], cfg, contexts))
    sc = pca_space.scales(art, train_names)
    Vc = next(c for c in contexts if c.name == held)
    Vc.model_col = pca_space.model_cols(Vc.gene_idx, art.space.gene_idx, Vc.name)
    Vc.fp_lookup, Vc.fp_rows = pca_space.context_fingerprints(art, Vc.name, np.unique(Vc.pert), train_names,
                                                              meta["n_perts"], sc, data.pcfg)
    tr_ctx = [c for c in contexts if c.name != held]
    trained = sorted({int(p) for c in tr_ctx for p in np.unique(c.pert) if p != 0})
    val_ds = FlowDataset([Vc], seed=seed + 1, keep_perts=np.array(trained, dtype=np.int64))
    tcfg = cfg["train"]
    _say("EVAL   cell-eval2 on the held-out line (identity, mean_shift, linear start, move only, move + stretch)...")
    scorer = ValScorer(cfg, tr_ctx, val_ds, device, art.space, n_steps=1,
                       min_cells=int(tcfg.get("cell_eval_min_cells", 5)),
                       max_cells=int(eval_cells if eval_cells is not None else tcfg.get("cell_eval_max_cells", 0)),
                       log=lambda s: None)
    ps = scorer.perts()[held]
    pstart = predict_perts(start, data, held, V.src, ps, device)

    def as_pred(pr):
        out = {"mu": data.lines[held].mu}
        out.update({p: (pr[p][:, 0].astype(np.float32), pr[p][:, 1].astype(np.float32)) for p in ps})
        return {held: out}

    df = scorer.score(None, methods=("identity", "mean_shift"), tag="reference | ")
    parts = [df]
    for label, method, pr in (("linear_start", "gene", pstart), ("gene_shift", "gene_mean", preds), ("gene_model", "gene", preds)):
        d = scorer.score(None, methods=(method,), gene_pred=as_pred(pr))
        d["method"] = label
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(run_dir / "scores.csv", index=False)
    ref = {m: summarize(df, m) for m in ("identity", "mean_shift", "linear_start", "gene_shift", "gene_model")}
    print("\n" + format_reference(ref).split("\n  AVERAGE =")[0])
    print("  gene_shift = the gene model, move only;  gene_model = move + stretch (the full gene model)")
    info = json.loads((run_dir / "run.json").read_text())
    info["cell_eval"] = {m: ref[m]["avg_score"] for m in ref}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    print(f"\nwritten to {run_dir}/  (gene.pt; stage 2 = `make gene-flow-holdout`)")


def run_crossval(cfg, contexts, held, out: Path, device) -> None:
    gc, seed = gm.gene_cfg(cfg), int(cfg.get("seed", 0))
    pool = [c for c in contexts if c.name != held]
    data = Data(cfg, pool, device, say=_say)
    names = [c.name for c in pool]
    lines: dict[str, list[str]] = {}
    for n in names:
        lines.setdefault(data.line_of(n), []).append(n)
    _say(f"CROSSVAL  leave one cell line out: {len(lines)} lines ({', '.join(lines)}); {held} is never used")
    folds = {}
    for k, (line, held_names) in enumerate(lines.items(), 1):
        tr_names = [n for n in names if n not in held_names]
        print(f"\n########## fold {k}/{len(lines)}: cell line {line} ({', '.join(held_names)}) unseen ##########")
        train = [Rows(data, n, data.others(n, tr_names)) for n in tr_names]
        carried = np.unique(np.concatenate([R.perts for R in train]))
        val = [Rows(data, n, data.others(n, tr_names), perts=carried) for n in held_names]
        val = [R for R in val if len(R)]
        model, start, history, best_ep = fit(data, train, val, gc, device, int(gc["epochs"]), seed, say=print)
        f = {"best_epoch": best_ep}
        for R in val:
            v, _ = evaluate(model, data, R, device)
            v0, _ = evaluate(start, data, R, device)
            f[R.name] = {**v, "mse_linear_start": v0["mse"], "rows": len(R)}
        folds[line] = f
        del model, start
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    rec = int(np.ceil(np.median([f["best_epoch"] for f in folds.values()])))
    (out / "summary.json").write_text(json.dumps({"folds": folds, "recommended_full_epochs": rec}, indent=2))
    print("\n=== leave-one-cell-line-out: the gene model on lines it never saw (errors per gene, all genes) ===")
    print(f"  {'unseen context':34s}{'rows':>7s}{'best ep':>8s}{'mse':>10s}{'lin start':>10s}{'zero':>10s}"
          f"{'vs lin':>8s}{'spread':>9s}{'sp zero':>9s}{'pert cos':>9s}")
    for line, f in folds.items():
        for n, v in f.items():
            if n == "best_epoch":
                continue
            print(f"  {n[:34]:34s}{v['rows']:7d}{f['best_epoch']:8d}{v['mse']:10.5f}{v['mse_linear_start']:10.5f}"
                  f"{v['mse_zero']:10.5f}{1 - v['mse'] / max(v['mse_linear_start'], 1e-12):+8.1%}"
                  f"{v['spread_mse']:9.5f}{v['spread_mse_zero']:9.5f}{v['cos_centered']:+9.3f}")
    print(f"\n  recommended gene.full_epochs (median best epoch): {rec}   written: {out / 'summary.json'}")


def run_full(cfg, contexts, run_dir: Path, device) -> None:
    gc, seed = gm.gene_cfg(cfg), int(cfg.get("seed", 0))
    data = Data(cfg, contexts, device, say=_say)
    names = [c.name for c in contexts]
    train = [Rows(data, n, data.others(n, names)) for n in names]
    epochs = int(gc["full_epochs"])
    _say(f"TRAIN  gene model on every line, exactly {epochs} epochs (gene.full_epochs)")
    model, _, history, best_ep = fit(data, train, None, gc, device, epochs, seed, say=print)
    _write(run_dir, data, names, cfg, model, history, best_ep, "full", [])
    print(f"\nwritten to {run_dir}/  (gene.pt = the final gene model; `make gene-prediction` uses it)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("holdout", "crossval", "full"), required=True)
    ap.add_argument("--holdout", default=None, metavar="CONTEXT",
                    help="holdout: the context to validate on; crossval: a context never used. "
                         "Default: data.pca.holdout")
    ap.add_argument("--exclude", nargs="+", default=None, metavar="CONTEXT")
    ap.add_argument("--eval-cells", type=int, default=None, metavar="N")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = pick_device(args.device or cfg["train"].get("device", "auto"))
    held = args.holdout or (pca_space.pca_cfg(cfg)["holdout"] or [None])[0]
    exclude = [] if args.exclude in (None, ["none"]) else list(args.exclude)
    base = Path(cfg["train"].get("out_dir") or Path("runs") / Path(args.config).stem) / "gene"
    dsmod.EAGER_BUDGET_BYTES = 0          # expression matrices stay on disk: only statistics are needed in RAM
    _say("SETUP  opening the processed contexts...")
    meta, contexts = dsmod.load_contexts(cfg["data"]["processed_dir"], exclude=exclude)
    run_dir = base / ("full" if args.mode == "full" else f"holdout_{held}" if args.mode == "holdout" else "crossval")
    if args.mode == "holdout" and held not in [c.name for c in contexts]:
        raise SystemExit(f"--holdout {held!r} is not a (non-excluded) context")
    run_dir.mkdir(parents=True, exist_ok=True)
    with _logged(run_dir, f"gene model {args.mode}" + (f" holding out {held}" if args.mode != "full" else "")):
        if exclude:
            print(f"excluded contexts: {', '.join(exclude)}")
        if args.mode == "full":
            run_full(cfg, contexts, run_dir, device)
        elif args.mode == "holdout":
            run_holdout(cfg, meta, contexts, held, run_dir, device, args.eval_cells)
        else:
            run_crossval(cfg, contexts, held, run_dir, device)


if __name__ == "__main__":
    main()
