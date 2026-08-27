"""Evaluate the flow model against the baselines it actually has to beat.

    python -m signalflow.evaluate --config configs/prototype.yaml \
        --checkpoint runs/prototype/best.pt --split test

METHODS COMPARED
    identity    predict no change: x1_hat = x0. The floor.
    mean_shift  x1_hat = x0 + mean delta of that (context, perturbation) on the
                TRAIN split. Deliberately strong -- it is given the answer's
                first moment. A flow model that does not beat this on the
                distributional metrics is not earning its keep, and most
                published gains evaporate against this baseline.
    flow        Euler-integrate the learned field from x0.

METRICS, per (context, perturbation), over that context's panel genes
    delta_r     Pearson r between predicted and true mean shift away from the
                control mean. The headline number: it asks whether the
                *direction of the perturbation effect* is right, with the
                large gene-to-gene baseline differences divided out. Plain
                expression correlation is near 1.0 for everything, including
                `identity`, and tells you nothing.
    mae         mean |predicted mean - true mean| per gene.
    energy      energy distance between the predicted and true cell clouds:
                2E|X-Y| - E|X-X'| - E|Y-Y'|. Zero iff the distributions match.
                This is the one that punishes collapsing to a point, and the
                reason for using a flow instead of a regressor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data.dataset import FlowDataset
from .flow import integrate
from .train import build_model, pick_device

MAX_ENERGY_CELLS = 300


def energy_distance(X: np.ndarray, Y: np.ndarray, rng, cap: int = MAX_ENERGY_CELLS):
    def sub(A):
        return A[rng.choice(len(A), cap, replace=False)] if len(A) > cap else A

    X, Y = sub(X), sub(Y)
    d = lambda A, B: np.sqrt(  # noqa: E731
        np.maximum(
            (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T, 0.0
        )
    )
    return float(2 * d(X, Y).mean() - d(X, X).mean() - d(Y, Y).mean())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed = cfg["data"]["processed_dir"]
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    train_ds = FlowDataset(processed, "train", seed=seed)
    eval_ds = FlowDataset(processed, args.split, control_split=args.split, seed=seed)

    ckpt_path = args.checkpoint or Path(cfg["train"].get("out_dir", "runs/prototype")) / "best.pt"
    device = pick_device(cfg["train"].get("device", "auto"))
    model = build_model(train_ds, cfg["model"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    pert_names = pd.read_csv(Path(processed) / "pert_vocab.csv").iloc[:, 0].tolist()
    rows = []

    for c in eval_ds.contexts:
        gi = c.gene_idx
        train_rows = train_ds.rows[c.index]
        train_ctrl = train_ds.control_pool[c.index]
        # reference control profile, from TRAIN controls (what you'd have in
        # practice), in the context's local gene space
        ctrl_mean = c.dense(train_ctrl).mean(0)

        eval_rows = eval_ds.rows[c.index]
        pool = eval_ds.control_pool[c.index]

        for p in np.unique(c.pert[eval_rows]):
            tgt = eval_rows[c.pert[eval_rows] == p]
            if len(tgt) < 5:
                continue
            src = rng.choice(pool, size=len(tgt))

            x1_true = c.dense(tgt)                      # [n, n_local]
            x0_local = c.dense(src)

            # mean_shift baseline: first moment of this pert on train cells
            tr = train_rows[train_ds.contexts[c.index].pert[train_rows] == p]
            shift = (
                c.dense(tr).mean(0) - ctrl_mean
                if len(tr) >= 5
                else np.zeros_like(ctrl_mean)
            )

            # flow: scatter to global space, integrate, gather back
            G = eval_ds.n_genes
            x0_g = np.zeros((len(src), G), dtype=np.float32)
            x0_g[:, gi] = x0_local
            with torch.no_grad():
                x1_hat = integrate(
                    model,
                    torch.from_numpy(x0_g).to(device),
                    torch.full((len(src),), int(p), dtype=torch.long, device=device),
                    torch.from_numpy(c.state[src]).to(device),
                    torch.full((len(src),), c.index, dtype=torch.long, device=device),
                    n_steps=args.n_steps,
                ).cpu().numpy()[:, gi]

            preds = {
                "identity": x0_local,
                "mean_shift": x0_local + shift[None, :],
                "flow": x1_hat,
            }
            true_delta = x1_true.mean(0) - ctrl_mean
            for name, P in preds.items():
                rows.append(
                    {
                        "context": c.name,
                        "pert": pert_names[int(p)],
                        "is_control": bool(p == 0),
                        "n_cells": len(tgt),
                        "method": name,
                        "delta_r": pearson(P.mean(0) - ctrl_mean, true_delta),
                        "mae": float(np.abs(P.mean(0) - x1_true.mean(0)).mean()),
                        "energy": energy_distance(P, x1_true, rng),
                        "true_effect": float(np.abs(true_delta).mean()),
                    }
                )

    df = pd.DataFrame(rows)
    out = Path(args.out or Path(ckpt_path).parent / f"eval_{args.split}.csv")
    df.to_csv(out, index=False)

    real = df[~df.is_control]
    summary = (
        real.groupby("method")[["delta_r", "mae", "energy"]]
        .agg(["mean", "median"])
        .round(4)
    )
    print(f"\n=== {args.split} split, {real.pert.nunique()} perturbations "
          f"across {real.context.nunique()} contexts ===")
    print(summary.to_string())
    print("\nper context (delta_r, mean):")
    print(real.pivot_table(index="context", columns="method", values="delta_r").round(3).to_string())
    ctrl = df[df.is_control]
    if len(ctrl):
        print("\nnon-targeting controls (should be ~no motion; energy is the check):")
        print(ctrl.groupby("method")[["mae", "energy"]].mean().round(4).to_string())
    print(f"\nper-perturbation rows -> {out}")


if __name__ == "__main__":
    main()
