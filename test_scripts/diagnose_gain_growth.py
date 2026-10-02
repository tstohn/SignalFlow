#!/usr/bin/env python3
"""Diagnostic: does the anchor gain (g_d, g_s, g_m) or the residual r blow up during Euler
integration, and how much does that grow over training epochs?

Answers the question from the 2026-09-28 conversation: expr_mse exploded on runs/v1_0_1/full
(32 epochs) while pds_cosine/dir_fidelity stayed good. The hypothesis: gain_head (an
UNCONSTRAINED linear layer on the hidden state h) is well-behaved on-manifold (where training
loss checks it) but can spike once Euler rollout drifts even slightly off the training
interpolant (something the one-step CFM loss never supervises) -- and since D/S/M are fixed,
correct-DIRECTION vectors, an inflated scalar gain multiplying them blows up MAGNITUDE while
leaving direction mostly intact. This script logs ||D||, ||S||, ||M||, ||r||, g, ||v|| and
||z_t|| at every one of the n_steps Euler steps, for a handful of real held-out cells, so we
can see WHERE in the 20 steps (if anywhere) the gain starts running away, instead of guessing.

    .venv/bin/python test_scripts/diagnose_gain_growth.py \
        --checkpoint runs/v1_0_1/full/last.pt --holdout VCC25__adata_Validation --n-cells 8

Reuses exactly the same setup as baselines/hybrid_eval.py (same PCA space, fingerprints,
anchor tables) so the numbers are directly comparable to that scores.csv run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from signalflow.data import pca_space
from signalflow.data.dataset import FlowDataset, load_contexts, densify_batch
from signalflow.models.build import build_model, pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout", default="VCC25__adata_Validation")
    ap.add_argument("--n-cells", type=int, default=8, help="perturbed cells to trace")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    root = Path(cfg["data"]["processed_dir"])
    device = pick_device(cfg["train"].get("device", "auto"))
    seed = int(cfg.get("seed", args.seed))

    meta, contexts = load_contexts(root)
    by = {c.name: c for c in contexts}
    V = by[args.holdout]
    train = [c for c in contexts if c.name != args.holdout]
    train_names = [c.name for c in train]

    info = json.loads((ckpt.parent / "run.json").read_text()) if (ckpt.parent / "run.json").is_file() else {}
    in_sample = args.holdout in info.get("train_contexts", [])
    print(f"checkpoint {ckpt} (epoch {ck.get('epoch')})  |  probing {V.name}"
          + ("  [TRAINED ON THIS CONTEXT -- in-sample check]" if in_sample else "  [held out]"))

    run_space = info.get("pca_space")
    art = pca_space.Artifact(Path(run_space) if run_space and Path(run_space).is_dir()
                             else pca_space.find(root, [V.name], cfg, contexts))
    space, pcfg = art.space, pca_space.pca_cfg(cfg)
    sc = info.get("fingerprint_scales") or pca_space.scales(art, train_names)
    V.model_col = pca_space.model_cols(V.gene_idx, space.gene_idx, V.name)
    V.fp_lookup, V.fp_rows = pca_space.context_fingerprints(art, V.name, np.unique(V.pert), train_names,
                                                            meta["n_perts"], sc, pcfg)
    use_state = any(k.startswith("state_enc.") for k in ck["model"])
    if use_state:
        V.state_rows, V.state = art.state(V.name, int(pcfg["state_knn"]))
    use_anchor = any(k.startswith("gain_head.") for k in ck["model"])
    if not use_anchor:
        raise SystemExit("this checkpoint is not anchored (no gain_head.* weights) -- nothing to diagnose")
    V.anchor_s, V.anchor_m = pca_space.context_anchor(art, V.name, V.fp_lookup, len(V.fp_rows), train_names, pcfg)

    trained_perts = sorted({int(p) for c in train for p in np.unique(c.pert) if p != 0})
    val_ds = FlowDataset([V], seed=seed + 1, keep_perts=np.array(trained_perts, dtype=np.int64))

    model = build_model(space.n_pcs, space.pc_sd, cfg["model"], state=use_state,
                        anchor={"gain0": [0.0, 0.0, 0.0], "scale_delta": 1.0}).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    K = model.n_pcs

    # ---- one real batch of N_CELLS perturbed cells (not controls), via the same collate/
    # densify path training and cell-eval use -- so z0, fp, state, anchor are exactly what
    # the model was trained/scored with.
    rng = np.random.default_rng(seed)
    rows0 = np.concatenate(val_ds.epoch_rows(0))
    pert0 = V.pert[rows0]
    pick = rows0[pert0 != 0]
    if len(pick) > args.n_cells:
        pick = rng.choice(pick, args.n_cells, replace=False)
    items = [(0, int(r)) for r in np.sort(pick)]
    raw = val_ds.collate(items, rng=rng)
    tspace = pca_space.TorchSpace(space, device)
    batch = densify_batch(raw, device, tspace)
    perts = [str(p) for p in batch["pert"].cpu().numpy()]
    print(f"tracing {len(items)} cells, perturbations: {perts}")

    z0, z1, fp = batch["z0"], batch["z1"], batch["fp"]
    u_norm = (z1 - z0).norm(dim=-1)
    print(f"REAL target ||z1 - z0|| per cell: {u_norm.cpu().numpy().round(2).tolist()}  "
          f"(mean {u_norm.mean():.3f}) -- what an ideal Euler path should sum to")
    state = batch.get("state")
    anchor = batch.get("anchor")
    D_fixed = fp[:, :K] * model.scale_delta
    S_fixed, M_fixed = anchor[:, :K], anchor[:, K:]
    print(f"fixed fingerprint norms (constant along the whole path):  "
          f"||D||={D_fixed.norm(dim=-1).mean():.3f}  ||S||={S_fixed.norm(dim=-1).mean():.3f}  "
          f"||M||={M_fixed.norm(dim=-1).mean():.3f}")

    @torch.no_grad()
    def step_forward(z_t, t):
        parts = [
            model.fp_delta(fp[:, :K], fp[:, K:K + 1]),
            model.fp_cov(fp[:, K + 1:2 * K + 1], fp[:, 2 * K + 1:2 * K + 2]),
            fp[:, 2 * K + 2:2 * K + 3],
            model.time_enc(t),
        ]
        if model.use_state:
            parts.append(model.state_enc(state / model.pc_sd))
        c = torch.cat(parts, dim=-1)
        h = model.inp(z_t / model.pc_sd)
        for blk in model.blocks:
            h = blk(h, c)
        h = model.norm_out(h)
        r = model.out(h) * model.pc_sd
        g = model.gain0 + model.gain_head(h)         # [B, 3]
        v = g[:, 0:1] * D_fixed + g[:, 1:2] * S_fixed + g[:, 2:3] * M_fixed + r
        return v, g, r

    rows = []
    z = z0.clone()
    dt = 1.0 / args.n_steps
    for i in range(args.n_steps):
        t = torch.full((z.shape[0],), i * dt, device=device)
        v, g, r = step_forward(z, t)
        rows.append({
            "step": i, "t": i * dt,
            "g_d": g[:, 0].mean().item(), "g_s": g[:, 1].mean().item(), "g_m": g[:, 2].mean().item(),
            "g_d_max": g[:, 0].abs().max().item(), "g_s_max": g[:, 1].abs().max().item(),
            "g_m_max": g[:, 2].abs().max().item(),
            "||r||_mean": r.norm(dim=-1).mean().item(), "||r||_max": r.norm(dim=-1).max().item(),
            "||v||_mean": v.norm(dim=-1).mean().item(), "||v||_max": v.norm(dim=-1).max().item(),
            "||z||_mean": z.norm(dim=-1).mean().item(),
        })
        z = z + dt * v

    final_shift = (z - z0).norm(dim=-1)
    print(f"\nPREDICTED total shift ||z_final - z0|| per cell: {final_shift.cpu().numpy().round(2).tolist()}  "
          f"(mean {final_shift.mean():.3f})  vs REAL {u_norm.mean():.3f}  "
          f"-> ratio {(final_shift.mean() / u_norm.mean()):.2f}x")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:9.3f}")
    print(df.to_string(index=False))
    out = ckpt.parent / "diagnose_gain_growth.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    g0 = df.iloc[0][["g_d", "g_s", "g_m"]].abs().sum()
    gN = df.iloc[-1][["g_d", "g_s", "g_m"]].abs().sum()
    v0, vN = df.iloc[0]["||v||_mean"], df.iloc[-1]["||v||_mean"]
    print(f"\n|g| (sum |g_d|+|g_s|+|g_m|, mean over cells): step 0 = {g0:.3f}  ->  step {args.n_steps - 1} = {gN:.3f}"
          f"  ({'GROWING' if gN > 1.5 * g0 else 'roughly stable'})")
    print(f"||v|| mean: step 0 = {v0:.3f}  ->  step {args.n_steps - 1} = {vN:.3f}"
          f"  ({'GROWING' if vN > 1.5 * v0 else 'roughly stable'})")


if __name__ == "__main__":
    main()
