"""One-off evaluation: a trained flow checkpoint with and without the linear population-mean
model on the genes the flow does not predict.

    make hybrid_eval [VERSION=...] [CHECKPOINT=...] [HOLDOUT=...]

Rebuilds training's own held-out scorer (`evaluation.cell_eval.ValScorer`) from the
checkpoint's saved config -- same PCA space, fingerprints, cell state, reference cells
and source cells -- then scores on the held-out line:

    identity      the source control cells unchanged
    mean_shift    the other lines' mean effect on the model genes
    flow          the checkpoint: model genes from the flow, every other gene copied from
                  the source control cell (what training reports)
    flow_linear   the SAME flow predictions on the model genes, every OTHER gene shifted by
                  the linear population-mean model (baselines/linear_population_mean.py),
                  fit on the checkpoint's training lines only

Nothing is trained or written back into the run; results go to
runs/baselines/hybrid_eval/<run>/scores.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import pca_space
from ..data.dataset import FlowDataset, load_contexts
from ..evaluation.cell_eval import MEMBERS, ValScorer
from ..models.build import build_model, pick_device
from ..models import linear_genes

OUT = Path("runs/baselines/hybrid_eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout", default="VCC25__adata_Validation")
    ap.add_argument("--eval-cells", type=int, default=None,
                    help="max cells per scored group (default: the checkpoint config's train.cell_eval_max_cells)")
    args = ap.parse_args()

    t0 = time.time()
    ckpt = Path(args.checkpoint)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tcfg = cfg["train"]
    if args.eval_cells is not None:
        tcfg["cell_eval_max_cells"] = args.eval_cells
    root = Path(cfg["data"]["processed_dir"])
    device = pick_device(tcfg.get("device", "auto"))
    seed = int(cfg.get("seed", 0))
    out = OUT / ckpt.parent.parent.name / ckpt.parent.name
    out.mkdir(parents=True, exist_ok=True)

    meta, contexts = load_contexts(root)
    by = {c.name: c for c in contexts}
    if args.holdout not in by:
        raise SystemExit(f"--holdout {args.holdout!r} is not a context: {sorted(by)}")
    V = by[args.holdout]
    train = [c for c in contexts if c.name != args.holdout]
    train_names = [c.name for c in train]
    print(f"checkpoint {ckpt} (epoch {ck.get('epoch')})  |  held out {V.name}  |  {len(train)} training contexts")
    info = json.loads((ckpt.parent / "run.json").read_text()) if (ckpt.parent / "run.json").is_file() else {}
    if V.name in info.get("train_contexts", []):
        print(f"\n  NOTE: this model was TRAINED on {V.name} (a `{info.get('mode')}` run) -- the scores below are a\n"
              f"  sanity check that it fits cells it has seen, NOT a held-out score.\n")

    # ---- the flow side: the PCA space and fingerprint scales the model was trained with
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
    if use_anchor:
        V.anchor_s, V.anchor_m = pca_space.context_anchor(art, V.name, V.fp_lookup, len(V.fp_rows), train_names, pcfg)
    trained_perts = sorted({int(p) for c in train for p in np.unique(c.pert) if p != 0})
    val_ds = FlowDataset([V], seed=seed + 1, keep_perts=np.array(trained_perts, dtype=np.int64))
    model = build_model(space.n_pcs, space.pc_sd, cfg["model"], state=use_state,
                        anchor={"gain0": [0.0, 0.0, 0.0], "scale_delta": 1.0} if use_anchor else None).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"PCA space {art.dir.name}  |  {space.n_pcs} PCs, {len(space.gene_idx):,} model genes  |  "
          f"cell state {'on' if use_state else 'off'}")
    scorer = ValScorer(cfg, train, val_ds, device, space, n_steps=int(tcfg.get("cell_eval_steps", 20)),
                       min_cells=int(tcfg.get("cell_eval_min_cells", 5)),
                       max_cells=int(tcfg.get("cell_eval_max_cells", 0)), log=lambda s: None)

    # ---- the linear side: fit on the same training lines
    gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    pert_names = pd.read_csv(root / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    groups_cfg = pcfg["cell_line_groups"]
    D, S = linear_genes.tables(train, contexts, linear_genes.table_dir(cfg), gene_names, pert_names, str(device))
    if info.get("linear_coef"):
        coef = np.asarray(info["linear_coef"], dtype=np.float64)     # the run's own fit
    else:
        coef, _ = linear_genes.fit(train, D, S, len(gene_names), lambda n: groups_cfg.get(n, n))
    lin, _, _ = linear_genes.heldout_shifts(V, train, coef, D, S, len(gene_names))
    n_other = len(V.gene_idx) - len(space.gene_idx)
    print(f"linear model: a={coef[0]:+.3f} b={coef[1]:+.3f} c={coef[2]:+.3f} e={coef[3]:+.4f}  |  "
          f"applied to the {n_other:,} non-model genes {V.name} measures")

    print(f"scoring {scorer.n_perts} knockdowns, {scorer.n_cells:,} reference cells ...")
    df = scorer.score(model, methods=("identity", "mean_shift", "flow", "flow_linear"),
                      other_shift={V.name: lin})
    df.to_csv(out / "scores.csv", index=False)
    short = [s for _, s in MEMBERS]
    print("\n  scaled (0 = no skill, 1 = perfect)   " + "".join(f"{m:>13s}" for m in df["method"]))
    for s in short:
        print(f"    {s:32s}" + "".join(f"{v:+13.3f}" for v in df[f"s_{s}"]))
    print(f"    {'AVERAGE':32s}" + "".join(f"{v:+13.3f}" for v in df["avg_score"]))
    print(f"\nwrote {out / 'scores.csv'}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
