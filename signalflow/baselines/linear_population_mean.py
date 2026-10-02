"""A deliberately simple baseline: a linear model on POPULATION MEANS, per gene, in gene space.

    make linear_population_mean [HOLDOUT=<context>]      (default VCC25__adata_Validation)

Stand-alone: it only READS the prepared contexts (`make prepare`, stage 1) and reuses the
cell-eval helpers. Nothing in the real pipeline (prepare / train / predict) imports it.

WHAT IS PREDICTED
    For a knockdown p in cell line L, the shift of every gene's mean log1p(CPM):

        pred(L, p, g) = a * d(p, g)       mean effect of p on gene g in OTHER cell lines
                      + b * s(L, p, g)    slope of gene g on the target gene over L's CONTROL
                                          cells: cov(g, target) / var(target)
                      + c * m(g)          gene g's generic response: its average shift over
                                          all knockdowns of the other training lines
                      + e

    Four numbers (a, b, c, e), shared by every gene and knockdown, fit by weighted least
    squares on the training lines' real shifts. A feature that does not exist (knockdown
    measured in no other line, gene or target not measured in L) is 0. Training rows use
    OTHER cell lines only (data.pca.cell_line_groups: K562_essential + K562_gwps = K562,
    the VCC25 splits = H1), so the fit mimics predicting an unseen line. Every training
    context weighs the same (K562_gwps would otherwise be half of all rows).

EVALUATION  (the same cell-eval2 suite, same held-out cells as training's scorer)
    Per scored knockdown, the same seeded choice of real cells and of source control
    cells as `evaluation.cell_eval.ValScorer`. Each source cell is shifted by the SAME
    per-gene prediction in log1p(CPM), un-logged, scaled to its own library size and
    rounded -- every gene the held-out line measures, not only a shared panel. Scored
    next to:  identity   (the source cells unchanged)
              mean_shift (a=1, b=c=e=0: the other lines' mean effect, all genes)

Tables (per context: mean shifts, slopes) are cached in <processed_dir>/linear_genes/ (shared
with `model.other_genes: linear`); results go to runs/baselines/linear_population_mean/.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

from ..data.dataset import load_contexts
from ..data.vocab import CONTROL_LABEL
from ..evaluation.cell_eval import MEMBERS, _adata, _counts_from_local, _raw, _scale, _to_global, captured
from ..models.flow import lognorm_counts
from ..models.linear_genes import FEATURES, fit, heldout_shifts, table_dir, tables

OUT = Path("runs/baselines/linear_population_mean")


# the model itself (tables, features, fit, held-out shifts) lives in models/linear_genes.py,
# shared with the real pipeline's `model.other_genes: linear` option


# ---- scoring (mirrors evaluation.cell_eval.ValScorer's cell selection) -------------

def score(V, preds: dict, keep_perts: np.ndarray, pert_names, genes, cfg, eval_cells: int, log) -> pd.DataFrame:
    import cell_eval2 as ce

    tc = cfg["train"]
    min_cells, max_cells = int(tc.get("cell_eval_min_cells", 50)), int(eval_cells)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    keep = np.union1d(keep_perts, [0])
    rows = np.arange(len(V.pert))[np.isin(V.pert, keep)]
    groups = []
    for p in np.unique(V.pert[rows]):
        tgt = rows[V.pert[rows] == p]
        if len(tgt) >= min_cells:
            if max_cells and len(tgt) > max_cells:
                tgt = np.sort(rng.choice(tgt, max_cells, replace=False))
            groups.append((int(p), tgt))
    G = len(genes)
    pool = V.control_rows.astype(np.int64)
    real_blocks, labels, srcs = [], [], []
    for p, tgt in groups:
        real_blocks.append(_counts_from_local(V, tgt, G))
        labels += [str(pert_names[p])] * len(tgt)
        srcs.append(rng.choice(pool, size=len(tgt)))
    real = _adata(real_blocks, labels, genes)
    n_pert = sum(p != 0 for p, _ in groups)
    log(f"  {V.name}: {n_pert} knockdowns + controls, {real.n_obs:,} reference cells "
        f"(<= {max_cells} per group)")
    ecfg = ce.EvalConfig(metrics="vcc2026", pert_col="target", control=CONTROL_LABEL, input_type="counts",
                         target_gene_map={g: g for g in set(labels) if g != CONTROL_LABEL},
                         num_threads=-1, device="cpu")

    results = {}
    for method in ("identity", "mean_shift", "linear"):
        blocks = []
        for (p, tgt), src in zip(groups, srcs):
            x0 = V.dense(src)
            if method != "identity" and p != 0 and p in preds[method]:
                x0 = x0 + preds[method][p][None, :]
            blocks.append(_to_global(sp.csr_matrix(lognorm_counts(x0, V.lib[src]).astype(np.float32)),
                                     V.gene_idx, G))
        pred = _adata(blocks, labels, genes)
        t0 = time.time()
        with warnings.catch_warnings(), captured(lambda s: None, method):
            warnings.simplefilter("ignore")
            results[method] = _raw(pred, real, ecfg)
        log(f"  scored {method}  [{time.time() - t0:.0f}s]")
    names = [n for n, _ in MEMBERS]
    common = [n for n in names if all(np.isfinite(results[m][0][n]) for m in ("identity", "mean_shift"))]
    rows_out = []
    for method, (raw, wide, _note) in results.items():
        scaled = _scale(raw, wide, common)
        r = {"method": method}
        for name, short in MEMBERS:
            r[short], r[f"s_{short}"] = raw[name], scaled[name]
        r["avg_score"] = scaled["avg_score"]
        rows_out.append(r)
    return pd.DataFrame(rows_out)


# ---- main --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a v1 config: processed_dir, seed, cell-eval settings, cell-line groups")
    ap.add_argument("--holdout", default="VCC25__adata_Validation")
    ap.add_argument("--exclude", nargs="*", default=[], help="contexts left out entirely")
    ap.add_argument("--eval-cells", type=int, default=None,
                    help="max cells per scored group (default: train.cell_eval_max_cells of the config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["data"]["processed_dir"])
    groups_cfg = (cfg["data"].get("pca") or {}).get("cell_line_groups") or {}
    line = lambda n: groups_cfg.get(n, n)
    eval_cells = args.eval_cells if args.eval_cells is not None else int(cfg["train"].get("cell_eval_max_cells", 200))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = OUT / args.holdout
    cache = table_dir(cfg)
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "log.txt", "w")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")

    t_all = time.time()
    meta, contexts = load_contexts(root, exclude=args.exclude)
    by = {c.name: c for c in contexts}
    if args.holdout not in by:
        raise SystemExit(f"--holdout {args.holdout!r} is not a (non-excluded) context: {sorted(by)}")
    V = by[args.holdout]
    train = [c for c in contexts if c.name != args.holdout]
    gene_names = pd.read_csv(root / "gene_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    pert_names = pd.read_csv(root / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    G = len(gene_names)
    log(f"linear population-mean baseline  |  held out {V.name}  |  {len(train)} training contexts  |  device {device}")

    log("1/3  mean shifts (training contexts) and slopes (all contexts), cached in " + str(cache))
    D, S = tables(train, contexts, cache, gene_names, pert_names, device)

    log("2/3  fitting a, b, c, e (weighted least squares; each training context weighs the same)")
    coef, fit_stats = fit(train, D, S, G, line)
    for f_, c_ in zip(FEATURES, coef):
        log(f"    {f_:34s} {c_:+.4f}")
    (run / "coef.json").write_text(json.dumps({"features": FEATURES, "coef": coef.tolist(),
                                               "fit_contexts": fit_stats}, indent=2))

    log(f"3/3  cell-eval on {V.name}")
    keep_perts = np.unique(np.concatenate([np.unique(c.pert) for c in train]))
    keep_perts = keep_perts[keep_perts != 0]
    lin, dmean, v_perts = heldout_shifts(V, train, coef, D, S, G)
    preds = {"mean_shift": dmean, "linear": lin}
    log(f"  {len(v_perts)} held-out knockdowns seen in training: mean-effect feature for "
        f"{sum(bool((v != 0).any()) for v in dmean.values())}")
    df = score(V, preds, keep_perts, pert_names, gene_names, cfg, eval_cells, log)
    df.to_csv(run / "scores.csv", index=False)

    short = [s_ for _, s_ in MEMBERS]
    log("\n  scaled (0 = no skill, 1 = perfect)   " + "".join(f"{m_:>12s}" for m_ in df["method"]))
    for s_ in short:
        log(f"    {s_:32s}" + "".join(f"{v:+12.3f}" for v in df[f"s_{s_}"]))
    log(f"    {'AVERAGE':32s}" + "".join(f"{v:+12.3f}" for v in df["avg_score"]))
    log(f"\nwrote {run}/ (coef.json, scores.csv, log.txt)  [{time.time() - t_all:.0f}s total]")
    logf.close()


if __name__ == "__main__":
    main()
