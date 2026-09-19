"""h5ad -> compact per-context arrays the model can stream.

One .npz per context (= per source file). Nothing is padded to the global gene
space on disk: each context keeps its own compact matrix plus `gene_idx`, the
map from its local column j to the global readout index. The scatter into
global space (and the mask that goes with it) happens per batch, in
`dataset.py`. That is what keeps this workable when the readout panel grows
from 1,213 genes to 18,533.

PREPARE DOES NOT SPLIT AND DOES NOT FIT A PCA
    It only converts and aligns. Which cell lines train, which is held out and
    which is left out of the final model is decided by `training/train.py`
    (`train.mode`), and the shared PCA basis has to be fit WITHOUT any held-out
    line's control cells or the hold-out is not honest -- so that fit, and the
    cell state that depends on it, also live in `train`, once per run. Run
    `prepare` when you add or change a source file, not when you change a split.

Layout written to <out>/:

    meta.json               vocab paths, per-context summary
    gene_vocab.csv          readout space  (index = row order)
    pert_vocab.csv          perturbation space, row 0 = "non-targeting"
    contexts/<name>.npz     per context:
        X_data/X_indices/X_indptr/X_shape   lognorm expression, CSR
        gene_idx     int32 [n_local]     local column -> global gene index
        pert         int32 [n_cells]     -> pert vocab (0 = control)
        is_control   bool  [n_cells]
        lib          f32   [n_cells]     total UMI count (from raw .X)
        scalars      f32   [n_cells, 3]  log1p(total UMI), log1p(genes detected), mean lognorm
        control_rows int32 [n_control]   row ids of control cells

Run:
    python -m signalflow.data.prepare --config configs/prototype.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import yaml

from .vocab import CONTROL_LABEL, GeneVocab, PertVocab

N_SCALAR_STATS = 3


def _context_name(path: Path) -> str:
    return path.stem.replace("_subset", "")


def prepare(cfg: dict) -> Path:
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
        panels[_context_name(f)] = ad.read_h5ad(f, backed="r").var_names.to_list()

    gene_list = data_cfg.get("gene_vocab_csv")
    genes = (
        GeneVocab.from_csv(gene_list)
        if gene_list
        else GeneVocab.from_union(panels.values())
    )
    perts = PertVocab.from_csv(data_cfg["pert_vocab_csv"])

    genes.to_csv(out / "gene_vocab.csv", "gene_name")
    perts.to_csv(out / "pert_vocab.csv", "pert")

    # ---- one context at a time: nothing is held across files ---------------
    contexts = []
    for ctx_idx, f in enumerate(files):
        name = _context_name(f)
        a = ad.read_h5ad(f)

        X_log = sp.csr_matrix(a.layers[layer], dtype=np.float32)
        X_raw = sp.csr_matrix(a.X, dtype=np.float32)

        missing = [g for g in a.var_names if g not in genes]
        if missing:
            keep = np.array([g in genes for g in a.var_names])
            print(f"  {name}: dropping {len(missing)} genes not in gene vocab")
            X_log, X_raw = X_log[:, keep], X_raw[:, keep]
            local_genes = [g for g in a.var_names if g in genes]
        else:
            local_genes = a.var_names.to_list()
        gene_idx = np.array(genes.indices(local_genes), dtype=np.int32)

        tg = a.obs["target_gene"].astype(str).to_numpy()
        unknown = sorted({g for g in tg if g not in perts})
        if unknown:
            raise SystemExit(
                f"{name}: {len(unknown)} perturbation(s) missing from the pert "
                f"vocab, e.g. {unknown[:5]}. Point data.pert_vocab_csv at a "
                f"gene list that covers them."
            )
        pert = np.array(perts.indices(tg), dtype=np.int32)
        is_control = pert == perts.control_index
        control_rows = np.flatnonzero(is_control).astype(np.int32)
        if len(control_rows) < 2:
            raise SystemExit(f"{name}: only {len(control_rows)} control cells")

        # the three per-cell scalars of the state vector: they do not depend on any
        # PCA basis, so they are computed here once rather than in every run
        lib = np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float32)
        n_det = np.asarray((X_raw > 0).sum(axis=1)).ravel().astype(np.float32)
        # float64 sum, so training and `prediction/predict.py` agree to float32 rounding
        # instead of depending on float32 accumulation order
        mean_log = (np.asarray(X_log.sum(axis=1, dtype=np.float64)).ravel() / X_log.shape[1]).astype(np.float32)
        scalars = np.stack([np.log1p(lib), np.log1p(n_det), mean_log], axis=1).astype(np.float32)

        np.savez_compressed(
            out / "contexts" / f"{name}.npz",
            X_data=X_log.data,
            X_indices=X_log.indices,
            X_indptr=X_log.indptr,
            X_shape=np.array(X_log.shape, dtype=np.int64),
            gene_idx=gene_idx,
            pert=pert,
            is_control=is_control,
            lib=lib,
            scalars=scalars,
            control_rows=control_rows,
        )
        contexts.append(
            {
                "index": ctx_idx,
                "name": name,
                "file": f"contexts/{name}.npz",
                "source": str(f),
                "n_cells": int(X_log.shape[0]),
                "n_local_genes": int(X_log.shape[1]),
                "n_control": int(len(control_rows)),
                "perts": sorted({str(g) for g in tg if g != CONTROL_LABEL}),
            }
        )
        print(
            f"  {name}: {X_log.shape[0]} cells, {X_log.shape[1]} genes, "
            f"{len(contexts[-1]['perts'])} perts, {len(control_rows)} controls"
        )

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
