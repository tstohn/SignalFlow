"""h5ad -> compact per-context arrays the model can stream.

One .npz per context (= per source file). Nothing is padded to the global gene
space on disk: each context keeps its own compact matrix plus `gene_idx`, the
map from its local column j to the global readout index. The scatter into
global space (and the mask that goes with it) happens per batch, in
`dataset.py`. That is what keeps this workable when the readout panel grows
from 1,213 genes to 18,533.

Layout written to <out>/:

    meta.json              vocab paths, per-context summary, prep config
    gene_vocab.csv         readout space  (index = row order)
    pert_vocab.csv         perturbation space, row 0 = "non-targeting"
    splits.json            train/val/test cell indices, per context
    contexts/<name>.npz    per context:
        X_data/X_indices/X_indptr/X_shape   lognorm expression, CSR
        gene_idx     int32 [n_local]     local column -> global gene index
        pert         int32 [n_cells]     -> pert vocab (0 = control)
        is_control   bool  [n_cells]
        state        f32   [n_cells, n_state]  cell-state conditioning vector
        lib          f32   [n_cells]     total UMI count (from raw .X)
        control_rows int32 [n_control]   row ids of control cells
        pca_components f32 [n_pcs, n_local]   \
        pca_mean       f32 [n_local]           |  so new cells can be
        state_mu       f32 [n_state]           |  projected the same way
        state_sd       f32 [n_state]          /

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
from sklearn.decomposition import PCA

from ..vocab import CONTROL_LABEL, GeneVocab, PertVocab

# cell-state = [PCs] + [these three scalars]
N_SCALAR_STATS = 3


def _context_name(path: Path) -> str:
    return path.stem.replace("_subset", "")


def _cell_state(
    X_log: sp.csr_matrix,
    X_raw: sp.csr_matrix,
    control_rows: np.ndarray,
    n_pcs: int,
) -> tuple[np.ndarray, PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Dumb-on-purpose cell state: top PCs + three scalar summaries.

    The PCA is fit on this context's *control* cells only -- at inference we
    only ever start a trajectory from a control cell, so that is the
    distribution the basis has to cover. It is context-local, which is fine
    for v0 because the model also gets a context embedding; a shared encoder
    across contexts is the obvious v1 upgrade (see models/encoders.py).
    """
    n_pcs = int(min(n_pcs, len(control_rows) - 1, X_log.shape[1] - 1))
    dense_ctrl = np.asarray(X_log[control_rows].todense(), dtype=np.float32)

    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0)
    pca.fit(dense_ctrl)

    dense_all = np.asarray(X_log.todense(), dtype=np.float32)
    pcs = pca.transform(dense_all).astype(np.float32)

    lib = np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float32)
    n_det = np.asarray((X_raw > 0).sum(axis=1)).ravel().astype(np.float32)
    mean_log = dense_all.mean(axis=1).astype(np.float32)
    scalars = np.stack([np.log1p(lib), np.log1p(n_det), mean_log], axis=1)

    state = np.concatenate([pcs, scalars], axis=1).astype(np.float32)

    # z-score using control-cell statistics only
    mu = state[control_rows].mean(axis=0)
    sd = state[control_rows].std(axis=0)
    sd[sd < 1e-6] = 1.0
    state = ((state - mu) / sd).astype(np.float32)
    return state, pca, mu.astype(np.float32), sd.astype(np.float32), lib


def _split_rows(
    pert: np.ndarray, frac: tuple[float, float, float], seed: int
) -> dict[str, np.ndarray]:
    """Stratified by perturbation, split on *cells*.

    Not on perturbations -- with a one-hot perturbation encoder a held-out
    perturbation gets an untrained embedding row, so a pert-level split cannot
    work until the pert encoder is feature-based. See README.
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("train", "val", "test")}
    for p in np.unique(pert):
        rows = np.flatnonzero(pert == p)
        rng.shuffle(rows)
        n = len(rows)
        n_tr = max(1, int(round(frac[0] * n)))
        n_va = int(round(frac[1] * n))
        # every group keeps at least one training cell
        n_va = min(n_va, max(0, n - n_tr))
        out["train"].append(rows[:n_tr])
        out["val"].append(rows[n_tr : n_tr + n_va])
        out["test"].append(rows[n_tr + n_va :])
    return {k: np.concatenate(v).astype(np.int32) for k, v in out.items()}


def prepare(cfg: dict) -> Path:
    data_cfg = cfg["data"]
    src = Path(data_cfg["source_dir"])
    out = Path(data_cfg["processed_dir"])
    (out / "contexts").mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(data_cfg.get("glob", "*.h5ad")))
    if not files:
        raise SystemExit(f"no h5ad files under {src}")

    layer = data_cfg.get("layer", "lognorm")
    n_pcs = int(data_cfg.get("n_pcs", 32))

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

    # ---- per context --------------------------------------------------
    contexts, splits = [], {}
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
        if len(control_rows) < n_pcs + 1:
            raise SystemExit(f"{name}: only {len(control_rows)} control cells")

        state, pca, mu, sd, lib = _cell_state(X_log, X_raw, control_rows, n_pcs)

        np.savez_compressed(
            out / "contexts" / f"{name}.npz",
            X_data=X_log.data,
            X_indices=X_log.indices,
            X_indptr=X_log.indptr,
            X_shape=np.array(X_log.shape, dtype=np.int64),
            gene_idx=gene_idx,
            pert=pert,
            is_control=is_control,
            state=state,
            lib=lib,
            control_rows=control_rows,
            pca_components=pca.components_.astype(np.float32),
            pca_mean=pca.mean_.astype(np.float32),
            state_mu=mu,
            state_sd=sd,
        )

        splits[name] = {
            k: v.tolist()
            for k, v in _split_rows(
                pert, tuple(data_cfg.get("split_fracs", (0.8, 0.1, 0.1))),
                int(cfg.get("seed", 0)),
            ).items()
        }
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
        "n_pcs": n_pcs,
        "n_state": n_pcs + N_SCALAR_STATS,
        "layer": layer,
        "contexts": contexts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "splits.json").write_text(json.dumps(splits))
    print(
        f"\nwrote {out}  |  {len(genes)} readout genes, {len(perts)} pert slots, "
        f"{len(contexts)} contexts, state dim {meta['n_state']}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
