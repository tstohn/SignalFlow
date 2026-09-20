"""Batching for conditional flow matching over heterogeneous gene panels.

A training item is one *perturbed* cell x1. Its partner x0 is a control cell
drawn at random from the same context -- independent coupling, no pairing.
Control cells appear as targets too, with pert index 0, so the model learns
that "non-targeting" means near-zero net displacement (with the real
control-to-control spread still in it). That anchor matters: without it
nothing pins the magnitude scale.

WHICH CELLS GO WHERE IS NOT DECIDED HERE
    A `FlowDataset` is built from a list of contexts and uses ALL of their cells.
    There is no train/val/test split of cells: `training/train.py` decides which
    whole cell lines are training data and which (if any) is held out, per run.

Genes: each context stores a compact matrix plus `gene_idx`. Batches scatter
into the global readout space here, and each batch carries its own `mask`
tensor built from that scatter -- which entries are measurements and which are
structural zeros. That distinction is the whole reason for masking -- an
unmeasured gene is not a gene measured as zero -- and the model receives it
directly per call, never by looking up a trained context id (see
models/velocity.py, models/encoders.py). `ctx` is also carried in each batch,
but only for per-source-dataset reporting; the model never sees it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, Sampler


class ContextStore:
    """One context's arrays, held in memory. `state` is attached by `data/state.py`."""

    def __init__(self, path: Path, index: int, name: str) -> None:
        z = np.load(path)
        if "scalars" not in z.files:
            raise SystemExit(
                f"{path}: written by an older `prepare` (no `scalars`). The processed format "
                f"changed -- re-run `python -m signalflow.data.prepare --config <config>`."
            )
        self.index, self.name = index, name
        self.X = sp.csr_matrix(
            (z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"])
        )
        self.gene_idx = z["gene_idx"]
        self.pert = z["pert"]
        self.is_control = z["is_control"]
        self.lib = z["lib"]
        self.scalars = z["scalars"]
        self.control_rows = z["control_rows"]
        self.state: np.ndarray | None = None
        # this line's gene-gene correlations, one row per perturbation it carries
        # (data/gene_corr.py). Missing only in a directory written before they existed.
        self.corr = z["corr_rows"] if "corr_rows" in z.files else np.zeros((0, len(self.gene_idx)), np.float16)
        cp = z["corr_perts"] if "corr_perts" in z.files else np.zeros(0, np.int32)
        self.corr_at = {int(p): i for i, p in enumerate(cp)}

    def dense(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[rows].todense(), dtype=np.float32)

    def corr_row(self, pert: int) -> np.ndarray | None:
        """This line's correlation of `pert`'s gene with every LOCAL gene, or None.

        None means there is nothing to say: the knocked-out gene is not in this
        line's panel (most perturbed genes are not readout genes), or it has no
        variance across its controls. The caller turns that into an all-zero row
        plus a 0 in the `ok` flag -- never a silent zero.
        """
        i = self.corr_at.get(int(pert))
        return None if i is None else self.corr[i].astype(np.float32)


def load_contexts(processed_dir: str | Path) -> tuple[dict, list[ContextStore]]:
    """meta.json and every context of a processed directory, without any state yet."""
    d = Path(processed_dir)
    meta = json.loads((d / "meta.json").read_text())
    stores = [ContextStore(d / c["file"], c["index"], c["name"]) for c in meta["contexts"]]
    return meta, stores


class FlowDataset(Dataset):
    def __init__(
        self,
        contexts: list[ContextStore],
        n_genes: int,
        n_perts: int,
        seed: int = 0,
        keep_perts: np.ndarray | None = None,
    ) -> None:
        """`contexts` must already have `.state` (see `data/state.py`).

        `keep_perts`, if given, restricts the TARGET cells to those whose perturbation
        is in it (controls are always kept). The source-control pool is never
        restricted: a flow always starts from a control cell of the same context.
        """
        if not contexts:
            raise ValueError("a FlowDataset needs at least one context")
        if any(c.state is None for c in contexts):
            raise ValueError("every context needs .state; call data.state.attach_state first")

        self.contexts = contexts
        self.n_genes, self.n_perts = int(n_genes), int(n_perts)
        self.n_state = int(contexts[0].state.shape[1])
        self.n_contexts = len(contexts)
        self.rng = np.random.default_rng(seed)

        keep = None if keep_perts is None else np.union1d(np.asarray(keep_perts, dtype=np.int64), [0])
        self.rows: list[np.ndarray] = []
        self.control_pool: list[np.ndarray] = []
        for c in contexts:
            sel = np.arange(c.X.shape[0], dtype=np.int64)
            if keep is not None:
                sel = sel[np.isin(c.pert, keep)]
            self.rows.append(sel)
            self.control_pool.append(c.control_rows.astype(np.int64))

        self.items = np.concatenate(
            [
                np.stack([np.full(len(r), i, dtype=np.int64), r], axis=1)
                for i, r in enumerate(self.rows)
                if len(r)
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[int, int]:
        ctx, row = self.items[i]
        return int(ctx), int(row)

    # ---- batch assembly (called by the DataLoader) ---------------------
    def collate(self, items: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
        ctx_ids = np.array([c for c, _ in items], dtype=np.int64)
        rows = np.array([r for _, r in items], dtype=np.int64)
        B, G = len(items), self.n_genes

        x0 = np.zeros((B, G), dtype=np.float32)
        x1 = np.zeros((B, G), dtype=np.float32)
        mask = np.zeros((B, G), dtype=np.float32)
        state = np.zeros((B, self.n_state), dtype=np.float32)
        lib0 = np.zeros(B, dtype=np.float32)
        pcorr = np.zeros((B, G), dtype=np.float32)
        pcorr_ok = np.zeros((B, 1), dtype=np.float32)

        pert = np.array(
            [self.contexts[c].pert[r] for c, r in zip(ctx_ids, rows)], dtype=np.int64
        )

        for c_id in np.unique(ctx_ids):
            c = self.contexts[c_id]
            sel = np.flatnonzero(ctx_ids == c_id)
            tgt = rows[sel]
            src = self.rng.choice(self.control_pool[c_id], size=len(sel))
            # np.ix_ scatters the compact columns into the global gene space
            x1[np.ix_(sel, c.gene_idx)] = c.dense(tgt)
            x0[np.ix_(sel, c.gene_idx)] = c.dense(src)
            # which genes THIS context measures -- an argument to the model,
            # never a lookup by context identity (see models/velocity.py)
            mask[np.ix_(sel, c.gene_idx)] = 1.0
            # conditioning always comes from the SOURCE cell: at inference we
            # only ever have controls, so x1's state must never leak in here.
            state[sel] = c.state[src]
            lib0[sel] = c.lib[src]

            # the perturbation's data-derived embedding: how the knocked-out gene
            # co-varies with every gene IN THIS CELL LINE. Every cell of one
            # (context, perturbation) gets the same row -- it describes the
            # perturbation in this line, not the individual cell. Controls have no
            # knocked-out gene, so they keep the zero row and ok=0.
            for p in np.unique(pert[sel]):
                if p == 0:
                    continue
                row = c.corr_row(int(p))
                if row is None:
                    continue
                at = sel[pert[sel] == p]
                pcorr[np.ix_(at, c.gene_idx)] = row
                pcorr_ok[at] = 1.0

        return {
            "x0": torch.from_numpy(x0),
            "x1": torch.from_numpy(x1),
            "pert": torch.from_numpy(pert),
            "pcorr": torch.from_numpy(pcorr),
            "pcorr_ok": torch.from_numpy(pcorr_ok),
            "mask": torch.from_numpy(mask),
            # bookkeeping only -- which source context each row came from. NOT a model input.
            "ctx": torch.from_numpy(ctx_ids),
            "state": torch.from_numpy(state),
            "lib0": torch.from_numpy(lib0),
        }


class ContextBatchSampler(Sampler):
    """Batches drawn from a single context.

    Keeps the per-batch scatter cheap and makes every batch share one gene
    mask. Context order is shuffled each epoch so the gradient does not walk
    through the datasets in blocks.
    """

    def __init__(self, dataset: FlowDataset, batch_size: int, seed: int = 0) -> None:
        self.ds, self.bs = dataset, batch_size
        self.rng = np.random.default_rng(seed)
        self._by_ctx = [
            np.flatnonzero(dataset.items[:, 0] == c) for c in range(dataset.n_contexts)
        ]

    def __iter__(self):
        batches = []
        for idx in self._by_ctx:
            if not len(idx):
                continue
            idx = self.rng.permutation(idx)
            batches += [
                idx[i : i + self.bs].tolist() for i in range(0, len(idx), self.bs)
            ]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self) -> int:
        return sum(int(np.ceil(len(i) / self.bs)) for i in self._by_ctx if len(i))


def make_loader(
    dataset: FlowDataset, batch_size: int, seed: int = 0, shuffle: bool = True
):
    from torch.utils.data import DataLoader

    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=ContextBatchSampler(dataset, batch_size, seed),
            collate_fn=dataset.collate,
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate
    )
