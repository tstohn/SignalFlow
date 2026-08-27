"""Batching for conditional flow matching over heterogeneous gene panels.

A training item is one *perturbed* cell x1. Its partner x0 is a control cell
drawn at random from the same context -- independent coupling, no pairing.
Control cells appear as targets too, with pert index 0, so the model learns
that "non-targeting" means near-zero net displacement (with the real
control-to-control spread still in it). That anchor matters: without it
nothing pins the magnitude scale.

Genes: each context stores a compact matrix plus `gene_idx`. Batches scatter
into the global readout space here, and carry the context id; the boolean
`gene_mask` [n_contexts, n_genes] tells the model and the loss which entries
are measurements and which are structural zeros. That distinction is the whole
reason for masking -- an unmeasured gene is not a gene measured as zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, Sampler


class ContextStore:
    """One context's arrays, held in memory."""

    def __init__(self, path: Path, index: int, name: str) -> None:
        z = np.load(path)
        self.index, self.name = index, name
        self.X = sp.csr_matrix(
            (z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"])
        )
        self.gene_idx = z["gene_idx"]
        self.pert = z["pert"]
        self.is_control = z["is_control"]
        self.state = z["state"]
        self.lib = z["lib"]
        self.control_rows = z["control_rows"]

    def dense(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[rows].todense(), dtype=np.float32)


class FlowDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        control_split: str | None = None,
        seed: int = 0,
    ) -> None:
        d = Path(processed_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        splits = json.loads((d / "splits.json").read_text())

        self.n_genes = int(self.meta["n_genes"])
        self.n_perts = int(self.meta["n_perts"])
        self.n_state = int(self.meta["n_state"])
        self.split = split
        self.rng = np.random.default_rng(seed)

        self.contexts: list[ContextStore] = [
            ContextStore(d / c["file"], c["index"], c["name"])
            for c in self.meta["contexts"]
        ]
        self.n_contexts = len(self.contexts)

        self.gene_mask = np.zeros((self.n_contexts, self.n_genes), dtype=bool)
        for c in self.contexts:
            self.gene_mask[c.index, c.gene_idx] = True

        # target rows for this split, and the pool of control cells allowed as
        # flow sources. Sourcing from the same split keeps evaluation honest.
        cs = control_split or split
        self.rows: list[np.ndarray] = []
        self.control_pool: list[np.ndarray] = []
        for c in self.contexts:
            sel = np.array(splits[c.name][split], dtype=np.int64)
            self.rows.append(sel)
            pool_all = np.array(splits[c.name][cs], dtype=np.int64)
            pool = pool_all[c.is_control[pool_all]]
            if pool.size == 0:  # tiny split -> fall back to all controls
                pool = c.control_rows.astype(np.int64)
            self.control_pool.append(pool)

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
        state = np.zeros((B, self.n_state), dtype=np.float32)
        lib0 = np.zeros(B, dtype=np.float32)

        for c_id in np.unique(ctx_ids):
            c = self.contexts[c_id]
            sel = np.flatnonzero(ctx_ids == c_id)
            tgt = rows[sel]
            src = self.rng.choice(self.control_pool[c_id], size=len(sel))
            # np.ix_ scatters the compact columns into the global gene space
            x1[np.ix_(sel, c.gene_idx)] = c.dense(tgt)
            x0[np.ix_(sel, c.gene_idx)] = c.dense(src)
            # conditioning always comes from the SOURCE cell: at inference we
            # only ever have controls, so x1's state must never leak in here.
            state[sel] = c.state[src]
            lib0[sel] = c.lib[src]

        pert = np.array(
            [self.contexts[c].pert[r] for c, r in zip(ctx_ids, rows)], dtype=np.int64
        )

        return {
            "x0": torch.from_numpy(x0),
            "x1": torch.from_numpy(x1),
            "pert": torch.from_numpy(pert),
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
