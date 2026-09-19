"""Cell state for one training run: the shared PCA basis, and every cell's state vector.

This used to happen once in `prepare`, over every context. It now happens once per
RUN, in `training/train.py`, because which contexts may be looked at depends on the
run: a hold-out run must fit the basis WITHOUT the held-out line's control cells,
or the held-out line is not really new -- its structure would already be in the
basis. The basis is saved next to the run's checkpoint, and `prediction/predict.py`
reads it from there, so the basis always travels with the model it was fit for.

    fit_basis     ONE basis, fit on the control cells of the TRAINING contexts only
    attach_state  project every cell of a context onto it, add the stored scalars,
                  and z-score against that context's own control cells

The z-scoring is per context and needs nothing from training, so a context absent
from training gets exactly the same treatment (see `prediction/predict.py`, whose
`_cell_state` must stay in step with `attach_state`).
"""

from __future__ import annotations

import numpy as np

from . import shared_pca
from .dataset import ContextStore

CHUNK = 2048


def fit_basis(
    contexts: list[ContextStore], n_genes: int, n_pcs: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the shared basis on the control cells of `contexts`. Returns (loadings, mu)."""
    rng = np.random.default_rng(seed)
    blocks = []
    for c in contexts:
        rows = c.control_rows.astype(np.int64)
        if len(rows) > shared_pca.MAX_FIT_CELLS_PER_BLOCK:
            rows = rng.choice(rows, shared_pca.MAX_FIT_CELLS_PER_BLOCK, replace=False)
        blocks.append((c.gene_idx, c.dense(rows)))
    return shared_pca.fit(blocks, n_genes=n_genes, n_pcs=n_pcs, seed=seed)


def attach_state(c: ContextStore, loadings: np.ndarray, mu: np.ndarray) -> None:
    """Set `c.state`: shared-PCA scores + the three stored scalars, z-scored per context.

    Chunked rather than densifying the whole matrix at once.
    """
    n, n_pcs = c.X.shape[0], loadings.shape[0]
    pcs = np.empty((n, n_pcs), dtype=np.float32)
    for i in range(0, n, CHUNK):
        pcs[i : i + CHUNK] = shared_pca.project(
            c.gene_idx, c.dense(np.arange(i, min(i + CHUNK, n))), loadings, mu
        )
    state = np.concatenate([pcs, c.scalars], axis=1).astype(np.float32)

    # z-score using control-cell statistics only, still per context: derivable from
    # any dataset's own control cells with no dependency on a training-time identity
    ctx_mu = state[c.control_rows].mean(axis=0)
    ctx_sd = state[c.control_rows].std(axis=0)
    ctx_sd[ctx_sd < 1e-6] = 1.0
    c.state = ((state - ctx_mu) / ctx_sd).astype(np.float32)
