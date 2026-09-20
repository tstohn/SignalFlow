"""Per-cell-line gene-gene correlation, the perturbation's second embedding.

WHAT IT IS
    For one cell line, the Pearson correlation of the knocked-out gene with
    every gene in that line's panel, measured across that line's CONTROL cells.
    One row per perturbation, in the readout gene order -- so entry g is
    "how does gene g co-vary with the perturbed gene, in THIS cell line".

WHY IT GENERALISES WHERE THE ONE-HOT LOOKUP CANNOT
    `PertEncoder` has one free row per perturbation, learned only from cells
    carrying it, so a perturbation absent from training keeps its random init.
    This vector is not learned at all: it is computed from data. A cell line and
    a perturbation the model has never seen still get a meaningful, non-random
    input, as long as the perturbed gene is measured in that line -- which is
    what lets the network learn "shift the genes that co-vary with the target"
    rather than "recall what this particular knockout did".

CONTROL CELLS ONLY, AND THAT IS NOT A DETAIL
    Two reasons, and either alone would force it:
      1. no leakage -- correlations measured on perturbed cells would already
         carry the effect the model is asked to predict;
      2. it is all that exists at inference. For a new cell line (the VCC26
         contexts) we are handed unperturbed cells and nothing else, so an
         embedding that needed perturbed cells could not be computed at all.
    `prediction/predict.py` calls the SAME function on its input controls, so
    training and inference build this vector identically.

ONLY THE ROWS THAT ARE EVER READ
    The full matrix is gene x gene -- at 18,533 readout genes that is 343M
    entries per cell line, and all but a handful of rows would never be looked
    at. Only the row of the perturbed gene is ever an input, so `prepare` stores
    exactly the rows for the perturbations that cell line carries, and
    `predict.py` computes exactly the rows for the perturbations it is asked to
    predict. The numbers are the same either way; this is what keeps it small.

WHEN THERE IS NO ROW
    The perturbed gene has to be IN the cell line's panel to correlate with
    anything. Knocked-out genes are often outside the readout panel (see
    `vocab.py`), and a gene with no variance across controls has no correlation
    either. Those perturbations get an all-zero row and a 0 in the `ok` flag the
    model also receives, so "no information" is distinguishable from "correlates
    with nothing" -- the same discipline as the gene mask.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

MIN_CELLS = 30          # below this a correlation is mostly noise; we warn, not refuse
_EPS = 1e-8


def column_stats(X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene mean and (population) standard deviation over cells, float64."""
    n = max(X.shape[0], 1)
    mu = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    sq = np.asarray(X.multiply(X).sum(axis=0), dtype=np.float64).ravel() / n
    sd = np.sqrt(np.maximum(sq - mu * mu, 0.0))
    return mu, sd


def corr_rows(
    X: sp.csr_matrix,
    cols: np.ndarray,
    chunk: int = 256,
) -> np.ndarray:
    """Correlation of each gene in `cols` with every gene, over the rows of `X`.

    `X` is [n_cells, n_local] lognorm expression of ONE cell line's control
    cells; `cols` are local column indices. Returns [len(cols), n_local] float32.

    corr(i, j) = (E[xy] - E[x]E[y]) / (sd_x sd_y), with E[xy] from one sparse
    product so the dense [n_local, n_local] matrix is never formed. A gene with
    no variance across these cells correlates with nothing and gives 0.
    """
    cols = np.asarray(cols, dtype=np.int64)
    n_cells, n_local = X.shape
    out = np.zeros((len(cols), n_local), dtype=np.float32)
    if len(cols) == 0 or n_cells < 2:
        return out

    mu, sd = column_stats(X)
    inv_sd = np.where(sd > _EPS, 1.0 / np.maximum(sd, _EPS), 0.0)

    Xc = X.tocsr()
    Xt = Xc.T.tocsr()                       # [n_local, n_cells], for row slicing
    for i in range(0, len(cols), chunk):
        c = cols[i : i + chunk]
        ex_y = np.asarray((Xt[c] @ Xc).todense(), dtype=np.float64) / n_cells
        cov = ex_y - np.outer(mu[c], mu)
        out[i : i + chunk] = (cov * inv_sd[c][:, None] * inv_sd[None, :]).astype(np.float32)
    return np.clip(out, -1.0, 1.0, out=out)


def local_columns(pert_names, gene_to_local: dict[str, int]) -> tuple[list[int], list[int]]:
    """Split perturbations into those whose gene this panel measures, and the rest.

    Returns (positions into `pert_names`, their local column). A perturbation
    whose knocked-out gene is not a readout gene here simply has no row.
    """
    keep, cols = [], []
    for i, name in enumerate(pert_names):
        col = gene_to_local.get(str(name))
        if col is not None:
            keep.append(i)
            cols.append(col)
    return keep, cols
