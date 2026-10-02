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


def rank_offsets(X: sp.csr_matrix) -> sp.csr_matrix:
    """The Spearman transform of a sparse, non-negative matrix, kept sparse.

    Spearman correlation is Pearson correlation of the per-gene RANKS (ties get their
    average rank). In a sparse lognorm matrix every gene's zeros tie, so a gene's rank
    vector is one constant `r0` (the average rank of the zeros) everywhere except at its
    nonzero cells. Pearson is unchanged by subtracting a constant per gene, so
    corr(rank_i, rank_j) == corr(D_i, D_j) with
        D[cell, gene] = (rank - r0) / n_cells   at nonzero entries, 0 elsewhere,
    which has exactly X's sparsity pattern. `corr_rows(..., method="spearman")` is Pearson
    on this. For `z` zeros and nonzero rank `r` (1-based, among the nonzeros):
        rank - r0 = (z + r) - (z + 1) / 2 = r + (z - 1) / 2.
    Values must be >= 0 (zeros are then the smallest), which lognorm expression is.
    """
    from scipy.stats import rankdata

    n = X.shape[0]
    C = X.tocsc(copy=True)
    C.eliminate_zeros()
    if C.nnz and C.data.min() < 0:
        raise ValueError("rank_offsets needs non-negative values (zeros must be the smallest)")
    out = np.zeros(C.nnz, dtype=np.float64)
    ptr = C.indptr
    for j in range(C.shape[1]):
        lo, hi = ptr[j], ptr[j + 1]
        if hi == lo:
            continue
        r = rankdata(C.data[lo:hi], method="average")
        out[lo:hi] = (r + (n - (hi - lo) - 1) / 2.0) / n
    return sp.csc_matrix((out, C.indices, ptr), shape=C.shape).tocsr()


def corr_rows(
    X: sp.csr_matrix,
    cols: np.ndarray,
    chunk: int = 256,
    method: str = "pearson",
) -> np.ndarray:
    """Correlation of each gene in `cols` with every gene, over the rows of `X`.

    `method` is "pearson" (on the lognorm values) or "spearman" (Pearson on their ranks,
    via `rank_offsets`; the values must then be non-negative).

    `X` is [n_cells, n_local] lognorm expression of ONE cell line's control
    cells; `cols` are local column indices. Returns [len(cols), n_local] float32.

    corr(i, j) = (E[xy] - E[x]E[y]) / (sd_x sd_y), with E[xy] from one sparse
    product so the dense [n_local, n_local] matrix is never formed. A gene with
    no variance across these cells correlates with nothing and gives 0.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown correlation method {method!r}")
    cols = np.asarray(cols, dtype=np.int64)
    n_cells, n_local = X.shape
    out = np.zeros((len(cols), n_local), dtype=np.float32)
    if len(cols) == 0 or n_cells < 2:
        return out
    if method == "spearman":
        X = rank_offsets(X)

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


class StreamingCorr:
    """Chunked counterpart to `column_stats` + `corr_rows`, for control-cell blocks
    too large to hold in memory at once (see `data/prepare.py`).

    `sum`, `sumsq` (-> mu, sd) and the cross term `X[:,cols].T @ X` (-> ex_y) are all
    sums over rows, so they accumulate exactly across row-chunks -- `.add` can be
    called on successive control-row blocks of a context in any grouping, and
    `.finish` reproduces `corr_rows(X_all_control_rows, cols)` bit-for-bit up to
    float summation-order noise (irrelevant after the float16 cast `prepare.py`
    writes). Kept separate from the eager functions above, which `prediction/predict.py`
    still calls directly on its (small) input contexts.
    """

    def __init__(self, n_local: int, cols: np.ndarray) -> None:
        self.cols = np.asarray(cols, dtype=np.int64)
        self.n = 0
        self.sum = np.zeros(n_local, dtype=np.float64)
        self.sumsq = np.zeros(n_local, dtype=np.float64)
        self.cross = np.zeros((len(self.cols), n_local), dtype=np.float64)

    def add(self, chunk: sp.csr_matrix) -> None:
        """`chunk` is a block of this context's CONTROL rows only."""
        if chunk.shape[0] == 0:
            return
        self.n += chunk.shape[0]
        self.sum += np.asarray(chunk.sum(axis=0), dtype=np.float64).ravel()
        self.sumsq += np.asarray(chunk.multiply(chunk).sum(axis=0), dtype=np.float64).ravel()
        if len(self.cols):
            ct = chunk.T.tocsr()
            self.cross += np.asarray((ct[self.cols] @ chunk).todense(), dtype=np.float64)

    def finish(self) -> np.ndarray:
        """[len(cols), n_local] float32, clipped to [-1, 1] -- same contract as `corr_rows`."""
        n_local = self.sum.shape[0]
        out = np.zeros((len(self.cols), n_local), dtype=np.float32)
        if len(self.cols) == 0 or self.n < 2:
            return out
        mu, sd = self.control_stats()
        inv_sd = np.where(sd > _EPS, 1.0 / np.maximum(sd, _EPS), 0.0)
        ex_y = self.cross / self.n
        cov = ex_y - np.outer(mu[self.cols], mu)
        out[:] = (cov * inv_sd[self.cols][:, None] * inv_sd[None, :]).astype(np.float32)
        return np.clip(out, -1.0, 1.0, out=out)

    def control_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """[n_local] mean and std over every control-cell block seen -- the same
        quantities `finish` uses internally, exposed for `PertEffectAccum` below,
        which needs them for every local gene, not just `cols`.
        """
        mu = self.sum / max(self.n, 1)
        sd = np.sqrt(np.maximum(self.sumsq / max(self.n, 1) - mu * mu, 0.0))
        return mu, sd


class PertEffectAccum:
    """How much each gene moves under perturbation, relative to its own control-cell
    noise -- a per-context, per-gene standardized effect size, meant as a TRAINING
    LOSS WEIGHT (see `models/flow.cfm_loss`'s `gene_weight`), not a correlation.

    WHY STANDARDIZED, NOT RAW VARIANCE OF THE RESPONSE
        A gene that is simply noisy (high variance in both control AND perturbed
        cells, no real signal) would get the same or a higher weight than a gene
        that moves a little but very reliably, under raw variance. Dividing by the
        control-cell std of the same gene turns "how much did this gene move" into
        "how much did it move relative to how much it jitters on its own" -- the
        same standardization a z-score or a t-statistic uses, and the reason a
        noisy gene does NOT get upweighted just for being noisy.

    PER PERTURBATION, THEN AVERAGED -- not perturbed cells pooled together
        Exactly the same reasoning as `couple.py`'s per-perturbation OT groups:
        pooling every perturbed cell into one mean would answer "what is the
        average effect of an average perturbation", dominated by whichever
        perturbations have the most cells, not "does this gene tend to respond".
        A perturbation with too few cells (`min_cells`) is left out of the average
        for that context rather than contributing a noisy per-perturbation mean.

    Computed PER CONTEXT here (this class doesn't know about other contexts); the
    caller aggregates across the TRAINING contexts of a given run into one global,
    holdout-respecting weight -- see `data/state.aggregate_gene_weight`.
    """

    def __init__(self, n_local: int, n_pert_vocab: int, perts: np.ndarray) -> None:
        self.perts = np.asarray(perts, dtype=np.int64)                  # local pert ids to track
        self.lut = np.full(n_pert_vocab, -1, dtype=np.int64)            # global pert id -> row in self.sum
        self.lut[self.perts] = np.arange(len(self.perts))
        self.sum = np.zeros((len(self.perts), n_local), dtype=np.float64)
        self.n = np.zeros(len(self.perts), dtype=np.int64)

    def add(self, chunk: sp.csr_matrix, pert_chunk: np.ndarray) -> None:
        """`chunk` is any row-block; `pert_chunk` is its pert id per row (0 = control,
        already excluded by `self.lut[0] == -1` unless `perts` explicitly included it).
        """
        if chunk.shape[0] == 0:
            return
        pos = self.lut[np.asarray(pert_chunk, dtype=np.int64)]
        keep = pos >= 0
        if not keep.any():
            return
        pos_keep = pos[keep]
        sub = chunk[keep]
        k = len(self.perts)
        ind = sp.csr_matrix(
            (np.ones(len(pos_keep)), (np.arange(len(pos_keep)), pos_keep)),
            shape=(len(pos_keep), k),
        )
        self.sum += np.asarray((ind.T @ sub).todense(), dtype=np.float64)
        self.n += np.bincount(pos_keep, minlength=k)

    def finish(self, control_mu: np.ndarray, control_sd: np.ndarray, min_cells: int = MIN_CELLS) -> np.ndarray:
        """[n_local] float32: mean, over perturbations with >= min_cells, of
        |perturbed mean - control mean| / control std. 0 where no perturbation
        qualified (e.g. every perturbation in this context is rare).
        """
        n_local = control_mu.shape[0]
        inv_sd = np.where(control_sd > _EPS, 1.0 / np.maximum(control_sd, _EPS), 0.0)
        qualifies = self.n >= min_cells
        if not qualifies.any():
            return np.zeros(n_local, dtype=np.float32)
        means = self.sum[qualifies] / self.n[qualifies][:, None]
        effect = np.abs(means - control_mu[None, :]) * inv_sd[None, :]
        return effect.mean(axis=0).astype(np.float32)


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
