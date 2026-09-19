"""One PCA basis, fit across datasets that share no fixed gene panel.

Per-context PCA (the old `prepare.py`) meant a cell line unseen during prep had
no basis to project into -- exactly the failure mode that made `ContextEncoder`
unusable on a held-out cell line. This fits ONE basis, in the union of every
gene any context measures, and every context (including one that shows up
later, after training) embeds into it the same way: `project()`.

THE IDEA
    PCA is a low-rank fit: find `scores` (cells x k) and `loadings` (k x genes)
    minimising `||X - scores @ loadings||^2`. This is the same fit with one
    change -- the error is summed only over (cell, gene) pairs that were
    actually MEASURED, never over the structural zeros a context's unmeasured
    genes would otherwise contribute. That is the same masking discipline as
    the flow-matching loss (`models/flow.py`), applied to fitting a PCA instead of a
    velocity field: a gene a context did not measure is never scored as zero,
    it is simply absent from that context's terms in the sum.

    Fitting alternates two linear solves (an EM / ALS scheme, since there is
    no closed-form SVD once entries go missing):
      1. loadings fixed -> solve each context's `scores` by ridge regression,
         using only the genes that context measures
      2. scores fixed -> solve `loadings`, per gene, using only the contexts
         that measure it
    Repeat until it stops moving. Each step is an ordinary, well-conditioned
    K x K solve (K = n_pcs, ~32) -- cheap regardless of how many genes or
    cells are involved.

    `mu`, the per-gene mean, is estimated ONCE up front (weighted by how many
    cells observe each gene) rather than re-estimated every round -- a
    deliberate simplification (`prepare.py`'s own "dumb on purpose" standard),
    not a numerical necessity.

WHY THIS MAKES A NEW CELL LINE FREE
    `project()` is exactly step 1 above, run once, against a FROZEN `loadings`
    -- a single small least-squares solve using only the genes that dataset
    measures. It needs no access to the training data or to any other
    context's blocks, and no pre-registered identity for the calling context.
    Any dataset with gene SYMBOLS this vocabulary knows can be embedded,
    whether or not it existed when `loadings` was fit.
"""

from __future__ import annotations

import numpy as np

MAX_FIT_CELLS_PER_BLOCK = 5000


def fit(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    n_genes: int,
    n_pcs: int,
    n_iter: int = 30,
    ridge: float = 1e-2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the shared basis. Returns (loadings [n_pcs, n_genes], mu [n_genes]).

    `blocks` is one `(gene_idx, X)` pair per context's CONTROL cells --
    `gene_idx` maps `X`'s local columns to the global 0..n_genes-1 index, `X`
    is that context's dense lognorm control-cell matrix. Control only, as in
    the original per-context version: at inference every trajectory starts
    from a real control cell, so that is the distribution the basis has to
    cover.
    """
    K = n_pcs
    rng = np.random.default_rng(seed)

    gene_sum = np.zeros(n_genes, dtype=np.float64)
    gene_n = np.zeros(n_genes, dtype=np.float64)
    for gene_idx, X in blocks:
        gene_sum[gene_idx] += X.sum(axis=0)
        gene_n[gene_idx] += X.shape[0]
    mu = np.divide(gene_sum, gene_n, out=np.zeros_like(gene_sum), where=gene_n > 0)
    mu = mu.astype(np.float32)

    centered = [(gi, X - mu[gi][None, :]) for gi, X in blocks]
    L = rng.normal(0.0, 0.01, size=(K, n_genes)).astype(np.float64)
    eyeK = np.eye(K)

    for _ in range(n_iter):
        # ---- scores step: one K x K solve per block ----------------------
        scores = []
        for gi, Xc in centered:
            A = L[:, gi]                                   # [K, |gi|]
            gram = A @ A.T + ridge * eyeK                  # [K, K]
            rhs = A @ Xc.T                                  # [K, n_c]
            scores.append(np.linalg.solve(gram, rhs).T)      # [n_c, K]

        # ---- loadings step: one batched (G x K x K) solve -----------------
        rhs_L = np.zeros((K, n_genes), dtype=np.float64)
        cov_L = np.zeros((n_genes, K, K), dtype=np.float64)
        for (gi, Xc), Sc in zip(centered, scores):
            rhs_L[:, gi] += Sc.T @ Xc                        # [K, |gi|]
            cov_L[gi] += Sc.T @ Sc                           # broadcast [K,K] -> [|gi|,K,K]
        cov_L += ridge * eyeK[None, :, :]
        L = np.linalg.solve(cov_L, rhs_L.T[:, :, None])[:, :, 0].T  # [K, G]

    return L.astype(np.float32), mu


def project(
    gene_idx: np.ndarray,
    X: np.ndarray,
    loadings: np.ndarray,
    mu: np.ndarray,
    ridge: float = 1e-2,
) -> np.ndarray:
    """Embed any cell x gene block into the shared space, via the frozen basis.

    The out-of-sample step: a K x K solve using only `gene_idx`'s columns of
    `loadings`. Works identically whether this data's context was in `fit`'s
    blocks or has never been seen before -- nothing here reads training data.
    """
    K = loadings.shape[0]
    A = loadings[:, gene_idx].astype(np.float64)             # [K, |gi|]
    Xc = X.astype(np.float64) - mu[gene_idx][None, :]
    gram = A @ A.T + ridge * np.eye(K)
    rhs = A @ Xc.T
    return np.linalg.solve(gram, rhs).T.astype(np.float32)   # [n, K]


def save(path, loadings: np.ndarray, mu: np.ndarray) -> None:
    np.savez_compressed(path, loadings=loadings, mu=mu)


def load(path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["loadings"], z["mu"]
