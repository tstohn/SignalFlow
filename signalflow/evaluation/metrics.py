"""The two hand-rolled evaluation metrics: energy distance and Pearson r.

`delta_r` (Pearson r on the shift from control), `mae` and `energy` are what
`evaluate.py` reports by default; the VCC2026 members come from `cell_eval.py`.
"""

from __future__ import annotations

import numpy as np

MAX_ENERGY_CELLS = 300


def energy_distance(X: np.ndarray, Y: np.ndarray, rng, cap: int = MAX_ENERGY_CELLS):
    def sub(A):
        return A[rng.choice(len(A), cap, replace=False)] if len(A) > cap else A

    X, Y = sub(X), sub(Y)
    d = lambda A, B: np.sqrt(  # noqa: E731
        np.maximum(
            (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T, 0.0
        )
    )
    return float(2 * d(X, Y).mean() - d(X, X).mean() - d(Y, Y).mean())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else float("nan")
