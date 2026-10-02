"""Conditional flow matching in the PCA space: the objective, the sampler, and the way
back to counts.

TRAINING (one step)
    z0 = PCA(control cell), z1 = PCA(its OT-paired perturbed cell), t ~ U(0, 1)
    (centered flow: z1 = PCA(perturbed cell) - that knockdown's real average shift)
    z_t = (1 - t) z0 + t z1          u = z1 - z0       (the transition vector)
    loss = MSE( v_theta(z_t, t | fingerprints, source-cell state), u )    over the K PC scores

SAMPLING
    Start at a real control cell's z0, Euler-integrate t: 0 -> 1, dz = z1 - z0.

BACK TO COUNTS: RESIDUAL, NOT A FULL INVERSE
    x_pred = x0 + dz @ loadings on the model genes -- the source cell's own log1p(CPM)
    plus the back-projected predicted shift. A full inverse (z1 @ loadings + mu) would
    throw away every cell's own noise and zeros, which K PCs cannot represent; this way
    only the perturbation effect passes through the PCA. Then un-log, clip at 0, and
    back to counts with the source cell's library size. Every non-model gene keeps the
    source cell's raw count. Used by BOTH the training-time scorer and predict.py.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .velocity import VelocityField


def cfm_loss(model: VelocityField, batch: dict[str, torch.Tensor], sigma: float = 0.0
             ) -> tuple[torch.Tensor, dict[str, float]]:
    z0, z1, fp = batch["z0"], batch["z1"], batch["fp"]
    if "center" in batch:
        # CENTERED flow (model.centered, stage 2 of the two-stage model): the target cell minus its
        # knockdown's real average shift, so the flow learns only how cells scatter around the
        # average -- the average itself is the mean model's (models/mean_model.py)
        z1 = z1 - batch["center"]
    t = torch.rand(z0.shape[0], device=z0.device)
    z_t = (1.0 - t)[:, None] * z0 + t[:, None] * z1
    if sigma > 0:
        z_t = z_t + sigma * torch.randn_like(z_t) * model.pc_sd
    u = z1 - z0
    v = model(z_t, t, fp, batch.get("state"), batch.get("anchor"))
    loss = ((v - u) ** 2).mean()
    with torch.no_grad():
        base = (u ** 2).mean()          # the loss of "predict no change"
        stats = {"loss": loss.item(), "identity_loss": base.item(),
                 "frac_var_explained": (1.0 - loss / base.clamp_min(1e-12)).item()}
    return loss, stats


@torch.no_grad()
def integrate(model: VelocityField, z0: torch.Tensor, fp: torch.Tensor, n_steps: int = 20,
              state: torch.Tensor | None = None, anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Euler-integrate from t=0 to t=1; returns z1. `state`: the source cells' k-NN state
    (fixed along the path), for a model that conditions on it."""
    was_training = model.training
    model.eval()
    z = z0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device)
        z = z + dt * model(z, t, fp, state, anchor)
    if was_training:
        model.train()
    return z


def lognorm_counts(x_log: np.ndarray, lib0: np.ndarray) -> np.ndarray:
    """log1p(CPM) -> whole, non-negative counts at the given library sizes (dense float64)."""
    cpm = np.expm1(np.clip(x_log.astype(np.float64), 0.0, None))
    return np.rint(cpm * np.asarray(lib0, dtype=np.float64)[:, None] / 1e6)


def residual_counts(x0_log: np.ndarray, model_order: np.ndarray, dz: np.ndarray | None,
                    loadings: np.ndarray, lib0: np.ndarray, cap: float | None = None,
                    shift: np.ndarray | None = None, other_shift: np.ndarray | None = None) -> sp.csr_matrix:
    """Predicted counts over the source cells' LOCAL panel.

    `x0_log` [n, n_local]   the source control cells' log1p(CPM)
    `model_order` [Gm]      local column of each model gene
    `dz` [n, K]             predicted z1 - z0 (None: no model shift)
    `shift` [Gm]            optional fixed log1p shift on the model genes (the mean_shift baseline)
    `lib0` [n]              the source cells' library sizes
    `cap`                   optional per-cell total cap (cells above it are scaled down)
    `other_shift` [n_local] optional log1p shift for the NON-model genes (e.g. the linear
                            population-mean baseline); None = they keep the source cells' counts
    Non-model genes -> the source cells' own counts (+ `other_shift`).
    """
    counts = lognorm_counts(x0_log if other_shift is None else x0_log + other_shift[None, :], lib0)
    x_m = x0_log[:, model_order].astype(np.float64)
    if dz is not None:
        x_m = x_m + dz.astype(np.float64) @ loadings.astype(np.float64)
    if shift is not None:
        x_m = x_m + shift[None, :]
    counts[:, model_order] = lognorm_counts(x_m, lib0)
    if cap is not None:
        total = counts.sum(axis=1)
        hot = total > cap
        if hot.any():
            counts[hot] = np.rint(counts[hot] * (cap / total[hot])[:, None])
    out = sp.csr_matrix(counts.astype(np.float32))
    out.eliminate_zeros()
    return out
