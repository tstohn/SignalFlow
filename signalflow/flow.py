"""Conditional flow matching: the objective, and the sampler that inverts it.

TRAINING (one step)
    x0 ~ control cells of a context        (independent coupling: no pairing)
    x1 ~ perturbed cells, same context, target gene p
    t  ~ U(0, 1)
    x_t = (1 - t) x0 + t x1                (+ optional sigma * noise)
    u   = x1 - x0                          (velocity of the straight path)
    loss = masked MSE( v_theta(x_t, t | p, s(x0), ctx),  u )

WHY THE LOSS IS ON THE VELOCITY, NOT ON x1
    Regressing x1 directly gives the conditional *mean* -- one point per
    perturbation, cell-to-cell heterogeneity gone. That is what GEARS-style
    models do, and it is why they score poorly on distributional metrics.
    Under the flow-matching loss the optimum is
        v*(x_t, t, c) = E[x1 - x0 | x_t, c],
    and integrating that field transports the whole control distribution onto
    the whole perturbed distribution. The spread comes out for free.

SAMPLING
    Start at a real control cell, Euler-integrate t: 0 -> 1. The endpoint is a
    predicted perturbed cell in lognorm space. `to_counts` takes it back to
    UMIs if you need them.

The mask enters as an average over measured genes only. Never mean() over G:
different contexts measure different numbers of genes, and an unmasked mean
would make loss magnitudes incomparable across them.
"""

from __future__ import annotations

import torch

from .models.velocity import VelocityField


def cfm_loss(
    model: VelocityField,
    batch: dict[str, torch.Tensor],
    sigma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    x0, x1 = batch["x0"], batch["x1"]
    pert, ctx, state = batch["pert"], batch["ctx"], batch["state"]

    m = model.mask_for(ctx)
    t = torch.rand(x0.shape[0], device=x0.device)

    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    if sigma > 0:
        # widens the region of x-space the field is trained on; helps when the
        # two endpoint clouds barely overlap. sigma=0 is the rectified-flow
        # / independent-coupling CFM case.
        x_t = x_t + sigma * torch.randn_like(x_t) * m

    u = (x1 - x0) * m
    v = model(x_t, t, pert, state, ctx)

    denom = m.sum().clamp_min(1.0)
    loss = (((v - u) ** 2) * m).sum() / denom

    with torch.no_grad():
        base = ((u**2) * m).sum() / denom          # loss of "predict no change"
        stats = {
            "loss": loss.item(),
            "identity_loss": base.item(),
            "frac_var_explained": (1.0 - loss / base.clamp_min(1e-12)).item(),
        }
    return loss, stats


@torch.no_grad()
def integrate(
    model: VelocityField,
    x0: torch.Tensor,
    pert: torch.Tensor,
    state: torch.Tensor,
    ctx: torch.Tensor,
    n_steps: int = 20,
    clamp_min: float | None = 0.0,
) -> torch.Tensor:
    """Euler-integrate the field from t=0 to t=1. Returns predicted lognorm."""
    was_training = model.training
    model.eval()

    m = model.mask_for(ctx)
    x = x0 * m
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, t, pert, state, ctx)
        if clamp_min is not None:
            # lognorm = log1p(CPM) is non-negative by construction
            x = x.clamp_min(clamp_min) * m

    if was_training:
        model.train()
    return x


def to_counts(x_lognorm: torch.Tensor, lib: torch.Tensor) -> torch.Tensor:
    """lognorm -> UMI counts, reusing the source cell's library size.

    Inverse of the CPM+log1p in the h5ads: expm1 back to CPM, rescale to the
    cell's own depth. Keeping the source cell's library size is the simple
    choice; modelling how a perturbation shifts library size is a separate
    (real) problem and is out of scope for v0.
    """
    cpm = torch.expm1(x_lognorm).clamp_min(0.0)
    return cpm * (lib[:, None] / 1e6)
