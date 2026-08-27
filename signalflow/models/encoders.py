"""Conditioning encoders. Each is a small, swappable block.

The three things the velocity field is conditioned on:

  PertEncoder     which gene was knocked out
  StateEncoder    what kind of cell we started from
  time features   where along the flow we are

Every one of these is the "obviously too simple" version on purpose. The
upgrade paths are noted at each class -- they are meant to be swapped one at a
time, keeping everything else fixed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PertEncoder(nn.Module):
    """One-hot perturbation -> dense vector.

    `nn.Embedding(V, d)` is exactly `one_hot(p) @ W` with W of shape [V, d],
    just without materialising the V-wide one-hot row. V is the *perturbation*
    vocabulary (VCC26's 18,533 symbols + the control slot), not the readout
    panel -- most knocked-out genes are not in the readout panel at all.

    THE LIMIT, stated plainly: a one-hot encoder has one free row per
    perturbation, learned only from cells carrying that perturbation. A
    perturbation never seen in training keeps its random init, so this model
    cannot generalise to unseen perturbations -- not badly, but *at all*. In
    the prototype data no perturbation is shared between any two of the eight
    files, so a held-out-perturbation split would score pure noise.

    Upgrade path (this is the one that unlocks unseen perturbations): replace
    the lookup with a projection of *features* of the knocked-out gene --
    its own expression profile across control cells, an OmniPath/STRING
    network embedding, a DepMap essentiality vector. Same output shape, so
    nothing downstream changes.
    """

    def __init__(self, n_perts: int, dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_perts, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        # the control slot starts at exactly zero: "no perturbation"
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.out_dim = dim

    def forward(self, pert: torch.Tensor) -> torch.Tensor:
        return self.emb(pert)


class StateEncoder(nn.Module):
    """Cell-state summary -> dense vector.

    Input is the precomputed state vector from `data/prepare.py`: the top PCs
    of the source cell (PCA fit on that context's control cells) plus three
    scalars -- log total UMI, log genes detected, mean lognorm. That is the
    whole "cell state" for v0.

    Upgrade path: a shared encoder fit across contexts (so states are
    comparable between cell lines), or a pretrained embedding
    (scFoundation/Geneformer), or scBaseCount-derived coordinates. Swap the
    input, keep the interface.
    """

    def __init__(self, n_state: int, dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )
        self.out_dim = dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ContextEncoder(nn.Module):
    """Which dataset/cell-line this cell came from. A plain lookup.

    Upgrade path: replace with cell-line features (DepMap, baseline
    expression) so a *new* cell line is representable.
    """

    def __init__(self, n_contexts: int, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_contexts, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.out_dim = dim

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.emb(ctx)


class TimeEncoder(nn.Module):
    """Fourier features of t in [0, 1]."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
        self.register_buffer("freqs", freqs)
        self.out_dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
