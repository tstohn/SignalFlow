"""Conditioning encoders. Each is a small, swappable block.

The three things the velocity field is conditioned on:

  PertEncoder     which gene was knocked out (a learned row per perturbation)
  PertCorrEncoder how that gene co-expresses in THIS cell line (computed, not
                  learned -- the half that survives an unseen perturbation)
  StateEncoder    what kind of cell we started from
  time features   where along the flow we are

There is deliberately no per-cell-line encoder here. A learned lookup keyed by
"which dataset/cell line" cannot represent a cell line absent at training
time -- exactly the shape of unmeasured-gene generalisation `PertEncoder`
already cannot do, applied to cell lines instead of perturbations. Cell-line
identity is meant to be entirely subsumed by `StateEncoder`'s input: the state
vector now comes from ONE shared PCA basis (`data/shared_pca.py`), fit across every
training context and reusable, unmodified, on a context that never existed at
fit time. What genes a cell's context measures still matters -- but that enters
as the `mask` argument to `VelocityField.forward`, a property of the DATA, not
a trained embedding of the context's identity.

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

    That limit is now only half the story: `PertCorrEncoder` below adds a
    computed, per-cell-line description of the same gene and is summed with this
    one, so an unseen perturbation is no longer left with nothing but its random
    row. This lookup remains the part that memorises a specific knockout well.

    Upgrade path: feed *more* features of the knocked-out gene the same way --
    an OmniPath/STRING network embedding, a DepMap essentiality vector. Same
    output shape, so nothing downstream changes.
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


class PertCorrEncoder(nn.Module):
    """The knocked-out gene's co-expression profile IN THIS CELL LINE -> dense vector.

    Input is one row of `data/gene_corr.py`: for the cell the model is looking
    at, the correlation of the perturbed gene with every readout gene, measured
    across that cell line's control cells, zero outside its panel. It is
    computed from data, never learned, so a perturbation the model never trained
    on still arrives as something meaningful rather than as a random row --
    which is exactly what `PertEncoder` alone cannot do. The two are summed, so
    this is a correction on top of the lookup: a perturbation seen often keeps
    its own learned row, an unseen one is carried by this term alone.

    `ok` is 0 when there is no row at all (the knocked-out gene is not a readout
    gene here, or is silent in the controls). The output is multiplied by it, so
    "nothing is known" contributes exactly zero rather than a zero vector that
    would read as "correlates with nothing" -- the gene-mask discipline again.

    The last layer is zero-initialised: at the start of training this term is
    exactly 0, so the run begins from the one-hot model's behaviour and moves
    away from it only as the correlations earn their place.

    Scaling matters here and is easy to get wrong: a cell line measuring 16,000
    genes would otherwise deliver ~4x the input magnitude of one measuring
    1,000, for no biological reason. Dividing by sqrt(number of measured genes)
    makes the two comparable -- the same reason the loss never means() over G.

    Upgrade path: this is a bag of correlations, order-free apart from the
    weights. A gene-axis attention over the top-k correlated genes would let it
    say "these particular genes move", which is closer to what a pathway is.
    """

    def __init__(self, n_genes: int, dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(n_genes, hidden)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.out_dim = dim

    def forward(
        self,
        corr: torch.Tensor,     # [B, G] correlations, 0 outside the panel
        ok: torch.Tensor,       # [B, 1] 1 when a row exists
        mask: torch.Tensor,     # [B, G] which genes this cell's line measures
    ) -> torch.Tensor:
        scale = mask.sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
        h = torch.nn.functional.silu(self.proj(corr / scale))
        return self.out(h) * ok


class StateEncoder(nn.Module):
    """Cell-state summary -> dense vector.

    Input is the precomputed state vector from `data/prepare.py`: the scores of
    the source cell on ONE shared PCA basis (fit across every context's control
    cells, `data/shared_pca.py`) plus three scalars -- log total UMI, log genes
    detected, mean lognorm. That is the whole "cell state" for v0, and it is
    already comparable between cell lines.

    Upgrade path: a pretrained embedding (scFoundation/Geneformer) or
    scBaseCount-derived coordinates in place of the PCA. Swap the input, keep the
    interface.
    """

    def __init__(self, n_state: int, dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )
        self.out_dim = dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


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
