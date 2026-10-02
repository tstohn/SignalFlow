"""Conditioning encoders: time, and a perturbation fingerprint.

The model conditions on the perturbation ONLY through data-derived fingerprints in the
PCA space (`data/pca_space.py`) -- no learned per-perturbation lookup, no cell-state
encoder. The cell itself enters through z_t, the velocity field's input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FingerprintEncoder(nn.Module):
    """[vector (K) | ok flag] -> dim. A missing fingerprint (all zeros, ok = 0) maps to a
    learned constant, which is exactly the "no prior" fallback."""

    def __init__(self, n_pcs: int, hidden: int = 512, dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_pcs + 1, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.out_dim = dim

    def forward(self, v: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([v * ok, ok], dim=-1))


class TimeEncoder(nn.Module):
    """Fourier features of t in [0, 1]."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000.0), dim // 2)))
        self.out_dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
