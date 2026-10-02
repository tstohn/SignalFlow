"""The velocity field v_theta(z_t, t | fingerprints) in the shared PCA space.

Input and output are PC scores (K = data.pca.n_pcs). Internally the scores are divided
by each PC's standard deviation, so every input dimension is on a unit scale, and the
output is multiplied back -- the loss (flow.py) stays in raw PC units, i.e. gene-space
squared error inside the subspace.

Conditioning (FiLM, every block): the two perturbation fingerprints, each through its
own encoder, an is-perturbed flag (a control and an unknown knockdown both have empty
fingerprints; this tells them apart), time, and -- with `state=True` -- the SOURCE
cell's state: the mean of its k nearest control cells in the PCA space (data/pca_space.py,
data.pca.state_knn), i.e. what kind of cell this is with its sampling noise averaged
out. It stays fixed along the whole path; the moving cell itself is z_t.

ANCHORED (`anchor=...`, model.anchor: true): the linear model is a skip path next to the
network, and the network only learns the correction r:

    v = g_d * D + g_s * S + g_m * M + r
        D  the knockdown's mean effect in other cell lines (the delta fingerprint, PC units)
        S  slope of the PC scores on the knocked-out gene over this line's controls
        M  the generic knockdown response of the other lines -- perturbed cells only (a
           control cell has no knockdown; fix 2026-09-30, before it was added to every cell)
        g  = gain0 (the linear fit's a, b, c) + gain_head(h), the head zero-initialized
             (model.learn_gains: false freezes the head at zero: g = gain0 throughout)
        r  the network's own correction (see DIRECTION / MAGNITUDE below)
    D, S, M are data (never learned); gain_head and the network are learned by the same CFM
    loss. At step 0 the model predicts approximately the linear model on the model genes
    (r starts near, not exactly at, zero -- see below).

DIRECTION / MAGNITUDE (`model.mag_max`), 2026-09-28: r is NOT one raw linear readout of the
hidden state any more. A cell-eval on runs/v1_0_1/full found expr_mse blown up (4.7 vs
identity's 1.0) while pds_cosine/dir_fidelity had IMPROVED -- direction was fine, magnitude
was not, and cell_eval2's expr_mse sums per-gene counts over a group then goes through
expm1's convex blow-up, so a handful of outlier-magnitude cells (diagnosed with
test_scripts/diagnose_gain_growth.py) can dominate the whole metric even though the average
magnitude looked reasonable. So the output head is split:

    raw = out(h) * pc_sd                       -- same linear readout as before, in PC units
    direction = raw / ||raw||                  -- UNCONSTRAINED (the part that was already fine)
    magnitude = mag_max * sigmoid(mag_head(h)) -- BOUNDED: saturates at mag_max, cannot run away
    r = magnitude * direction

`mag_max` (config, PC units) is a hard cap on ||r||'s norm regardless of what the hidden
state looks like -- including hidden states an Euler rollout drifts into that the one-step
CFM loss never directly supervises. `out` keeps a small random (not zero) init so its
direction is well-defined from step 0 (normalizing an exact zero vector is singular);
`mag_head` is zero-weight with a strongly negative starting bias so magnitude starts near
zero instead -- the same "starts near the linear model" behavior as before, achieved through
the bounded scalar rather than a zero vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import FingerprintEncoder, TimeEncoder


class _ResBlock(nn.Module):
    """Pre-norm residual block with FiLM (scale and shift), zero-initialized (adaLN-zero):
    every block starts as an identity pass-through."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond(c).chunk(2, dim=-1)
        z = self.norm(h) * (1 + gamma) + beta
        return h + self.fc2(self.drop(F.silu(self.fc1(z))))


class VelocityField(nn.Module):
    MAG_INIT_BIAS = -6.0   # sigmoid(-6) = 0.0025: magnitude starts at ~0.25% of mag_max

    def __init__(self, n_pcs: int, pc_sd: torch.Tensor, hidden: int = 1024, n_blocks: int = 4,
                 pert_dim: int = 256, fp_hidden: int = 512, time_dim: int = 32, dropout: float = 0.0,
                 state: bool = False, anchor: dict | None = None, mag_max: float = 150.0,
                 learn_gains: bool = True) -> None:
        super().__init__()
        self.n_pcs = int(n_pcs)
        self.register_buffer("pc_sd", torch.as_tensor(pc_sd, dtype=torch.float32).clone())
        self.fp_delta = FingerprintEncoder(n_pcs, fp_hidden, pert_dim)
        self.fp_cov = FingerprintEncoder(n_pcs, fp_hidden, pert_dim)
        self.time_enc = TimeEncoder(time_dim)
        self.use_state = bool(state)
        self.state_enc = (nn.Sequential(nn.Linear(n_pcs, fp_hidden), nn.SiLU(), nn.Linear(fp_hidden, pert_dim))
                          if self.use_state else None)
        cond_dim = (3 if self.use_state else 2) * pert_dim + 1 + time_dim
        self.inp = nn.Sequential(nn.Linear(n_pcs, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([_ResBlock(hidden, cond_dim, dropout) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(hidden)
        # `out` is now a DIRECTION readout only (normalized in forward()): a small random init
        # gives a well-defined direction from step 0 -- zero-init would make normalize() singular.
        self.out = nn.Linear(hidden, n_pcs)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        # magnitude: a bounded scalar, saturating at mag_max; zero-init weight + negative bias
        # so it starts near zero (the "starts at no motion" behavior, now via a scalar not a
        # zero vector) and grows only as training pushes the bias/weight up.
        self.mag_max = float(mag_max)
        self.mag_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mag_head.weight)
        nn.init.constant_(self.mag_head.bias, self.MAG_INIT_BIAS)
        self.last_mag: torch.Tensor | None = None
        self.use_anchor = anchor is not None
        if self.use_anchor:
            self.register_buffer("gain0", torch.as_tensor(anchor["gain0"], dtype=torch.float32).reshape(3).clone())
            self.register_buffer("scale_delta", torch.tensor(float(anchor["scale_delta"])))
            self.gain_head = nn.Linear(hidden, 3)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)
            # model.learn_gains: false -- the head stays at zero, so g = gain0 (the linear fit) for
            # every cell; kept as a module so anchored checkpoints look the same either way
            self.learn_gains = bool(learn_gains)
            self.gain_head.requires_grad_(self.learn_gains)
            self.last_gain: torch.Tensor | None = None

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, fp: torch.Tensor,
                state: torch.Tensor | None = None, anchor: torch.Tensor | None = None) -> torch.Tensor:
        K = self.n_pcs
        parts = [
            self.fp_delta(fp[:, :K], fp[:, K : K + 1]),
            self.fp_cov(fp[:, K + 1 : 2 * K + 1], fp[:, 2 * K + 1 : 2 * K + 2]),
            fp[:, 2 * K + 2 : 2 * K + 3],
            self.time_enc(t),
        ]
        if self.use_state:
            if state is None:
                raise ValueError("this model conditions on the cell state (data.pca.state_knn > 0); "
                                 "forward() needs `state`")
            parts.append(self.state_enc(state / self.pc_sd))
        c = torch.cat(parts, dim=-1)
        h = self.inp(z_t / self.pc_sd)
        for blk in self.blocks:
            h = blk(h, c)
        h = self.norm_out(h)
        raw = self.out(h) * self.pc_sd                              # [B, K], direction candidate
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = raw / norm                                      # unconstrained
        magnitude = self.mag_max * torch.sigmoid(self.mag_head(h))  # [B, 1], bounded in (0, mag_max)
        r = magnitude * direction
        self.last_mag = magnitude.detach().mean()
        if not self.use_anchor:
            return r
        if anchor is None:
            raise ValueError("this model is anchored (model.anchor: true); forward() needs `anchor` [B, 2K]")
        D = fp[:, :K] * self.scale_delta               # 0 where there is no delta fingerprint
        S, M = anchor[:, :K], anchor[:, K:]
        # M (the generic KNOCKDOWN response) only for perturbed cells: a control cell has no
        # knockdown, and adding M there made the network learn to cancel it (it did not, 2026-09-30)
        is_pert = fp[:, 2 * K + 2 : 2 * K + 3]
        g = self.gain0 + self.gain_head(h)             # [B, 3]
        self.last_gain = g.detach().mean(0)
        return g[:, 0:1] * D + g[:, 1:2] * S + g[:, 2:3] * M * is_pert + r
