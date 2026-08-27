"""The velocity field v_theta(x_t, t | pert, cell state, context).

Output is a per-cell velocity vector in log-normalised gene space, one entry
per readout gene, zeroed outside the context's gene panel. Integrating it from
a real control cell gives the predicted perturbed cell -- so the model's
*output* is a field and the *deliverable* is cells.

The gene mask enters in three places, and all three matter:
  1. the input is [x_t * mask, mask], so an unmeasured gene is distinguishable
     from a gene measured as zero;
  2. the output is multiplied by the mask, so the field never moves in
     directions the data cannot speak to;
  3. the loss averages over measured entries only (see flow.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ContextEncoder, PertEncoder, StateEncoder, TimeEncoder


class _ResBlock(nn.Module):
    """Pre-norm residual block, conditioning injected additively."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond = nn.Linear(cond_dim, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = self.norm(h) + self.cond(c)
        return h + self.fc2(self.drop(F.silu(self.fc1(z))))


class VelocityField(nn.Module):
    """head="plain"   -> v is a free vector.

    head="dirmag" -> v = softplus(m) * normalize(d), the direction/magnitude
    factorisation from model.md: one unit-norm direction head and one scalar
    magnitude head. Off by default -- get the plain version working, then
    flip this and compare, because it changes what the loss can express.
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_contexts: int,
        n_state: int,
        hidden: int = 512,
        n_blocks: int = 3,
        pert_dim: int = 64,
        state_dim: int = 64,
        ctx_dim: int = 32,
        time_dim: int = 32,
        dropout: float = 0.0,
        head: str = "plain",
        gene_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.head_kind = head

        self.pert_enc = PertEncoder(n_perts, pert_dim)
        self.state_enc = StateEncoder(n_state, state_dim)
        self.ctx_enc = ContextEncoder(n_contexts, ctx_dim)
        self.time_enc = TimeEncoder(time_dim)
        cond_dim = pert_dim + state_dim + ctx_dim + time_dim

        # gene mask lives with the model so a checkpoint is self-contained
        mask = (
            torch.ones(n_contexts, n_genes)
            if gene_mask is None
            else torch.as_tensor(gene_mask, dtype=torch.float32)
        )
        self.register_buffer("gene_mask", mask)

        self.inp = nn.Linear(2 * n_genes, hidden)
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden, cond_dim, dropout) for _ in range(n_blocks)]
        )
        self.norm_out = nn.LayerNorm(hidden)

        if head == "plain":
            self.out = nn.Linear(hidden, n_genes)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif head == "dirmag":
            self.dir_head = nn.Linear(hidden, n_genes)
            self.mag_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.dir_head.bias)
            nn.init.zeros_(self.mag_head.weight)
            nn.init.constant_(self.mag_head.bias, -3.0)  # start near no motion
        else:
            raise ValueError(f"unknown head {head!r}")

    def mask_for(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.gene_mask[ctx]

    def forward(
        self,
        x_t: torch.Tensor,     # [B, G]
        t: torch.Tensor,       # [B]
        pert: torch.Tensor,    # [B]
        state: torch.Tensor,   # [B, n_state]
        ctx: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        m = self.mask_for(ctx)
        h = self.inp(torch.cat([x_t * m, m], dim=-1))

        c = torch.cat(
            [
                self.pert_enc(pert),
                self.state_enc(state),
                self.ctx_enc(ctx),
                self.time_enc(t),
            ],
            dim=-1,
        )
        for blk in self.blocks:
            h = blk(h, c)
        h = self.norm_out(h)

        if self.head_kind == "plain":
            v = self.out(h)
        else:
            d = self.dir_head(h) * m
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            v = F.softplus(self.mag_head(h)) * d
        return v * m
