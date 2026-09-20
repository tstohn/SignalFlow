"""The velocity field v_theta(x_t, t | pert, cell state, mask).

Output is a per-cell velocity vector in log-normalised gene space, one entry
per readout gene, zeroed outside the calling cell's gene panel. Integrating it
from a real control cell gives the predicted perturbed cell -- so the model's
*output* is a field and the *deliverable* is cells.

`mask` is an ARGUMENT, not a lookup. There is no per-context identity anywhere
in this model: earlier versions indexed a registered `gene_mask` buffer (and a
learned `ContextEncoder`) by a trained context id, which meant a cell line
absent from that buffer had no way to be scored at all. Passing the mask
directly -- "which genes does THIS cell have" -- works for any data, whether
or not its context existed when the model was trained. See `encoders.py` and
`data/shared_pca.py` for the rest of that change.

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

from .encoders import PertCorrEncoder, PertEncoder, StateEncoder, TimeEncoder


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
        n_state: int,
        hidden: int = 512,
        n_blocks: int = 3,
        pert_dim: int = 64,
        state_dim: int = 64,
        time_dim: int = 32,
        dropout: float = 0.0,
        head: str = "plain",
        pert_corr: bool = True,
        pert_corr_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.head_kind = head

        self.pert_enc = PertEncoder(n_perts, pert_dim)
        # summed with the lookup, not concatenated: same cond_dim, and a
        # perturbation with no correlation row falls back to exactly the lookup
        self.pert_corr_enc = (
            PertCorrEncoder(n_genes, pert_dim, pert_corr_hidden) if pert_corr else None
        )
        self.state_enc = StateEncoder(n_state, state_dim)
        self.time_enc = TimeEncoder(time_dim)
        cond_dim = pert_dim + state_dim + time_dim

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

    def forward(
        self,
        x_t: torch.Tensor,     # [B, G]
        t: torch.Tensor,       # [B]
        pert: torch.Tensor,    # [B]
        state: torch.Tensor,   # [B, n_state]
        mask: torch.Tensor,    # [B, G] -- which genes THIS cell's data has
        pert_corr: torch.Tensor | None = None,    # [B, G] -- see encoders.PertCorrEncoder
        pert_corr_ok: torch.Tensor | None = None,  # [B, 1]
    ) -> torch.Tensor:
        m = mask
        h = self.inp(torch.cat([x_t * m, m], dim=-1))

        p = self.pert_enc(pert)
        if self.pert_corr_enc is not None:
            if pert_corr is None or pert_corr_ok is None:
                raise ValueError(
                    "this model was built with pert_corr=True, so forward() needs "
                    "pert_corr and pert_corr_ok (dataset.collate puts them in the batch "
                    "as 'pcorr'/'pcorr_ok'; see data/gene_corr.py)"
                )
            p = p + self.pert_corr_enc(pert_corr, pert_corr_ok, m)

        c = torch.cat(
            [
                p,
                self.state_enc(state),
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
