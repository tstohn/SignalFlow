"""Build a model, and pick the device to run it on.

Lives with the model, not with `training/train.py`: `evaluation/` and
`prediction/` also need to construct the network to load a checkpoint into, and a
runnable script should never be imported by another.
"""

from __future__ import annotations

import torch

from .velocity import VelocityField


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_genes: int, n_perts: int, n_state: int, mcfg: dict) -> VelocityField:
    return VelocityField(
        n_genes=n_genes,
        n_perts=n_perts,
        n_state=n_state,
        hidden=int(mcfg.get("hidden", 512)),
        n_blocks=int(mcfg.get("n_blocks", 3)),
        pert_dim=int(mcfg.get("pert_dim", 64)),
        state_dim=int(mcfg.get("state_dim", 64)),
        time_dim=int(mcfg.get("time_dim", 32)),
        dropout=float(mcfg.get("dropout", 0.0)),
        head=mcfg.get("head", "plain"),
    )
