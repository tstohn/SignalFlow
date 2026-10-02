"""Build a model, and pick the device to run it on.

Lives with the model, not with `training/train.py`: `evaluation/` and `prediction/`
also construct the network to load a checkpoint into.
"""

from __future__ import annotations

import numpy as np
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


def build_model(n_pcs: int, pc_sd: np.ndarray, mcfg: dict, state: bool = False,
                anchor: dict | None = None) -> VelocityField:
    """`state`: condition on the source cell's k-NN state (data.pca.state_knn > 0).
    `anchor`: {"gain0": (a, b, c), "scale_delta": s} for the anchored model (model.anchor).
    `mcfg["mag_max"]`: hard cap (PC units) on the free residual's norm (models/velocity.py
    DIRECTION / MAGNITUDE); default 150.0 -- tune against a run's own real ||z1 - z0||
    (test_scripts/diagnose_gain_growth.py prints it) rather than trusting the default blind."""
    return VelocityField(
        n_pcs=n_pcs,
        pc_sd=torch.from_numpy(np.asarray(pc_sd, dtype=np.float32)),
        hidden=int(mcfg.get("hidden", 1024)),
        n_blocks=int(mcfg.get("n_blocks", 4)),
        pert_dim=int(mcfg.get("pert_dim", 256)),
        fp_hidden=int(mcfg.get("fp_hidden", 512)),
        time_dim=int(mcfg.get("time_dim", 32)),
        dropout=float(mcfg.get("dropout", 0.0)),
        state=state,
        anchor=anchor,
        mag_max=float(mcfg.get("mag_max", 150.0)),
        learn_gains=bool(mcfg.get("learn_gains", True)),
    )
