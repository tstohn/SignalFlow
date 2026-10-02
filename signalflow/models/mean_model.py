"""The MEAN MODEL: a knockdown's AVERAGE shift in a cell line, stage 1 of the two-stage model.

    stage 1  mean model   (this file; trained by training/train_mean.py)  -> WHERE the cells go on average
    stage 2  flow         (models/velocity.py, trained centered)          -> how they SCATTER around that

WHY IT EXISTS (diagnosed 2026-09-30 on runs/v1_0_2)
    The flow is trained per cell pair, where one pair differs by ~55 PC units and the knockdown's
    real average effect is ~3. The average is ~0.1% of that loss, so the flow never learned it: on
    its own training data its average was off by ~5 -- more than the effect itself. cell-eval
    scores mostly averages. So the average gets its own model, trained on AVERAGES (each
    (line, knockdown)'s mean over all its cells, where the per-cell noise has cancelled).

WHAT IT PREDICTS
    For one (cell line L, knockdown p): delta_hat [K], the average PC-score shift of p's cells vs
    L's controls. At prediction every control cell of L is moved by delta_hat (mean model alone),
    or by delta_hat + its centered flow wiggle (two-stage model). Controls themselves: nothing.

NOTHING IS STORED PER CELL LINE -- a line the model never saw is described by its own controls:
    D  p's average effect in the TRAINING lines          (training data; per knockdown, not per line)
    S  slope of L's PC scores on p's target gene over L's CONTROL cells   (L's controls)
    M  the generic knockdown response of the training lines                (training data)
    C  L's average control cell (only the first `n_c` PCs)                 (L's controls)
    e  p's target gene in L's controls: mean log1p(CPM), share of cells detecting it, measured-flag
       -- a gene that is not expressed cannot be knocked down        (L's controls)
    Everything about L comes from L's control cells, computed the same way for a training line,
    the held-out line and a brand-new line at predict time (`control_features`).

    C CAREFULLY: the training data holds only ~7 distinct cell lines, so C takes ~7 different
    values -- a network could memorise "this C = K562" (a lookup table in disguise, useless on a
    new line). So C gets few PCs, a small encoder, noise and dropout (mean.c_noise / mean.c_drop).
    M is constant within a line too, so it enters ONLY the linear part, never the network. S, D, e
    differ per (line, knockdown) -- tens of thousands of values -- and carry the network.

THE MODEL
    delta_hat = g_d * D + g_s * S + g_m * M          the linear model (gains start at a least-squares
                                                      fit on the training rows, then learned)
              + MLP( enc(D), enc(S), enc(C), enc(e) ) * out_sd     correction, last layer zero-init:
                                                      at step 0 the mean model IS the linear model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..data import pca_space

KIND = "mean_model"
DEFAULTS = {
    "n_c": 20,               # PCs of the line's average control cell the network sees
    "enc": 256,              # D / S encoder width
    "hidden": 512,           # MLP width
    "dropout": 0.1,
    "c_noise": 0.5,          # Gaussian noise on C during training, in PC standard deviations
    "c_drop": 0.3,           # probability of hiding C (zeroed) for a training row
    "ctrl_cells": 5000,      # control cells per line for C and e
    "n0": 50,                # row weight n_cells / (n_cells + n0): noisy (few-cell) averages count less
    "context_balance": 0.5,  # per-context row weight ~ n_rows ** -alpha (Orion has most of the rows)
    "batch_size": 256,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "ema_decay": 0.99,
    "epochs": 80,            # holdout / crossval: the maximum (best epoch kept)
    "full_epochs": 40,       # full mode: exact (take it from `make mean-crossval`)
    "patience": 15,          # epochs without a better held-out error before stopping
}


def mean_cfg(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("mean") or {})}


# ---- features from a line's control cells (training, held-out and predict alike) -------------

def control_features(X_ctrl: sp.csr_matrix, mcol: np.ndarray, space: pca_space.Space,
                     target_cols: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """From a sample of ONE line's control cells (local genes, log1p CPM):
    C [K]     the average control cell in the PCA space
    e [n, 3]  per target column (-1 = the line does not measure it): mean log1p(CPM), share of
              cells with a nonzero value, measured-flag."""
    X_ctrl = X_ctrl.tocsr()
    n = X_ctrl.shape[0]
    zsum = np.zeros(space.n_pcs, dtype=np.float64)
    for i in range(0, n, chunk):
        zsum += space.project(pca_space.to_model_dense(X_ctrl[i : i + chunk], mcol, len(space.gene_idx))).sum(0)
    C = (zsum / max(n, 1)).astype(np.float32)
    target_cols = np.asarray(target_cols, dtype=np.int64)
    e = np.zeros((len(target_cols), 3), dtype=np.float32)
    have = target_cols >= 0
    if have.any():
        T = X_ctrl[:, target_cols[have]]
        e[have, 0] = np.asarray(T.mean(0)).ravel()
        e[have, 1] = np.asarray((T > 0).mean(0)).ravel()
        e[have, 2] = 1.0
    return C, e


def context_inputs(art: pca_space.Artifact, c, perts, feat_names: list[str], pcfg: dict,
                   pert_names, gene_names, ctrl_cells: int, seed: int, cache: dict | None = None) -> dict:
    """The mean model's inputs for knockdowns `perts` of context `c` (a ContextStore), with D and
    M from the TRAINING contexts `feat_names` (other cell lines only, as for the fingerprints).
    D, S are [n, K] in PC units (0 where missing), d_ok / s_ok [n], M [K], C [K], e [n, 3]."""
    from ..data.hvg import sample_rows

    space = art.space
    K = space.n_pcs
    perts = np.asarray(perts, dtype=np.int64)
    one = {"delta": 1.0, "cov": 1.0}                                  # PC units, not the flow's scale
    lookup, rows = pca_space.context_fingerprints(art, c.name, np.unique(perts), feat_names, len(pert_names),
                                                  one, pcfg)
    idx = lookup[perts]
    S_all, M = pca_space.context_anchor(art, c.name, lookup, len(rows), feat_names, pcfg)
    mcol = c.model_col if c.model_col is not None else pca_space.model_cols(c.gene_idx, space.gene_idx, c.name)
    local = {str(gene_names[g]): j for j, g in enumerate(c.gene_idx)}
    tcols = np.array([local.get(str(pert_names[p]), -1) for p in perts], dtype=np.int64)
    key = (c.name, int(ctrl_cells), int(seed), hash(tcols.tobytes()))
    if cache is not None and key in cache:
        C, e = cache[key]                            # control features do not depend on feat_names
    else:
        C, e = control_features(c.csr(sample_rows(c.control_rows, ctrl_cells, seed)), mcol, space, tcols)
        if cache is not None:
            cache[key] = (C, e)
    S = S_all[idx]
    return {"D": rows[idx, :K].copy(), "d_ok": rows[idx, K].copy(), "S": S,
            "s_ok": (np.abs(S).sum(1) > 0).astype(np.float32), "M": M.astype(np.float32), "C": C, "e": e}


# ---- the network --------------------------------------------------------------------------

class MeanModel(nn.Module):
    def __init__(self, n_pcs: int, pc_sd, gain0=(0.0, 0.0, 0.0), scale_d: float = 1.0, scale_s: float = 1.0,
                 out_sd=None, n_c: int = 20, enc: int = 256, hidden: int = 512, dropout: float = 0.1,
                 c_noise: float = 0.5, c_drop: float = 0.3) -> None:
        super().__init__()
        K = int(n_pcs)
        self.hparams = dict(n_pcs=K, n_c=int(n_c), enc=int(enc), hidden=int(hidden), dropout=float(dropout),
                            c_noise=float(c_noise), c_drop=float(c_drop))
        self.K, self.n_c, self.c_noise, self.c_drop = K, int(n_c), float(c_noise), float(c_drop)
        pc_sd = torch.as_tensor(np.asarray(pc_sd, dtype=np.float32))
        self.register_buffer("c_sd", pc_sd[: self.n_c].clone().clamp_min(1e-6))
        self.register_buffer("out_sd", torch.as_tensor(np.ones(K, np.float32) if out_sd is None
                                                       else np.asarray(out_sd, np.float32)).clone())
        self.register_buffer("scale_d", torch.tensor(float(scale_d)))
        self.register_buffer("scale_s", torch.tensor(float(scale_s)))
        self.gain = nn.Parameter(torch.as_tensor(np.asarray(gain0, dtype=np.float32)).reshape(3).clone())
        self.enc_d = nn.Sequential(nn.Linear(K + 1, enc), nn.SiLU())
        self.enc_s = nn.Sequential(nn.Linear(K + 1, enc), nn.SiLU())
        self.enc_c = nn.Sequential(nn.Linear(self.n_c, 32), nn.SiLU())
        self.enc_e = nn.Sequential(nn.Linear(3, 16), nn.SiLU())
        self.mlp = nn.Sequential(nn.Linear(2 * enc + 48, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, K))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def linear(self, D, S, M) -> torch.Tensor:
        return self.gain[0] * D + self.gain[1] * S + self.gain[2] * M

    def forward(self, D, d_ok, S, s_ok, M, C, e) -> torch.Tensor:
        """D, S [B, K] PC units (0 = missing), d_ok, s_ok [B], M [B, K], C [B, >= n_c], e [B, 3]."""
        d_ok, s_ok = d_ok.reshape(-1, 1), s_ok.reshape(-1, 1)
        c = C[:, : self.n_c] / self.c_sd
        if self.training:
            c = c + self.c_noise * torch.randn_like(c)
            c = c * (torch.rand(c.shape[0], 1, device=c.device) >= self.c_drop)
        h = torch.cat([self.enc_d(torch.cat([D / self.scale_d * d_ok, d_ok], -1)),
                       self.enc_s(torch.cat([S / self.scale_s * s_ok, s_ok], -1)),
                       self.enc_c(c), self.enc_e(e)], -1)
        return self.linear(D, S, M) + self.mlp(h) * self.out_sd


def fit_gains(D, S, M, Y, w) -> np.ndarray:
    """(g_d, g_s, g_m) by weighted least squares, Y ~ g_d D + g_s S + g_m M (torch tensors, [n, K])."""
    F = [D, S, M]
    A = torch.zeros(3, 3, dtype=torch.float64, device=D.device)
    b = torch.zeros(3, dtype=torch.float64, device=D.device)
    ww = w[:, None].double()
    for i in range(3):
        b[i] = (ww * F[i].double() * Y.double()).sum()
        for j in range(i, 3):
            A[i, j] = A[j, i] = (ww * F[i].double() * F[j].double()).sum()
    return torch.linalg.solve(A + 1e-9 * torch.eye(3, dtype=torch.float64, device=D.device), b).cpu().numpy()


@torch.no_grad()
def predict(model: MeanModel, inp: dict, device, batch: int = 4096) -> np.ndarray:
    """[n, K] predicted average shifts for the rows of `inp` (context_inputs' dict: M and C are
    ONE line's vectors and are broadcast)."""
    model.eval()
    n = len(inp["d_ok"])
    out = np.zeros((n, model.K), dtype=np.float32)
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
    M, C = t(inp["M"]), t(inp["C"])
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        b = len(inp["d_ok"][sl])
        out[sl] = model(t(inp["D"][sl]), t(inp["d_ok"][sl]), t(inp["S"][sl]), t(inp["s_ok"][sl]),
                        M[None].expand(b, -1), C[None].expand(b, -1), t(inp["e"][sl])).cpu().numpy()
    return out


# ---- checkpoint -------------------------------------------------------------------------

def save(path: Path, model: MeanModel, cfg: dict, **meta) -> None:
    torch.save({"kind": KIND, "model": model.state_dict(), "hparams": model.hparams, "config": cfg, **meta}, path)


def load(path: Path, device) -> tuple[MeanModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != KIND:
        raise SystemExit(f"{path} is not a mean-model checkpoint (train one with `make mean-holdout` / `make mean-full`)")
    h = ck["hparams"]
    m = MeanModel(h["n_pcs"], np.ones(h["n_pcs"], np.float32), n_c=h["n_c"], enc=h["enc"], hidden=h["hidden"],
                  dropout=h["dropout"], c_noise=h["c_noise"], c_drop=h["c_drop"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def is_mean_checkpoint(path: Path) -> bool:
    return torch.load(path, map_location="cpu", weights_only=False).get("kind") == KIND
