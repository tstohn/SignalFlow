"""The GENE MODEL: a per-gene DENSITY SHIFT, learned from statistics over many cells. No PCA anywhere.

    stage 1  gene model  (this file; trained by training/train_gene.py)   minutes-to-an-hour
    stage 2  flow        (optional; training/train.py, model.stage1: gene)

WHAT IT PREDICTS -- for one (cell line L, knockdown p) and EVERY gene g that L measures (up to all
18,533 readout genes; Orion and VCC25 measure ~18k, Replogle / Nadig ~8-9k):
    shift_g   how far g's distribution moves        (mean of log1p CPM, perturbed - control)
    lsr_g     how much wider / narrower it gets     (log sd ratio, perturbed vs control)
Both are learned from per-(line, knockdown, gene) STATISTICS over all of the knockdown's cells
(>= MIN_CELLS), where the single-cell noise has averaged out -- never from single cells.

HOW CELLS ARE MADE (the transport map; per gene, the optimal-transport map between two
bell-shaped distributions): every control cell i of L, gene by gene,
    x'_ig = x_ig + shift_g + (exp(lsr_g) - 1) * (x_ig - mu_g)        mu_g = L's control mean of g
so the cloud moves by shift_g and stretches by exp(lsr_g) around its new centre, and every cell
keeps its own position inside the cloud (its own noise). Then clip >= 0, expm1, x library size,
round -> counts. Controls themselves are never changed.

WHAT IT CANNOT EXPRESS: gene-gene correlations, sub-populations (only some cells respond),
responses that depend on the cell's state -- the (optional) flow on top is for those.

THE NETWORK -- ONE small MLP shared by every gene (it learns rules, not per-gene answers):
    per gene   d, d_ok   the knockdown's mean shift of g in OTHER cell lines (training data)
               v         ... and its log sd ratio there
               s         slope of g on the knocked-out gene over L's CONTROLS
               m, mv     generic knockdown response of g (mean, spread) in other lines
               mu, lsd   g's control mean and log sd in L                     (L's controls)
               is_target g is the knocked-out gene itself
    per row    s_ok, the target's control mean in L and whether L measures it, the knockdown's
               overall strength elsewhere (rms of d over L's genes) and its effect on its own
               target elsewhere
    output = linear(features) + MLP(features)    linear part initialised by least squares, MLP
             last layer zero -> at step 0 the gene model IS a (richer) linear model
NOTHING PER CELL LINE IS STORED: a never-seen line is described by its own controls (mu, lsd, s,
target expression); d / v / m / mv are per knockdown, from the training lines.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

KIND = "gene_model"
DIR = "gene_stats"              # <processed_dir>/gene_stats/<context>.npz (cached, holdout-independent)
MIN_CELLS = 20                  # cells a knockdown needs for its statistics
V0 = 0.01                       # variance floor in the log sd ratio (log1p CPM units squared)
LSR_CLIP = 1.0                  # |predicted log sd ratio| is capped: the cloud at most e x wider / narrower
BLOCK = 50_000
FEATURES = ("d", "d_ok", "v", "s", "m", "mv", "mu", "lsd", "is_target",
            "s_ok", "t_mu", "t_ok", "d_rms", "d_self", "d_self_ok")
N_FEAT = len(FEATURES)

DEFAULTS = {
    "hidden": 128,              # MLP width (shared by all genes)
    "layers": 2,
    "dropout": 0.0,
    "rows_per_step": 64,        # knockdowns per step (one context per step)
    "genes_per_step": 4096,     # random genes of that context per step
    "steps_per_epoch": 800,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-5,
    "ema_decay": 0.995,
    "epochs": 30,               # holdout / crossval: maximum (best held-out epoch kept)
    "patience": 6,
    "full_epochs": 10,          # full mode: exact (take it from `make gene-crossval`)
    "n0": 50,                   # row weight n / (n + n0): statistics over few cells count less
    "context_balance": 0.5,     # how often a context is drawn ~ rows ** (1 - alpha)
    "spread_weight": 1.0,       # loss weight of the log sd ratio next to the mean shift
    "spread": True,             # false = mean shift only (the cloud is moved, never stretched)
}


def gene_cfg(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("gene") or {})}


# ---- per-context statistics (cached; one pass over each processed file) --------------------

def _fp(c) -> dict:
    st = Path(c._path).stat()
    return {"n_cells": int(len(c.pert)), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def context_stats(c, cache: Path, say=print) -> dict:
    """Per knockdown with >= MIN_CELLS cells, over the context's LOCAL genes (all it measures):
    delta = mean - control mean, lsr = 0.5 * log((var + V0) / (control var + V0)) (both float16),
    n = cells; plus ctrl_mean, ctrl_sd over all its control cells."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{c.name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if json.loads(str(z["fp"])) == _fp(c):
            return {k: z[k] for k in ("perts", "n", "delta", "lsr", "ctrl_mean", "ctrl_sd")}
    t0 = time.time()
    perts, inv = np.unique(c.pert, return_inverse=True)
    if perts[0] != 0:
        raise SystemExit(f"{c.name}: no control cells")
    n_loc = len(c.gene_idx)
    s1 = np.zeros((len(perts), n_loc), dtype=np.float32)
    s2 = np.zeros((len(perts), n_loc), dtype=np.float32)
    n = len(c.pert)
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        X = c.csr(np.arange(lo, hi)).tocsr().astype(np.float32)
        oh = sp.csr_matrix((np.ones(hi - lo, np.float32), (inv[lo:hi], np.arange(hi - lo))), shape=(len(perts), hi - lo))
        s1 += (oh @ X).toarray()
        s2 += (oh @ X.multiply(X)).toarray()
    cnt = np.bincount(inv, minlength=len(perts)).astype(np.float64)
    mean = s1 / np.maximum(cnt, 1)[:, None]
    var = np.maximum(s2 / np.maximum(cnt, 1)[:, None] - mean.astype(np.float64) ** 2, 0.0)
    del s1, s2
    keep = np.flatnonzero((cnt >= MIN_CELLS) & (perts != 0))
    out = {"perts": perts[keep].astype(np.int32), "n": cnt[keep].astype(np.float32),
           "delta": (mean[keep] - mean[0]).astype(np.float16),
           "lsr": (0.5 * np.log((var[keep] + V0) / (var[0] + V0))).astype(np.float16),
           "ctrl_mean": mean[0].astype(np.float32), "ctrl_sd": np.sqrt(var[0]).astype(np.float32)}
    np.savez(f, **out, fp=json.dumps(_fp(c)))
    say(f"    gene statistics {c.name}: {len(keep):,} knockdowns x {n_loc:,} genes  [{time.time() - t0:.0f}s]")
    return out


def line_stats_from_controls(X_ctrl: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """(mean, sd) of every local gene over a line's control cells (log1p CPM) -- for a line whose
    statistics are not cached (a new line at predict time)."""
    X = X_ctrl.tocsr().astype(np.float64)
    m = np.asarray(X.mean(0)).ravel()
    m2 = np.asarray(X.multiply(X).mean(0)).ravel()
    return m.astype(np.float32), np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


# ---- "other lines" features: gathered from the source contexts' statistics -----------------

class Sources:
    """Knockdown statistics of the TRAINING contexts, in the global gene space, from which a
    row's d / v / m / mv are gathered (averaged over the sources that measured the gene).
    A source: perts [P] (vocab indices), delta / lsr [P, g] float16 over its genes gene_idx [g],
    optional ok [P, g] (merged tables: which entries exist)."""

    def __init__(self, G: int, n_vocab: int) -> None:
        self.G, self.n_vocab = int(G), int(n_vocab)
        self.src: dict[str, dict] = {}

    def add(self, name: str, perts, delta, lsr, gene_idx, ok=None, gen_d=None, gen_v=None) -> None:
        rowmap = np.full(self.n_vocab, -1, dtype=np.int64)
        rowmap[np.asarray(perts, dtype=np.int64)] = np.arange(len(perts))
        colmap = np.full(self.G, -1, dtype=np.int64)
        colmap[np.asarray(gene_idx, dtype=np.int64)] = np.arange(len(gene_idx))
        if gen_d is None:
            gen_d = delta.astype(np.float32).mean(0)
            gen_v = lsr.astype(np.float32).mean(0)
        self.src[name] = {"delta": delta, "lsr": lsr, "ok": ok, "rowmap": rowmap, "colmap": colmap,
                          "gen_d": np.asarray(gen_d, np.float32), "gen_v": np.asarray(gen_v, np.float32)}

    def gather(self, names, perts: np.ndarray, genes: np.ndarray):
        """For knockdowns `perts` [B] and GLOBAL genes `genes` [n]: d, v [B, n] (mean over the
        sources in `names` that have the entry), d_ok [B, n], and m, mv [n] (generic responses)."""
        B, n = len(perts), len(genes)
        ds = np.zeros((B, n), np.float32); vs = np.zeros((B, n), np.float32); cnt = np.zeros((B, n), np.float32)
        ms = np.zeros(n, np.float32); mvs = np.zeros(n, np.float32); mc = np.zeros(n, np.float32)
        for name in names:
            s = self.src[name]
            cols = s["colmap"][genes]
            cv = np.flatnonzero(cols >= 0)
            if not len(cv):
                continue
            ms[cv] += s["gen_d"][cols[cv]]; mvs[cv] += s["gen_v"][cols[cv]]; mc[cv] += 1
            rows = s["rowmap"][perts]
            rv = np.flatnonzero(rows >= 0)
            if not len(rv):
                continue
            ix = np.ix_(rows[rv], cols[cv])
            okm = np.ones((len(rv), len(cv)), np.float32) if s["ok"] is None else s["ok"][ix].astype(np.float32)
            jx = np.ix_(rv, cv)
            ds[jx] += s["delta"][ix].astype(np.float32) * okm
            vs[jx] += s["lsr"][ix].astype(np.float32) * okm
            cnt[jx] += okm
        d_ok = (cnt > 0).astype(np.float32)
        c = np.maximum(cnt, 1)
        mc = np.maximum(mc, 1)
        return ds / c, vs / c, d_ok, ms / mc, mvs / mc

    def merged(self, names, pert_names) -> dict:
        """One table over ALL the sources in `names` (what prediction needs, saved with the run):
        per knockdown the mean over the sources measuring each gene, in the global gene space."""
        perts = sorted({int(p) for nm in names for p in np.flatnonzero(self.src[nm]["rowmap"] >= 0)})
        genes = np.arange(self.G)
        d = np.zeros((len(perts), self.G), np.float16); v = np.zeros_like(d); ok = np.zeros(d.shape, bool)
        for lo in range(0, len(perts), 512):
            pp = np.array(perts[lo:lo + 512])
            dd, vv, oo, m, mv = self.gather(names, pp, genes)
            d[lo:lo + 512], v[lo:lo + 512], ok[lo:lo + 512] = dd, vv, oo > 0
        return {"pert_names": np.array([str(pert_names[p]) for p in perts]), "d": d, "v": v, "ok": ok,
                "gen_d": m, "gen_v": mv}


# ---- features of one line ----------------------------------------------------------------------

class Line:
    """What the gene model knows about ONE cell line: its genes (global ids), its controls' mean and
    sd per gene, and the slopes of its genes on each knocked-out gene over its controls (from its
    controls only). The same object for a training line, the held-out line and a new line."""

    def __init__(self, name, gene_idx, ctrl_mean, ctrl_sd, slope_perts, slope, gene_names) -> None:
        self.name = name
        self.gene_idx = np.asarray(gene_idx, dtype=np.int64)
        self.mu = np.asarray(ctrl_mean, np.float32)
        self.lsd = np.log(np.asarray(ctrl_sd, np.float32) + 0.05)
        self.slope_row = {int(p): i for i, p in enumerate(slope_perts)}
        self.slope = slope
        self.col_of = {str(gene_names[g]): j for j, g in enumerate(self.gene_idx)}

    def features(self, sources: Sources, names, perts: np.ndarray, pert_names, cols: np.ndarray | None = None):
        """X [B, n, N_FEAT] for knockdowns `perts` over this line's local genes `cols` (default all)."""
        perts = np.asarray(perts, dtype=np.int64)
        B = len(perts)
        d, v, d_ok, m, mv = sources.gather(names, perts, self.gene_idx)             # over ALL local genes
        tcol = np.array([self.col_of.get(str(pert_names[p]), -1) for p in perts], dtype=np.int64)
        s = np.zeros((B, len(self.gene_idx)), np.float32)
        s_ok = np.zeros(B, np.float32)
        for b, p in enumerate(perts):
            r = self.slope_row.get(int(p))
            if r is not None:
                s[b] = self.slope[r]
                s_ok[b] = 1.0
        n_ok = np.maximum(d_ok.sum(1), 1)
        d_rms = np.sqrt((d * d * d_ok).sum(1) / n_ok)
        has_t = tcol >= 0
        d_self = np.where(has_t, d[np.arange(B), np.maximum(tcol, 0)], 0.0)
        d_self_ok = np.where(has_t, d_ok[np.arange(B), np.maximum(tcol, 0)], 0.0)
        t_mu = np.where(has_t, self.mu[np.maximum(tcol, 0)], 0.0)
        if cols is None:
            cols = np.arange(len(self.gene_idx))
        n = len(cols)
        X = np.empty((B, n, N_FEAT), np.float32)
        X[..., 0], X[..., 1], X[..., 2], X[..., 3] = d[:, cols], d_ok[:, cols], v[:, cols], s[:, cols]
        X[..., 4], X[..., 5] = m[cols][None], mv[cols][None]
        X[..., 6], X[..., 7] = self.mu[cols][None], self.lsd[cols][None]
        X[..., 8] = (cols[None, :] == tcol[:, None]).astype(np.float32)
        for k, a in zip(range(9, 15), (s_ok, has_t.astype(np.float32), t_mu, d_rms, d_self, d_self_ok)):
            X[..., k] = np.asarray(a, np.float32)[:, None]
        return X


# ---- the network --------------------------------------------------------------------------

class GeneModel(nn.Module):
    """[..., N_FEAT] -> [..., 2] (mean shift, log sd ratio), one small network for every gene."""

    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.0,
                 feat_mean=None, feat_sd=None) -> None:
        super().__init__()
        self.hparams = dict(hidden=int(hidden), layers=int(layers), dropout=float(dropout))
        self.register_buffer("feat_mean", torch.zeros(N_FEAT) if feat_mean is None else torch.as_tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_sd", torch.ones(N_FEAT) if feat_sd is None else torch.as_tensor(feat_sd, dtype=torch.float32))
        self.lin = nn.Linear(N_FEAT, 2)
        mods, w = [], N_FEAT
        for _ in range(int(layers)):
            mods += [nn.Linear(w, hidden), nn.SiLU(), nn.Dropout(dropout)]
            w = hidden
        mods.append(nn.Linear(w, 2))
        self.mlp = nn.Sequential(*mods)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def norm(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.feat_mean) / self.feat_sd

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.norm(X)
        return self.lin(h) + self.mlp(h)


@torch.no_grad()
def predict(model: GeneModel, X: np.ndarray, device, chunk: int = 1 << 20) -> np.ndarray:
    """X [..., N_FEAT] -> [..., 2] numpy."""
    model.eval()
    flat = X.reshape(-1, N_FEAT)
    out = np.empty((flat.shape[0], 2), np.float32)
    for i in range(0, flat.shape[0], chunk):
        out[i : i + chunk] = model(torch.from_numpy(flat[i : i + chunk]).to(device)).cpu().numpy()
    return out.reshape(*X.shape[:-1], 2)


def save(path: Path, model: GeneModel, cfg: dict, **meta) -> None:
    torch.save({"kind": KIND, "model": model.state_dict(), "hparams": model.hparams, "config": cfg, **meta}, path)


def load(path: Path, device) -> tuple[GeneModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != KIND:
        raise SystemExit(f"{path} is not a gene-model checkpoint (train one with `make gene-holdout` / `make gene-full`)")
    m = GeneModel(**ck["hparams"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# ---- from predictions to counts -------------------------------------------------------------

def transform_counts(x0: np.ndarray, shift: np.ndarray, lsr: np.ndarray | None, mu: np.ndarray, lib0,
                     cap: float | None = None, dz: np.ndarray | None = None, order=None, loadings=None) -> sp.csr_matrix:
    """Counts of control cells `x0` [n, n_local] (log1p CPM) after the per-gene density shift:
    x' = x + shift + (exp(lsr) - 1)(x - mu)  (lsr None = shift only). Optional `dz` [n, K]: a
    (centered) flow's PC-space move added on the model genes `order` (stage 2)."""
    from .flow import lognorm_counts

    x = x0.astype(np.float64) + shift[None, :].astype(np.float64)
    if lsr is not None:
        r = np.exp(np.clip(lsr.astype(np.float64), -LSR_CLIP, LSR_CLIP))
        x += (r - 1.0)[None, :] * (x0 - mu[None, :])
    if dz is not None:
        x[:, order] += dz.astype(np.float64) @ loadings.astype(np.float64)
    counts = lognorm_counts(x, lib0)
    if cap is not None:
        total = counts.sum(axis=1)
        hot = total > cap
        if hot.any():
            counts[hot] = np.rint(counts[hot] * (cap / total[hot])[:, None])
    out = sp.csr_matrix(counts.astype(np.float32))
    out.eliminate_zeros()
    return out
