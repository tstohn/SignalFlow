"""The shared PCA space the model lives in, and the two perturbation fingerprints.

Built by `make prepare` (stage 3), per held-out set, into
<processed_dir>/pca_space/<label>/, and read by `training/train.py`.

MODEL GENES
    `data.pca.model_genes: shared` = the genes EVERY context measures (the intersection of
    all contexts' panels, ~6.1k here). Every context therefore measures every model gene,
    so a plain PCA works and no gene mask is needed anywhere downstream. Genes outside
    this set are never modelled: at evaluation/prediction time they are copied from the
    source control cell's raw counts.

THE BASIS
    PCA of log1p(CPM) over the model genes, fit on up to `fit_cells` cells per context --
    control AND perturbed, so directions that separate perturbed from control cells are
    in the basis -- from every context except the held-out ones (`data.pca.holdout`,
    default VCC25__adata_Validation). Exact: eigendecomposition of the gene x gene
    covariance, accumulated in chunks. Not whitened: an MSE on raw PC scores is the
    gene-space MSE restricted to the subspace.

FINGERPRINT 1: the perturbation's average effect ELSEWHERE (delta)
    Per (context, perturbation) with >= min_cells cells: mean log1p(CPM) of its cells
    minus the context's control mean, over the model genes, projected with the loadings
    (no mean subtracted -- a difference of two cells' worth of expression is a direction,
    not a cell). Only for fit contexts: a held-out context's own effects are never
    computed. A cell's fingerprint is the average of this over the TRAINING contexts of
    OTHER cell lines (`cell_line_groups`), so training sees what testing sees: an effect
    measured in some other line, never its own answer.

FINGERPRINT 2: covariance with the knocked-out gene (cov)
    Per context, for every perturbation whose target gene the context measures: the
    covariance, over the context's CONTROL cells, of the target gene's log1p(CPM) with
    every model gene -- projected into the PCA space. Projection is linear, so this is
    cov(target, PC score) directly; the gene x gene matrix is never formed. Uses control
    cells only, so it exists for held-out lines and new cell lines alike (predict.py
    computes it from its input controls) and carries no outcome information.

CELL STATE (state_knn)
    Per context, for EVERY control cell: the mean PC score of its `state_knn` nearest
    control cells of the same context (itself included), in the model's PCA space --
    the cell's state with most of its single-cell sampling noise averaged out. Stored
    in state/<context>__k<k>.npz and handed to the model as conditioning for the SOURCE
    cell of each flow. Controls only, so it exists for held-out lines too, and
    predict.py computes it the same way (`knn_mean`) from its input controls. Its own
    stage: changing `state_knn` recomputes only these tables, not the basis.

Both fingerprints are scaled by one global scalar each (per-element RMS over the
training contexts), so the model sees them on a unit scale. Missing -> zeros + ok flag 0.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch

DIR = "pca_space"
DEFAULTS = {
    "model_genes": "shared",
    "n_pcs": 512,
    "fit_cells": 20000,          # cells per context for the PCA fit (control + perturbed)
    "holdout": ["VCC25__adata_Validation"],
    "min_cells": 20,             # cells a perturbation needs for a delta fingerprint
    "cov_cells": 20000,          # control cells per context for the covariance fingerprint
    "cell_line_groups": {},      # context -> cell line; unlisted contexts are their own line
    "state_knn": 50,             # cell-state conditioning: mean of each control cell's k nearest
                                 #   controls (itself included) in the PCA space; 0 = off
}
PARAM_KEYS = ("model_genes", "n_pcs", "fit_cells", "min_cells", "cov_cells")
BLOCK_ROWS = 50_000


def pca_cfg(cfg: dict) -> dict:
    out = {**DEFAULTS, **(cfg.get("data", {}).get("pca") or {})}
    if out["model_genes"] != "shared":
        raise SystemExit("data.pca.model_genes: only 'shared' is implemented (a gene panel that "
                         "differs between contexts needs a masked PCA -- see data/shared_pca.py)")
    out["holdout"] = sorted(out["holdout"] or [])
    out["cell_line_groups"] = dict(out["cell_line_groups"] or {})
    return out


def cell_line(name: str, pcfg: dict) -> str:
    return pcfg["cell_line_groups"].get(name, name)


def fp_width(n_pcs: int) -> int:
    """[delta (K) | delta_ok | cov (K) | cov_ok | is_perturbed]"""
    return 2 * n_pcs + 3


# ---- genes ----------------------------------------------------------------------

def shared_gene_idx(processed_dir: Path, meta: dict) -> np.ndarray:
    """Sorted GLOBAL indices of the genes every context in `meta` measures."""
    shared = None
    for c in meta["contexts"]:
        with h5py.File(Path(processed_dir) / c["file"], "r") as h:
            g = set(h["gene_idx"][:].tolist())
        shared = g if shared is None else shared & g
    return np.array(sorted(shared), dtype=np.int64)


def model_cols(gene_idx: np.ndarray, model_idx: np.ndarray, where: str) -> np.ndarray:
    """[n_local] int32: local column -> model gene position, -1 for non-model genes.
    Raises if the panel lacks any model gene."""
    pos = {int(g): j for j, g in enumerate(model_idx)}
    out = np.array([pos.get(int(g), -1) for g in gene_idx], dtype=np.int32)
    have = int((out >= 0).sum())
    if have != len(model_idx):
        raise SystemExit(f"{where}: measures only {have}/{len(model_idx)} of the model genes; the PCA "
                         f"space needs all of them (data.pca.model_genes: shared)")
    return out


def local_model_order(mcol: np.ndarray) -> np.ndarray:
    """[Gm] local columns in model-gene order (inverse of `model_cols`)."""
    order = np.empty(int((mcol >= 0).sum()), dtype=np.int64)
    at = np.flatnonzero(mcol >= 0)
    order[mcol[at]] = at
    return order


def to_model_dense(X: sp.csr_matrix, mcol: np.ndarray, n_model: int) -> np.ndarray:
    """Local CSR rows -> dense [n, Gm] float32 in model-gene order, without a column slice."""
    X = X.tocsr()
    m = mcol[X.indices]
    keep = m >= 0
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))[keep]
    out = np.zeros((X.shape[0], n_model), dtype=np.float32)
    out[rows, m[keep]] = X.data[keep]
    return out


# ---- the basis ------------------------------------------------------------------

class Space:
    """loadings [K, Gm], mu [Gm], gene_idx [Gm] (global), var [K] (eigenvalues)."""

    def __init__(self, loadings, mu, gene_idx, var, total_var) -> None:
        self.loadings = np.asarray(loadings, dtype=np.float32)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.gene_idx = np.asarray(gene_idx, dtype=np.int64)
        self.var = np.asarray(var, dtype=np.float32)
        self.total_var = float(total_var)

    @property
    def n_pcs(self) -> int:
        return self.loadings.shape[0]

    @property
    def pc_sd(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-12)).astype(np.float32)

    def project(self, Xm: np.ndarray) -> np.ndarray:
        return ((Xm - self.mu) @ self.loadings.T).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, loadings=self.loadings, mu=self.mu, gene_idx=self.gene_idx, var=self.var,
                 total_var=np.array(self.total_var))

    @classmethod
    def load(cls, path: Path) -> "Space":
        z = np.load(path)
        return cls(z["loadings"], z["mu"], z["gene_idx"], z["var"], float(z["total_var"]))


class TorchSpace:
    """The basis on a device, for projecting batches."""

    def __init__(self, space: Space, device) -> None:
        self.L = torch.from_numpy(space.loadings).to(device)
        self.mu = torch.from_numpy(space.mu).to(device)
        self.n_model = self.L.shape[1]

    def project(self, Xm: torch.Tensor) -> torch.Tensor:
        return (Xm - self.mu) @ self.L.T


def _read_model(c, rows: np.ndarray, mcol: np.ndarray, n_model: int) -> np.ndarray:
    return to_model_dense(c.csr(rows), mcol, n_model)


def fit_space(contexts, mcols: dict, model_idx: np.ndarray, n_pcs: int, fit_cells: int, seed: int,
              device: str) -> tuple[Space, dict]:
    from .hvg import sample_rows

    Gm = len(model_idx)
    dev = torch.device(device)
    ss = torch.zeros((Gm, Gm), dtype=torch.float64, device=dev)
    s = torch.zeros(Gm, dtype=torch.float64, device=dev)
    n_tot, per_ctx = 0, {}
    for c in contexts:
        t0 = time.time()
        rows = sample_rows(np.arange(len(c.pert)), fit_cells, seed)
        for i in range(0, len(rows), 4096):
            X = torch.from_numpy(_read_model(c, rows[i : i + 4096], mcols[c.name], Gm)).to(dev)
            ss += (X.T @ X).double()
            s += X.sum(0).double()
        n_tot += len(rows)
        per_ctx[c.name] = int(len(rows))
        print(f"    PCA fit: {c.name}: {len(rows):,} cells  [{time.time() - t0:.0f}s]", flush=True)
    mean = s / n_tot
    cov = (ss - n_tot * torch.outer(mean, mean)) / max(n_tot - 1, 1)
    cov = cov.cpu().numpy()
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:n_pcs]
    total = float(np.trace(cov))
    space = Space(V[:, order].T, mean.cpu().numpy(), model_idx, np.maximum(w[order], 0), total)
    info = {"n_fit_cells": per_ctx, "var_explained": float(space.var.sum() / max(total, 1e-12))}
    return space, info


# ---- fingerprints ---------------------------------------------------------------

def delta_table(c, space: Space, mcol: np.ndarray, min_cells: int) -> dict:
    """Mean log1p(CPM) of each perturbation minus the control mean, projected."""
    Gm = len(space.gene_idx)
    perts, inv = np.unique(c.pert, return_inverse=True)       # perts[0] == 0 (control)
    if perts[0] != 0:
        raise SystemExit(f"{c.name}: no control cells")
    sums = np.zeros((len(perts), Gm), dtype=np.float64)
    n = len(c.pert)
    for lo in range(0, n, BLOCK_ROWS):
        hi = min(lo + BLOCK_ROWS, n)
        X = to_model_dense_csr(c.csr(np.arange(lo, hi)), mcol, Gm)
        g = inv[lo:hi]
        onehot = sp.csr_matrix((np.ones(hi - lo), (g, np.arange(hi - lo))), shape=(len(perts), hi - lo))
        sums += (onehot @ X).toarray()
    counts = np.bincount(inv, minlength=len(perts)).astype(np.float64)
    means = sums / np.maximum(counts, 1)[:, None]
    keep = np.flatnonzero((counts >= min_cells) & (perts != 0))
    dx = means[keep] - means[0]
    dz = (dx @ space.loadings.T.astype(np.float64)).astype(np.float32)
    captured = (dz.astype(np.float64) ** 2).sum(1) / np.maximum((dx ** 2).sum(1), 1e-12)
    return {"perts": perts[keep].astype(np.int32), "n_cells": counts[keep].astype(np.int32),
            "fp": dz, "captured": captured.astype(np.float32)}


def to_model_dense_csr(X: sp.csr_matrix, mcol: np.ndarray, n_model: int) -> sp.csr_matrix:
    """Local CSR -> CSR over the model genes (column remap, no densify)."""
    X = X.tocsr()
    m = mcol[X.indices]
    keep = m >= 0
    row = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    ptr = np.r_[0, np.cumsum(np.bincount(row[keep], minlength=X.shape[0]))]
    return sp.csr_matrix((X.data[keep], m[keep], ptr), shape=(X.shape[0], n_model))


def cov_fingerprint(X_local: sp.csr_matrix, mcol: np.ndarray, target_cols: np.ndarray,
                    space: Space) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[n_t, K] covariance of each target column with every PC score over the rows of
    `X_local` (log1p(CPM), control cells), [n_t] bool: target varies at all, and [n_t] the
    target's variance (cov / var = the SLOPE, the anchored model's S)."""
    Xm = to_model_dense(X_local, mcol, len(space.gene_idx))
    Z = space.project(Xm).astype(np.float64)
    T = np.asarray(X_local[:, target_cols].todense(), dtype=np.float64)
    n = X_local.shape[0]
    Tc = T - T.mean(0)
    Zc = Z - Z.mean(0)
    cov = (Tc.T @ Zc) / max(n - 1, 1)
    ok = Tc.std(0) > 1e-8
    cov[~ok] = 0.0
    var = (Tc ** 2).sum(0) / max(n - 1, 1)
    return cov.astype(np.float32), ok, var.astype(np.float32)


def cov_table(c, space: Space, mcol: np.ndarray, gene_names: np.ndarray, pert_names: np.ndarray,
              cov_cells: int, seed: int) -> dict:
    from .hvg import sample_rows

    local = {str(gene_names[g]): j for j, g in enumerate(c.gene_idx)}
    perts = np.array([int(p) for p in np.unique(c.pert) if p != 0 and str(pert_names[p]) in local],
                     dtype=np.int32)
    if not len(perts):
        return {"perts": perts, "fp": np.zeros((0, space.n_pcs), np.float32), "var": np.zeros(0, np.float32)}
    tcols = np.array([local[str(pert_names[p])] for p in perts], dtype=np.int64)
    rows = sample_rows(c.control_rows, cov_cells, seed)
    cov, ok, var = cov_fingerprint(c.csr(rows), mcol, tcols, space)
    return {"perts": perts[ok], "fp": cov[ok], "var": var[ok], "n_control": int(len(rows))}


def build_target_var(out: Path, contexts, gene_names, pert_names, cov_cells: int, seed: int) -> None:
    """Backfill each target's control variance into cov/<ctx>.npz for artifacts built before
    the anchored model needed it (same control rows as the covariance)."""
    from .hvg import sample_rows

    for c in contexts:
        f = out / "cov" / f"{c.name}.npz"
        z = dict(np.load(f))
        if "var" in z:
            continue
        local = {str(gene_names[g]): j for j, g in enumerate(c.gene_idx)}
        tcols = np.array([local[str(pert_names[p])] for p in z["perts"]], dtype=np.int64)
        rows = sample_rows(c.control_rows, cov_cells, seed)
        T = np.asarray(c.csr(rows)[:, tcols].todense(), dtype=np.float64) if len(tcols) else np.zeros((len(rows), 0))
        z["var"] = T.var(0, ddof=1).astype(np.float32)
        np.savez(f, **z)
        print(f"  target variances {c.name}: {len(tcols):,}  (backfilled)", flush=True)


# ---- cell state: the mean of each control cell's nearest controls ------------------

def knn_mean(Z: np.ndarray, k: int, device: str = "cpu", chunk: int = 2048) -> np.ndarray:
    """[n, K]: for every row of Z, the mean of its k nearest rows of Z (itself included),
    euclidean in the PCA space. THE definition of the cell-state conditioning, used by
    `prepare` and by predict.py alike."""
    dev = torch.device(device)
    Zt = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)).to(dev)
    k = min(int(k), Zt.shape[0])
    sq = (Zt * Zt).sum(1)
    out = np.empty(Z.shape, dtype=np.float32)
    for i in range(0, Zt.shape[0], chunk):
        q = Zt[i : i + chunk]
        d = sq[i : i + chunk, None] + sq[None, :] - 2.0 * (q @ Zt.T)      # squared distances
        idx = d.topk(k, dim=1, largest=False).indices
        out[i : i + chunk] = Zt[idx].mean(1).cpu().numpy()
    return out


def project_rows(c, rows: np.ndarray, mcol: np.ndarray, space: "Space", chunk: int = 4096) -> np.ndarray:
    """[len(rows), K] PC scores of a context's rows (sorted), read in chunks."""
    out = np.empty((len(rows), space.n_pcs), dtype=np.float32)
    for i in range(0, len(rows), chunk):
        r = rows[i : i + chunk]
        out[i : i + len(r)] = space.project(to_model_dense(c.csr(r), mcol, len(space.gene_idx)))
    return out


def build_states(out: Path, contexts, k: int, device: str, force: bool = False) -> None:
    """state/<context>__k<k>.npz for every context: rows (its sorted control rows) and
    state [n_control, K] (float16). Skips tables that already exist unless `force`."""
    if k <= 0:
        print("  cell state: off (data.pca.state_knn: 0)")
        return
    space = Space.load(out / "pca.npz")
    sdir = out / "state"
    sdir.mkdir(exist_ok=True)
    for c in contexts:
        f = sdir / f"{c.name}__k{k}.npz"
        if f.exists() and not force:
            print(f"  cell state {c.name} (k={k}): up to date")
            continue
        t0 = time.time()
        rows = np.sort(c.control_rows.astype(np.int64))
        Z = project_rows(c, rows, model_cols(c.gene_idx, space.gene_idx, c.name), space)
        S = knn_mean(Z, k, device)
        np.savez(f, rows=rows, state=S.astype(np.float16), k=np.array(k))
        print(f"  cell state {c.name}: {len(rows):,} controls, mean of {min(k, len(rows))} nearest  "
              f"[{time.time() - t0:.0f}s]", flush=True)


# ---- building the artifact (prepare) ---------------------------------------------

def label_for(holdout: list[str], exclude: list[str], params: dict) -> str:
    """Folder name: what is held out / excluded, the number of PCs, and a short hash of the
    other settings -- so artifacts with different settings (e.g. 512 vs 1024 PCs) coexist."""
    import hashlib

    parts = []
    if holdout:
        parts.append("holdout_" + "+".join(sorted(holdout)))
    if exclude:
        parts.append("ex_" + "+".join(sorted(exclude)))
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return "__".join(parts or ["full"]) + f"__pcs{params['n_pcs']}_{h}"


def _fingerprint_file(c) -> dict:
    st = Path(c._path).stat()
    return {"n_cells": int(len(c.pert)), "file_size": int(st.st_size), "file_mtime": int(st.st_mtime)}


def build(cfg: dict, holdout=None, exclude=(), device: str = "auto", force: bool = False) -> Path:
    from .dataset import load_contexts

    pcfg = pca_cfg(cfg)
    root = Path(cfg["data"]["processed_dir"])
    seed = int(cfg.get("seed", 0))
    # `--holdout none` = hold nothing out: the artifact a final (`full`) model needs, with
    # every dataset in the basis and in the mean effects stored for prediction
    holdout = [] if holdout == ["none"] else (sorted(holdout) if holdout else pcfg["holdout"])
    exclude = sorted(exclude or [])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta, contexts = load_contexts(root, exclude=exclude)
    unknown = set(holdout) - {c.name for c in contexts}
    if unknown:
        raise SystemExit(f"data.pca.holdout / --holdout: no such (non-excluded) context {sorted(unknown)}")
    fit_ctx = [c for c in contexts if c.name not in holdout]
    out = root / DIR / label_for(holdout, exclude, {k: pcfg[k] for k in PARAM_KEYS})
    man_path = out / "manifest.json"
    if not force and man_path.exists():
        m = json.loads(man_path.read_text())
        if (m.get("params") == {k: pcfg[k] for k in PARAM_KEYS} and m.get("seed") == seed
                and m.get("holdout") == holdout and m.get("excluded") == exclude
                and m.get("contexts") == {c.name: _fingerprint_file(c) for c in contexts}):
            print(f"{out}: up to date (held out {holdout or 'nothing'})")
            build_target_var(out, contexts, np.array(_read_names(root / "gene_vocab.csv")),
                             np.array(_read_names(root / "pert_vocab.csv")), int(pcfg["cov_cells"]), seed)
            build_states(out, contexts, int(pcfg["state_knn"]), device, force)
            return out
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "state", ignore_errors=True)      # the basis is refit: old state tables are stale
    gene_names = np.array(_read_names(root / "gene_vocab.csv"))
    pert_names = np.array(_read_names(root / "pert_vocab.csv"))

    model_idx = shared_gene_idx(root, meta)
    mcols = {c.name: model_cols(c.gene_idx, model_idx, c.name) for c in contexts}
    print(f"{out}\n  model genes: {len(model_idx):,} measured in all {len(meta['contexts'])} contexts  |  "
          f"PCA fit on {len(fit_ctx)} context(s), held out {holdout or 'none'}, excluded {exclude or 'none'}  |  "
          f"{pcfg['n_pcs']} PCs, <= {pcfg['fit_cells']:,} cells per context, device {device}", flush=True)

    t0 = time.time()
    space, info = fit_space(fit_ctx, mcols, model_idx, int(pcfg["n_pcs"]), int(pcfg["fit_cells"]), seed, device)
    space.save(out / "pca.npz")
    np.savetxt(out / "model_genes.csv", gene_names[model_idx], fmt="%s", header="gene_name", comments="")
    print(f"  basis {space.loadings.shape}: {info['var_explained']:.1%} of the model genes' variance  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    (out / "delta").mkdir(exist_ok=True)
    (out / "cov").mkdir(exist_ok=True)
    stats = {}
    for c in contexts:
        t1 = time.time()
        st = {}
        if c.name not in holdout:
            d = delta_table(c, space, mcols[c.name], int(pcfg["min_cells"]))
            np.savez(out / "delta" / f"{c.name}.npz", **d)
            st["delta_perts"] = int(len(d["perts"]))
            st["delta_captured_median"] = float(np.median(d["captured"])) if len(d["captured"]) else None
        cv = cov_table(c, space, mcols[c.name], gene_names, pert_names, int(pcfg["cov_cells"]), seed)
        np.savez(out / "cov" / f"{c.name}.npz", perts=cv["perts"], fp=cv["fp"], var=cv["var"])
        st["cov_perts"] = int(len(cv["perts"]))
        st["n_perts"] = int(len(np.unique(c.pert)) - 1)
        stats[c.name] = st
        cap = st.get("delta_captured_median")
        print(f"  {c.name}{' (held out)' if c.name in holdout else ''}: "
              + (f"delta {st['delta_perts']:,} perts (median {cap:.0%} of each effect inside the PCs), "
                 if "delta_perts" in st else "no delta (held out), ")
              + f"cov {st['cov_perts']:,}/{st['n_perts']:,} perts  [{time.time() - t1:.0f}s]", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "holdout": holdout, "excluded": exclude,
        "fit_contexts": sorted(c.name for c in fit_ctx),
        "params": {k: pcfg[k] for k in PARAM_KEYS}, "seed": seed,
        "n_model_genes": int(len(model_idx)),
        "contexts": {c.name: _fingerprint_file(c) for c in contexts},
        **info, "fingerprints": stats,
    }, indent=2))
    build_states(out, contexts, int(pcfg["state_knn"]), device, force)
    print(f"\nwrote {out}  [{time.time() - t0:.0f}s]")
    return out


def _read_names(path: Path) -> list[str]:
    import pandas as pd
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


# ---- using the artifact (train) ---------------------------------------------------

def find(root: Path, val_names: list[str], cfg: dict, contexts, full: bool = False) -> Path:
    """The artifact whose held-out set covers every validation context and whose settings
    and source files match; the one holding out the fewest contexts wins. `full` (a final
    model): only an artifact that held NOTHING out, so every dataset is in the basis and
    in the mean effects the model stores."""
    pcfg = pca_cfg(cfg)
    want = {k: pcfg[k] for k in PARAM_KEYS}
    fps = {c.name: _fingerprint_file(c) for c in contexts}
    hits = []
    for man in sorted((Path(root) / DIR).glob("*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("params") != want or not set(val_names) <= set(m.get("holdout", [])):
            continue
        if full and m.get("holdout"):
            continue
        if any(m["contexts"].get(n) != fp for n, fp in fps.items()):
            continue
        hits.append((len(m["holdout"]), man.parent))
    if not hits:
        need = " ".join(sorted(val_names)) or "none"
        have = []
        for man in sorted((Path(root) / DIR).glob("*/manifest.json")):
            m = json.loads(man.read_text())
            have.append(f"    {man.parent.name}: held out {m.get('holdout') or 'nothing'}, {m.get('params')}")
        raise SystemExit(
            f"\nNO MATCHING PCA SPACE for this run: it needs one that holds out "
            f"{'NOTHING (a final model)' if full else (sorted(val_names) or 'nothing')} with the settings\n"
            f"    {want}\n"
            f"what exists in {Path(root) / DIR}:\n" + ("\n".join(have) or "    (none)") + "\n"
            f"-> run `make prepare HOLDOUT={need}` with this config first (stages 1-2 are reused; minutes).")
    return sorted(hits)[0][1]


class Artifact:
    def __init__(self, d: Path) -> None:
        self.dir = Path(d)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.space = Space.load(self.dir / "pca.npz")
        self.delta, self.cov = {}, {}
        for f in sorted((self.dir / "delta").glob("*.npz")):
            z = np.load(f)
            self.delta[f.stem] = dict(zip(z["perts"].tolist(), z["fp"]))
        self.cov_var = {}
        for f in sorted((self.dir / "cov").glob("*.npz")):
            z = np.load(f)
            self.cov[f.stem] = dict(zip(z["perts"].tolist(), z["fp"]))
            if "var" in z:
                self.cov_var[f.stem] = dict(zip(z["perts"].tolist(), z["var"].tolist()))

    def state(self, ctx_name: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(sorted control rows, state [n_control, K] float32) of one context."""
        f = self.dir / "state" / f"{ctx_name}__k{k}.npz"
        if not f.exists():
            raise SystemExit(f"{f} is missing: the cell-state table for data.pca.state_knn={k} was not "
                             f"built -- run `make prepare` with this config (only the state tables are computed)")
        z = np.load(f)
        return z["rows"], z["state"].astype(np.float32)


def scales(art: Artifact, train_names: list[str]) -> dict:
    """Per-element RMS of each fingerprint over the TRAINING contexts' tables."""
    out = {}
    for kind, tab in (("delta", art.delta), ("cov", art.cov)):
        vals = [v for n in train_names for v in tab.get(n, {}).values()]
        out[kind] = float(np.sqrt(np.mean(np.stack(vals) ** 2))) if vals else 1.0
    return out


def make_row(n_pcs: int, delta, cov, sc: dict) -> np.ndarray:
    row = np.zeros(fp_width(n_pcs), dtype=np.float32)
    if delta is not None:
        row[:n_pcs] = delta / sc["delta"]
        row[n_pcs] = 1.0
    if cov is not None:
        row[n_pcs + 1 : 2 * n_pcs + 1] = cov / sc["cov"]
        row[2 * n_pcs + 1] = 1.0
    row[2 * n_pcs + 2] = 1.0
    return row


def context_fingerprints(art: Artifact, ctx_name: str, ctx_perts, train_names: list[str], n_perts: int,
                         sc: dict, pcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lookup [n_perts] -> row, rows [R, D]) for one context. Row 0 = control (all zero).
    delta: mean over TRAINING contexts of OTHER cell lines; cov: this context's own."""
    K = art.space.n_pcs
    line = cell_line(ctx_name, pcfg)
    sources = [n for n in train_names if cell_line(n, pcfg) != line and n in art.delta]
    own_cov = art.cov.get(ctx_name, {})
    rows = [np.zeros(fp_width(K), dtype=np.float32)]
    lookup = np.zeros(n_perts, dtype=np.int32)
    for p in ctx_perts:
        p = int(p)
        if p == 0:
            continue
        ds = [art.delta[n][p] for n in sources if p in art.delta[n]]
        lookup[p] = len(rows)
        rows.append(make_row(K, np.mean(ds, axis=0) if ds else None, own_cov.get(p), sc))
    return lookup, np.stack(rows)


def generic_response(art: Artifact, sources: list[str]) -> np.ndarray:
    """[K] M: the average knockdown effect in PC units -- per source context the mean of its
    perturbations' mean effects, averaged over the source contexts."""
    per = [np.mean(np.stack(list(art.delta[n].values())), axis=0) for n in sources if art.delta.get(n)]
    return (np.mean(per, axis=0) if per else np.zeros(art.space.n_pcs)).astype(np.float32)


def context_anchor(art: Artifact, ctx_name: str, lookup: np.ndarray, n_rows: int, train_names: list[str],
                   pcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The anchored model's data inputs for one context, aligned with its fingerprint rows:
    S [n_rows, K] = slope of the PC scores on the knocked-out gene over this context's
    controls, cov / var(target) (0 if not measured), and M [K] = the generic response of the
    training contexts of OTHER cell lines (all training contexts for a held-out line)."""
    K = art.space.n_pcs
    if ctx_name in art.cov and ctx_name not in art.cov_var:
        raise SystemExit(f"{art.dir}: no target variances (built before the anchored model) -- run `make prepare`")
    S = np.zeros((n_rows, K), dtype=np.float32)
    cov, var = art.cov.get(ctx_name, {}), art.cov_var.get(ctx_name, {})
    for p in np.flatnonzero(lookup):
        if int(p) in cov and var.get(int(p), 0.0) > 1e-8:
            S[lookup[p]] = cov[int(p)] / var[int(p)]
    line = cell_line(ctx_name, pcfg)
    sources = [n for n in train_names if cell_line(n, pcfg) != line]
    return S, generic_response(art, sources)


def prediction_table(art: Artifact, train_names: list[str]) -> dict:
    """Per perturbation, the mean delta over EVERY training context (a new cell line has
    no own-line entry to leave out) -- saved beside the checkpoint for predict.py."""
    acc: dict[int, list] = {}
    for n in train_names:
        for p, v in art.delta.get(n, {}).items():
            acc.setdefault(p, []).append(v)
    perts = np.array(sorted(acc), dtype=np.int32)
    fp = np.stack([np.mean(acc[p], axis=0) for p in perts]) if len(perts) else np.zeros((0, art.space.n_pcs))
    return {"perts": perts, "fp": fp.astype(np.float32)}


def save_for_run(run_dir: Path, art: Artifact, train_names: list[str], sc: dict, pert_names) -> None:
    """Everything prediction needs travels with the checkpoint: basis, model genes, the
    delta table over the training contexts, the fingerprint scales."""
    shutil.copy(art.dir / "pca.npz", run_dir / "pca.npz")
    shutil.copy(art.dir / "model_genes.csv", run_dir / "model_genes.csv")
    tab = prediction_table(art, train_names)
    np.savez(run_dir / "fingerprints.npz", pert_names=np.array([str(pert_names[p]) for p in tab["perts"]]),
             fp=tab["fp"], scale_delta=np.array(sc["delta"]), scale_cov=np.array(sc["cov"]),
             generic=generic_response(art, train_names))
