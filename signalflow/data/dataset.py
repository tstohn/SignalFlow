"""Batching for conditional flow matching in the shared PCA space.

A training item is one cell x1 (perturbed, or a control with pert index 0 -- the
"no net displacement" anchor). Its partner x0 is a control cell of the same context,
picked by the per-dataset optimal-transport table (`data/couple.py`), or at random
where there is none. Batches carry the two cells' log1p(CPM) as sparse local rows
plus `model_col` (local column -> model gene); `densify_batch` keeps only the model
genes and projects both cells into the PCA space on the GPU (`data/pca_space.py`).

Each batch also carries its perturbations' fingerprint rows (`c.fp_rows`, attached by
`training/train.py`): [delta | delta_ok | cov | cov_ok | is_perturbed], once per
DISTINCT perturbation plus an inverse index -- and, when the cell-state conditioning is
on, each SOURCE cell's state (mean of its k nearest controls in the PCA space,
`c.state`, precomputed by `prepare`; data/pca_space.py).

WHICH CELLS GO WHERE IS NOT DECIDED HERE
    A `FlowDataset` is built from a list of contexts and uses all of their cells (up to
    `nodelta_cap`, below). `training/train.py` decides which whole cell lines train and
    which is held out.

CELLS WITHOUT FINGERPRINTS
    A perturbed cell with NEITHER fingerprint (no other cell line measured the
    knockdown, and this line does not measure the knocked-out gene) carries nothing
    about which perturbation it is, so it is dropped: there is nothing to learn from
    it. A perturbed cell with only the covariance fingerprint is kept, but
    `nodelta_cap` limits those to that share of the cells each epoch (a different
    random subset every epoch), so they cannot crowd out the cells that have both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

# Total bytes of context matrices `load_contexts` will hold fully in RAM before the rest
# fall back to disk-backed access. A GLOBAL budget, not a per-context one. It covers only
# these matrices; the control-row caches come on top. Re-tune per machine.
EAGER_BUDGET_BYTES = 8 * 2**30


class ContextStore:
    """One context's arrays. Small per-cell metadata is read fully into RAM; the expression
    matrix is in RAM (`load_contexts` decides, against a shared budget) or disk-backed.

    Attached by `training/train.py`: `match` (OT table), `model_col` (local column ->
    model gene, -1 elsewhere), `fp_lookup`/`fp_rows` (fingerprint row per perturbation),
    and optionally `state_rows`/`state` (cell state of every control row, sorted rows).
    """

    def __init__(self, path: Path, index: int, name: str) -> None:
        self._path, self._pid = Path(path), os.getpid()
        self._layouts: dict[str, tuple | None] = {}
        self._fd: int | None = None
        self._fd_pid = -1
        try:
            self._h5 = h5py.File(path, "r")
        except OSError as e:
            raise SystemExit(
                f"{path}: not a readable HDF5 file ({e}). Re-run "
                f"`python -m signalflow.data.prepare --config <config>`."
            ) from e
        z = self._h5
        if "scalars" not in z:
            raise SystemExit(f"{path}: written by an older `prepare` -- re-run it.")
        self.index, self.name = index, name
        self.gene_idx = z["gene_idx"][:]
        self.pert = z["pert"][:]
        self.is_control = z["is_control"][:]
        self.lib = z["lib"][:]
        self.scalars = z["scalars"][:]
        self.control_rows = z["control_rows"][:]
        self.gene_effect = z["gene_effect"][:] if "gene_effect" in z else None
        self.match: np.ndarray | None = None
        self.model_col: np.ndarray | None = None
        self.fp_lookup: np.ndarray | None = None
        self.fp_rows: np.ndarray | None = None
        self.state_rows: np.ndarray | None = None
        self.state: np.ndarray | None = None
        self.anchor_s: np.ndarray | None = None     # [fp rows, K] slope rows (anchored model)
        self.anchor_m: np.ndarray | None = None     # [K] generic response (anchored model)
        # centered flow (model.centered): [fp rows, K] each knockdown's REAL average shift in THIS
        # context (row 0 = controls = 0) and [fp rows] bool: known (>= data.pca.min_cells cells)
        self.center_rows: np.ndarray | None = None
        self.center_ok: np.ndarray | None = None
        self._sources: np.ndarray | None = None

        self._n_local = int(z["X_shape"][1])
        self.n_bytes = int(z["X_data"].shape[0]) * 4 + int(z["X_indices"].shape[0]) * 4

        self.X: sp.csr_matrix | None = None
        self._indptr: np.ndarray | None = None
        self._control_X: sp.csr_matrix | None = None
        self._cached_rows: np.ndarray | None = None

    def _finalize(self, eager: bool) -> None:
        z = self._h5
        if eager:
            self.X = sp.csr_matrix(
                (z["X_data"][:], z["X_indices"][:], z["X_indptr"][:]), shape=tuple(z["X_shape"][:])
            )
            self._h5.close()
            self._h5 = None
        else:
            self._indptr = z["X_indptr"][:]

    def cache_controls(self, max_cells: int | None = None, seed: int = 0) -> None:
        """Keep this (disk-backed) context's control rows in RAM -- all, or a fixed seeded
        subset of `max_cells`. No-op if `.X` is in memory."""
        if self.X is not None:
            return
        rows = np.sort(self.control_rows.astype(np.int64))
        if max_cells and len(rows) > max_cells:
            rows = np.sort(np.random.default_rng(seed).choice(rows, max_cells, replace=False))
        if self._cached_rows is not None and len(self._cached_rows) >= len(rows):
            return
        self._control_X = self._gather_sorted(rows)
        self._cached_rows = rows

    def set_sources(self, rows: np.ndarray) -> None:
        """Fix the control rows x0 may come from to the pool the coupling was solved
        against, and keep them in RAM if disk-backed."""
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        if not np.isin(rows, self.control_rows).all():
            raise SystemExit(f"{self.name}: the coupling's source rows are not all control cells "
                             f"-- the processed data changed since `prepare`; re-run it")
        self._sources = rows
        if self.X is None:
            self._control_X = self._gather_sorted(rows)
            self._cached_rows = rows

    def source_rows(self) -> np.ndarray:
        if self._sources is not None:
            return self._sources
        if self._cached_rows is not None:
            return self._cached_rows
        return self.control_rows.astype(np.int64)

    def _own_handle(self) -> "h5py.File":
        """After a DataLoader fork an inherited handle shares its file offset: reopen once."""
        if self._pid != os.getpid():
            self._h5 = h5py.File(self._path, "r")
            self._pid = os.getpid()
        return self._h5

    def _layout(self, name: str):
        """(chunk_elems, byte offset of every chunk, dtype) of an uncompressed chunked
        dataset, or None to fall back to h5py (which reads whole chunks per row)."""
        if name not in self._layouts:
            lay = None
            ds = self._own_handle()[name]
            try:
                if ds.compression is None and ds.chunks is not None and not ds.id.get_create_plist().get_nfilters():
                    C = int(ds.chunks[0])
                    n = ds.id.get_num_chunks()
                    if n == -(-int(ds.shape[0]) // C):
                        offs = np.empty(n, dtype=np.int64)
                        for i in range(n):
                            info = ds.id.get_chunk_info(i)
                            offs[info.chunk_offset[0] // C] = info.byte_offset
                        lay = (C, offs, ds.dtype)
            except Exception:
                lay = None
            self._layouts[name] = lay
        return self._layouts[name]

    def _read(self, name: str, lo: int, hi: int) -> np.ndarray:
        """Elements [lo, hi) of dataset `name`, by pread when possible (fork-safe)."""
        lay = self._layout(name)
        if lay is None:
            return self._own_handle()[name][lo:hi]
        C, offs, dt = lay
        if self._fd is None or self._fd_pid != os.getpid():
            self._fd, self._fd_pid = os.open(self._path, os.O_RDONLY), os.getpid()
        item = dt.itemsize
        out = np.empty(hi - lo, dtype=dt)
        pos = lo
        while pos < hi:
            ch = pos // C
            end = min(hi, (ch + 1) * C)
            buf = os.pread(self._fd, (end - pos) * item, int(offs[ch]) + (pos - ch * C) * item)
            out[pos - lo : end - lo] = np.frombuffer(buf, dtype=dt)
            pos = end
        return out

    def _gather_sorted(self, rows: np.ndarray) -> sp.csr_matrix:
        """`rows` sorted ascending, unique. One read per contiguous run of rows."""
        indptr = self._indptr
        datas, indices, new_indptr = [], [], [0]
        i, n = 0, len(rows)
        while i < n:
            j = i
            while j + 1 < n and rows[j + 1] == rows[j] + 1:
                j += 1
            r0, r1 = int(rows[i]), int(rows[j])
            lo, hi = int(indptr[r0]), int(indptr[r1 + 1])
            datas.append(self._read("X_data", lo, hi))
            indices.append(self._read("X_indices", lo, hi))
            new_indptr.extend((indptr[r0 : r1 + 1 + 1] - lo + new_indptr[-1])[1:].tolist())
            i = j + 1
        data = np.concatenate(datas) if datas else np.zeros(0, np.float32)
        idx = np.concatenate(indices) if indices else np.zeros(0, np.int32)
        return sp.csr_matrix((data, idx, np.array(new_indptr, dtype=np.int64)), shape=(n, self._n_local))

    def csr(self, rows: np.ndarray) -> sp.csr_matrix:
        """Sparse local-gene rows, in the order (and with the repeats) of `rows`."""
        rows = np.asarray(rows, dtype=np.int64)
        if self.X is not None:
            return self.X[rows]
        if self._cached_rows is not None and len(rows):
            local = np.minimum(np.searchsorted(self._cached_rows, rows), len(self._cached_rows) - 1)
            if (self._cached_rows[local] == rows).all():
                return self._control_X[local]
        uniq, inv = np.unique(rows, return_inverse=True)
        m = self._gather_sorted(uniq)
        return m if len(uniq) == len(rows) and (uniq == rows).all() else m[inv]

    def dense(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.csr(rows).todense(), dtype=np.float32)

    def state_of(self, rows: np.ndarray) -> np.ndarray:
        """[len(rows), K] cell state of control `rows` (any order, repeats allowed)."""
        at = np.searchsorted(self.state_rows, rows)
        if not (self.state_rows[np.minimum(at, len(self.state_rows) - 1)] == rows).all():
            raise ValueError(f"{self.name}: cell state asked for a non-control row")
        return self.state[at]


def load_contexts(processed_dir: str | Path, exclude=()) -> tuple[dict, list[ContextStore]]:
    """meta.json and every non-excluded context. Eager (in RAM) vs disk-backed is decided
    per context against one shared `EAGER_BUDGET_BYTES`, smallest first."""
    d = Path(processed_dir)
    meta = json.loads((d / "meta.json").read_text())
    exclude = set(exclude or ())
    unknown = exclude - {c["name"] for c in meta["contexts"]}
    if unknown:
        raise SystemExit(f"exclude: no such context {sorted(unknown)}; available:\n  "
                         + "\n  ".join(c["name"] for c in meta["contexts"]))
    stores = [ContextStore(d / c["file"], c["index"], c["name"]) for c in meta["contexts"]
              if c["name"] not in exclude]
    budget = EAGER_BUDGET_BYTES
    for store in sorted(stores, key=lambda s: s.n_bytes):
        eager = store.n_bytes <= budget
        if eager:
            budget -= store.n_bytes
        store._finalize(eager)
    return meta, stores


class FlowDataset(Dataset):
    def __init__(self, contexts: list[ContextStore], seed: int = 0, keep_perts: np.ndarray | None = None,
                 nodelta_cap: float | None = None, pert_cap: int | None = None) -> None:
        """`keep_perts` restricts the TARGET cells to those perturbations (controls are always
        kept). Perturbed cells with neither fingerprint are always dropped. `nodelta_cap`
        (training only): cells without a delta fingerprint make up at most this share of
        each epoch's cells. `pert_cap` (training only): at most this many cells per (context,
        perturbation) each epoch, a fresh random draw every epoch, so every perturbation is
        seen every epoch; a context's control targets shrink in the same proportion as its
        perturbed cells (its control share stays what it was)."""
        if not contexts:
            raise ValueError("a FlowDataset needs at least one context")
        for c in contexts:
            if c.model_col is None or c.fp_rows is None:
                raise ValueError(f"{c.name}: no model_col / fingerprints attached (see training/train.py)")
        has_state = [c.state is not None for c in contexts]
        if any(has_state) and not all(has_state):
            raise ValueError("some contexts have a cell-state table attached and some do not")
        self.use_state = all(has_state)
        has_anchor = [c.anchor_s is not None for c in contexts]
        if any(has_anchor) and not all(has_anchor):
            raise ValueError("some contexts have anchor inputs attached and some do not")
        self.use_anchor = all(has_anchor)
        has_center = [c.center_rows is not None for c in contexts]
        if any(has_center) and not all(has_center):
            raise ValueError("some contexts have centering targets attached and some do not")
        self.use_center = all(has_center)
        self.contexts = contexts
        self.n_contexts = len(contexts)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        keep = None if keep_perts is None else np.union1d(np.asarray(keep_perts, dtype=np.int64), [0])
        self.rows: list[np.ndarray] = []
        self.nodelta: list[np.ndarray] = []      # per context: bool over self.rows[i]
        self.control_pool: list[np.ndarray] = []
        self.src_pool: list[np.ndarray] = []
        K = (contexts[0].fp_rows.shape[1] - 3) // 2
        self.n_dropped: dict[str, int] = {}
        for c in contexts:
            sel = np.arange(len(c.pert), dtype=np.int64)
            if keep is not None:
                sel = sel[np.isin(c.pert, keep)]
            row = c.fp_lookup[c.pert[sel]]
            has_delta, has_cov = c.fp_rows[row, K] > 0, c.fp_rows[row, 2 * K + 1] > 0
            empty = (c.pert[sel] != 0) & ~has_delta & ~has_cov
            if self.use_center:
                # centered flow: a knockdown whose own average is unknown (too few cells) has no
                # target to center on, so its cells are dropped too
                empty |= (c.pert[sel] != 0) & ~c.center_ok[row]
            self.n_dropped[c.name] = int(empty.sum())
            sel, has_delta = sel[~empty], has_delta[~empty]
            self.rows.append(sel)
            self.nodelta.append((c.pert[sel] != 0) & ~has_delta)
            self.control_pool.append(c.control_rows.astype(np.int64))
            self.src_pool.append(c.source_rows())

        # per context: the (expected) cells an epoch draws before the nodelta subsample --
        # nodelta cells, the rest, and (with pert_cap) how many control targets to keep
        self.pert_cap = int(pert_cap) if pert_cap else 0
        self.n_ctrl_keep: list[int] = []
        nd_c, rest_c = [], []
        for c, sel, nd in zip(contexts, self.rows, self.nodelta):
            p = c.pert[sel]
            if not self.pert_cap:
                nd_c.append(int(nd.sum()))
                rest_c.append(len(sel) - nd_c[-1])
                continue
            is_pert = p != 0
            u, inv, cnt = np.unique(p[is_pert], return_inverse=True, return_counts=True)
            capped = np.minimum(cnt, self.pert_cap)
            nd_g = np.bincount(inv, weights=nd[is_pert], minlength=len(u)) > 0   # nodelta is per perturbation
            n_ctrl = int((~is_pert).sum())
            keep_ctrl = int(round(n_ctrl * capped.sum() / max(int(cnt.sum()), 1)))
            self.n_ctrl_keep.append(keep_ctrl)
            nd_c.append(int(capped[nd_g].sum()))
            rest_c.append(int(capped[~nd_g].sum()) + keep_ctrl)
        n_nd, n_rest = sum(nd_c), sum(rest_c)
        self.keep_frac = 1.0
        if nodelta_cap is not None and n_nd:
            self.keep_frac = min(1.0, float(nodelta_cap) * n_rest / max((1.0 - float(nodelta_cap)) * n_nd, 1))
        self.n_nodelta, self.n_rest = n_nd, n_rest
        # expected cells per context per epoch (context_balance weights come from this)
        self.epoch_counts = [r + self.keep_frac * d for r, d in zip(rest_c, nd_c)]
        self._epoch_cache: tuple[int, list[np.ndarray]] | None = None

    def _capped(self, i: int, rng) -> np.ndarray:
        """bool over self.rows[i]: a random `pert_cap` cells of each perturbation and the
        context's `n_ctrl_keep[i]` control targets."""
        p = self.contexts[i].pert[self.rows[i]]
        order = np.lexsort((rng.random(len(p)), p))            # by perturbation, random within
        ps = p[order]
        start = np.searchsorted(ps, ps, side="left")
        rank = np.arange(len(ps)) - start
        lim = np.where(ps == 0, self.n_ctrl_keep[i], self.pert_cap)
        keep = np.zeros(len(p), dtype=bool)
        keep[order[rank < lim]] = True
        return keep

    def epoch_rows(self, epoch: int) -> list[np.ndarray]:
        """Each context's rows for this epoch (sorted): with `pert_cap`, a fresh random draw
        of at most that many cells per perturbation; then all cells with a delta
        fingerprint, plus a fresh random `keep_frac` of those without."""
        if self.keep_frac >= 1.0 and not self.pert_cap:
            return self.rows
        if self._epoch_cache is not None and self._epoch_cache[0] == epoch:
            return self._epoch_cache[1]
        rng = np.random.default_rng([self.seed, 7919, epoch])
        out = []
        for i, (r, nd) in enumerate(zip(self.rows, self.nodelta)):
            keep = self._capped(i, rng) if self.pert_cap else np.ones(len(r), dtype=bool)
            if self.keep_frac < 1.0:
                idx = np.flatnonzero(nd)
                keep[idx[rng.random(len(idx)) >= self.keep_frac]] = False
            out.append(r[keep])
        self._epoch_cache = (epoch, out)
        return out

    def __len__(self) -> int:
        return int(self.n_rest + round(self.keep_frac * self.n_nodelta))

    def _match_src(self, c_id: int, tgt_rows: np.ndarray, rng) -> np.ndarray:
        """One source control per target: the per-dataset OT table where it has one,
        otherwise (controls, or no table) a random control of the pool.

        Using x1 to CHOOSE x0 is not leakage: it only picks which pairs the loss sees."""
        pool = self.src_pool[c_id]
        c = self.contexts[c_id]
        if c.match is None:
            return rng.choice(pool, size=len(tgt_rows))
        src = c.match[tgt_rows].astype(np.int64)
        gap = src < 0
        if gap.any():
            src[gap] = rng.choice(pool, size=int(gap.sum()))
        return src

    def collate(self, items, rng=None, fetch_csr=None) -> dict[str, torch.Tensor]:
        """A single-context batch as SPARSE pieces; `densify_batch` expands it on the GPU."""
        rng = self.rng if rng is None else rng
        ctx = {c for c, _ in items}
        if len(ctx) != 1:
            raise ValueError("a batch is single-context (BlockStream batches are)")
        c_id = int(next(iter(ctx)))
        c = self.contexts[c_id]
        rows = np.array([r for _, r in items], dtype=np.int64)
        src = self._match_src(c_id, rows, rng)
        x1 = (fetch_csr(c_id, rows) if fetch_csr is not None else c.csr(rows)).tocsr()
        x0 = c.csr(src).tocsr()
        pert = c.pert[rows].astype(np.int64)
        uniq, inv = np.unique(c.fp_lookup[pert], return_inverse=True)

        t = torch.from_numpy
        out = {
            "x1_ptr": t(x1.indptr.astype(np.int64)), "x1_col": t(x1.indices.astype(np.int32)),
            "x1_val": t(x1.data.astype(np.float32)),
            "x0_ptr": t(x0.indptr.astype(np.int64)), "x0_col": t(x0.indices.astype(np.int32)),
            "x0_val": t(x0.data.astype(np.float32)),
            "model_col": t(c.model_col.astype(np.int64)),
            "fp_rows": t(c.fp_rows[uniq].astype(np.float32)), "fp_inv": t(inv.astype(np.int64)),
            "pert": t(pert),
            "ctx": t(np.full(len(rows), c_id, dtype=np.int64)),
        }
        if self.use_state:
            # the SOURCE cell's state: at inference only the control cell exists
            out["state"] = t(c.state_of(src).astype(np.float32))
        if self.use_anchor:
            out["anchor_s"] = t(c.anchor_s[uniq].astype(np.float32))      # rows aligned with fp_rows
            out["anchor_m"] = t(c.anchor_m.astype(np.float32))
        if self.use_center:
            out["center_rows"] = t(c.center_rows[uniq].astype(np.float32))   # rows aligned with fp_rows
        return out


def densify_batch(batch: dict[str, torch.Tensor], device, space) -> dict[str, torch.Tensor]:
    """Move a batch to `device`, keep the model genes, project both cells into the PCA
    space (`space`: pca_space.TorchSpace). Returns z0, z1 [B, K], fp [B, D], pert, ctx,
    and state [B, K] if the batch has one."""
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    B = b["pert"].shape[0]
    mcol = b["model_col"]

    def z(name):
        ptr, col, val = b[f"{name}_ptr"], b[f"{name}_col"].long(), b[f"{name}_val"]
        rows = torch.repeat_interleave(torch.arange(B, device=ptr.device), ptr[1:] - ptr[:-1])
        m = mcol[col]
        keep = m >= 0
        x = torch.zeros(B, space.n_model, device=ptr.device)
        x[rows[keep], m[keep]] = val[keep]
        return space.project(x)

    out = {"z0": z("x0"), "z1": z("x1"), "fp": b["fp_rows"][b["fp_inv"]],
           "pert": b["pert"], "ctx": b["ctx"]}
    if "state" in b:
        out["state"] = b["state"]
    if "anchor_s" in b:
        # [B, 2K]: the anchored model's S (slope on the knocked-out gene) | M (generic response)
        out["anchor"] = torch.cat([b["anchor_s"][b["fp_inv"]], b["anchor_m"][None, :].expand(B, -1)], dim=-1)
    if "center_rows" in b:
        # [B, K]: the centered flow's target shift -- the knockdown's real average in this context
        out["center"] = b["center_rows"][b["fp_inv"]]
    return out


class BlockStream(IterableDataset):
    """Batches assembled from BLOCKS of rows read off the training thread.

    One epoch = a plan of tasks. A task is `blocks_per_task` blocks (each `block_cells`
    consecutive entries of one context's epoch rows, drawn at random), each read as ONE
    contiguous span, shuffled together, cut into single-context batches. Everything is a
    function of (seed, epoch, task), whatever the number of workers. With `shuffle=False`
    (validation) the plan is the same every epoch.
    """

    def __init__(self, ds: FlowDataset, batch_size: int, seed: int = 0, shuffle: bool = True,
                 block_cells: int = 1024, blocks_per_task: int = 4) -> None:
        self.ds, self.bs, self.seed, self.shuffle = ds, int(batch_size), int(seed), shuffle
        self.block_cells, self.blocks_per_task = int(block_cells), int(blocks_per_task)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Workers are forked fresh per epoch from THIS copy, so the parent must say which epoch."""
        self._epoch = int(epoch)

    def _plan(self, epoch: int) -> list[tuple[int, list[np.ndarray]]]:
        e = epoch if self.shuffle else 0
        rng = np.random.default_rng([self.seed, e])
        tasks = []
        for c_id, rows in enumerate(self.ds.epoch_rows(e) if self.shuffle else self.ds.rows):
            if not len(rows):
                continue
            blocks = [rows[i : i + self.block_cells] for i in range(0, len(rows), self.block_cells)]
            order = rng.permutation(len(blocks)) if self.shuffle else np.arange(len(blocks))
            for i in range(0, len(order), self.blocks_per_task):
                tasks.append((c_id, [blocks[j] for j in order[i : i + self.blocks_per_task]]))
        if self.shuffle:
            tasks = [tasks[i] for i in rng.permutation(len(tasks))]
        return tasks

    def _n_batches(self, n_cells: int) -> int:
        return n_cells // self.bs if self.shuffle else -(-n_cells // self.bs)

    def __len__(self) -> int:
        return sum(self._n_batches(sum(len(b) for b in blocks)) for _, blocks in self._plan(self._epoch))

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        epoch, self._epoch = self._epoch, self._epoch + 1
        ds = self.ds
        for t, (c_id, blocks) in enumerate(self._plan(epoch)):
            if t % nw != wid:
                continue
            c = ds.contexts[c_id]
            blocks = sorted(blocks, key=lambda b: int(b[0]))
            rows = np.concatenate(blocks)
            # each block is read as one contiguous span (a capped epoch leaves gaps in it),
            # then the rows it actually uses are taken out of that span
            sub = sp.vstack([c.csr(np.arange(b[0], b[-1] + 1))[b - b[0]] for b in blocks], format="csr")
            rng = np.random.default_rng([self.seed, epoch if self.shuffle else 0, t])
            perm = rng.permutation(len(rows)) if self.shuffle else np.arange(len(rows))

            def fetch_csr(_cid, tgt, rows=rows, sub=sub):
                return sub[np.searchsorted(rows, tgt)]

            for i in range(0, len(perm), self.bs):
                idx = perm[i : i + self.bs]
                if self.shuffle and len(idx) < self.bs:
                    break
                yield ds.collate([(c_id, int(r)) for r in rows[idx]], rng=rng, fetch_csr=fetch_csr)
            del sub


def make_loader(dataset: FlowDataset, batch_size: int, seed: int = 0, shuffle: bool = True,
                num_workers: int = 0, block_cells: int = 1024, blocks_per_task: int = 4,
                prefetch_factor: int = 2, pin_memory: bool = False, persistent_workers: bool = False):
    from torch.utils.data import DataLoader

    stream = BlockStream(dataset, batch_size, seed, shuffle, block_cells, blocks_per_task)
    kw = {}
    if num_workers > 0:
        kw = dict(num_workers=num_workers, prefetch_factor=prefetch_factor,
                  persistent_workers=persistent_workers)
    return DataLoader(stream, batch_size=None, pin_memory=pin_memory, **kw)
