"""Stream predicted counts into an .h5ad on disk, one block at a time.

Split out of `predict.py` because it is the one part with its own contract: it
holds a single block in memory however large the file becomes, checks the value
rules on the very arrays it writes, and only moves the file into place after
re-reading it from disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
from anndata.io import sparse_dataset, write_elem

from ..submission.file_rules import check_genes, check_layout


class CountsWriter:
    """Append predicted blocks to an .h5ad on disk, checking each as it goes.

    Holds one block in memory however large the file becomes. The value rules
    (6, 8, 9) are checked on the very arrays being written; `finish()` then
    re-reads what landed on disk, and only then moves the file into place.

    `indptr` is 64-bit from the first block. anndata refuses to append once a
    32-bit `indptr` would overflow (2.147e9 stored entries), and a full VCC26
    panel is close enough to that for a slightly denser model to hit it -- a
    failure that would otherwise arrive partway through a long run.
    """

    def __init__(self, path, obs, var, uns, n_genes, cap, max_stored):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".partial")
        self.n_rows, self.n_genes = len(obs), n_genes
        self.cap, self.max_stored = cap, max_stored
        self.rows = self.nnz = 0
        self.worst_total = 0.0
        self._ds = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # obs / var / uns first, with no X; X is appended below
        ad.AnnData(obs=obs, var=var, uns=uns).write_h5ad(self.tmp)
        self._f = h5py.File(self.tmp, "a")

    def _fail(self, msg: str) -> None:
        self._f.close()
        self.tmp.unlink(missing_ok=True)
        raise SystemExit(msg)

    @staticmethod
    def _value_problem(block: sp.csr_matrix, cap: float) -> str | None:
        d = block.data
        if d.size:
            if not np.isfinite(d).all():
                return "rule 6: counts contain non-finite values"
            if d.min() < 0:
                return "rule 6: counts contain negative values"
            if not np.array_equal(d, np.rint(d)):
                return "rule 6: counts are fractional; they must be whole numbers"
            if (d == 0).any():
                return "rule 8: explicitly-stored zeros; call eliminate_zeros()"
        worst = float(np.asarray(block.sum(axis=1)).max()) if block.shape[0] else 0.0
        if worst > cap:
            return f"rule 9: a cell totals {worst:,.0f} counts, above {cap:,.0f}"
        return None

    def append(self, block: sp.csr_matrix) -> None:
        problem = self._value_problem(block, self.cap)
        if problem:
            self._fail(problem)
        if self.nnz + block.nnz > self.max_stored:
            self._fail(
                f"rule 8: {self.nnz + block.nnz:,} stored entries so far exceeds "
                f"{self.max_stored:,}; the model predicts too many expressed genes per cell"
            )
        if block.shape[1] != self.n_genes or self.rows + block.shape[0] > self.n_rows:
            self._fail("internal error: block does not fit the planned output")

        if self._ds is None:
            first = block.copy()
            first.indptr = first.indptr.astype(np.int64)
            write_elem(
                self._f, "X", first,
                dataset_kwargs={"compression": "gzip", "compression_opts": 4},
            )
            self._ds = sparse_dataset(self._f["X"])
        else:
            self._ds.append(block)

        self.rows += block.shape[0]
        self.nnz += block.nnz
        if block.shape[0]:
            self.worst_total = max(self.worst_total, float(np.asarray(block.sum(axis=1)).max()))

    def finish(self, genes, pert_col, ctx_col, expect_labels) -> None:
        """Re-check what is on disk, then move it into place."""
        if self.rows != self.n_rows:
            self._fail(f"wrote {self.rows:,} of the {self.n_rows:,} planned cells")
        self._f.close()

        with h5py.File(self.tmp, "r") as f:
            if f["X/indptr"].dtype != np.int64 or int(f["X/indptr"][-1]) != self.nnz:
                self.tmp.unlink(missing_ok=True)
                raise SystemExit("what landed on disk disagrees with what was written (indptr)")
            data_size = f["X/data"].shape[0]
        if data_size != self.nnz:
            self.tmp.unlink(missing_ok=True)
            raise SystemExit("what landed on disk disagrees with what was written (data)")

        b = ad.read_h5ad(self.tmp, backed="r")
        try:
            if b.shape != (self.n_rows, self.n_genes):
                raise SystemExit(f"on disk the file is {b.shape}, expected {(self.n_rows, self.n_genes)}")
            check_genes(b.var_names, genes)
            check_layout(b.obs, **expect_labels)
            # re-verify the values as stored, on the first, a middle and the last block
            span = min(400, self.n_rows)
            for start in sorted({0, (self.n_rows // 2) // span * span, self.n_rows - span}):
                blk = sp.csr_matrix(b.X[start : start + span])
                problem = self._value_problem(blk, self.cap)
                if problem:
                    raise SystemExit(f"on disk, rows {start}-{start + span}: {problem}")
        except SystemExit:
            b.file.close()
            self.tmp.unlink(missing_ok=True)
            raise
        b.file.close()

        os.replace(self.tmp, self.path)
        print(
            f"  verified on disk: {self.n_rows:,} cells x {self.n_genes:,} genes, "
            f"{self.nnz:,} stored entries ({self.nnz / self.max_stored:.1%} of cap, "
            f"{self.nnz / max(self.n_rows, 1):,.0f} per cell), max cell total {self.worst_total:,.0f}"
        )
