"""VCC2026 scoring, via the official `cell-eval2` implementation.

Called from `training/train.py` after every `train.cell_eval_every` epochs, on the
held-out cell line.

Nothing here re-implements a metric. `cell_eval2` owns the arithmetic; this
module's whole job is to hand it two AnnData objects in the shape the
competition defines, and to do that honestly.

THE SIX SCORED MEMBERS (profile `vcc2026`)
    pds_cosine                                separability of predicted profiles
    expr_mse_unbiased_capped_norm             size of the expression error
    de_wilcoxon_direction_fidelity_yield_raw  correctness of predicted directions
    de_wilcoxon_direction_reach_raw           depth over which directions hold
    de_wilcoxon_sig_jaccard                   responding-gene set agreement
    de_wilcoxon_lfc_nmae                      fold-change accuracy

COUNTS, NOT LOGNORM
    The competition scores raw counts, and `cell_eval2` says so: the group-sum
    profile behind `pds_cosine` and `expr_mse_*` needs P_g = sum_c y_cg, which
    lognorm input cannot supply -- declaring `input_type="lognorm"` silently
    swaps in a per-cell fallback profile and stops being VCC2026 arithmetic.

    So both sides are converted back to counts. That inversion is exact here,
    not an approximation: `prepare.py` writes CPM+log1p (expm1 of a stored row
    sums to 1e6) alongside `lib`, the cell's own UMI total, so
    `rint(expm1(x) * lib / 1e6)` returns the original integers to within 8e-4.
    Predicted cells are built by `models.flow.residual_counts` -- the same function
    `predict.py` uses: the source control cell plus the back-projected predicted
    PCA shift on the model genes, its own raw counts on every other gene, at its
    own library size (a perturbation that shifts sequencing depth is not modelled).
    The baselines go through the same function: identity = no shift, mean_shift =
    the cross-line mean shift on the model genes only, so all three are compared
    inside the same gene space.

THE FULL 18,533-GENE READOUT SPACE, NOT THE CONTEXT'S PANEL
    Five of the six metrics exclude the perturbation's own target gene, and
    `cell_eval2` refuses to score when NO target resolves to a feature -- that
    being the construct-ID-vs-symbol mismatch that would otherwise exclude
    nothing and return a plausible wrong number. Four of the eight prototype
    contexts have zero target genes inside their own measured panel, so
    scoring on the panel would fail outright on half the data.

    Emitting the full readout space fixes that (pert vocab and gene vocab are
    the same csv, so every target is a feature) and costs nothing: the
    unmeasured genes are structurally zero on BOTH sides, so they drop out of
    the low-expression DE gate, contribute nothing to either squared distance,
    and leave cosine distance unchanged.

WHAT THE SCALED SCORE IS HERE
    Official scoring is s = (u - b) / (r - b) against reference bundles
    measured on the competition's own contexts; those constants do not
    describe this data and are not distributed for it. This uses the scale
    `cell_eval2` ships instead -- 0 = no skill, 1 = perfect -- so the six
    members land on one comparable axis. Raw values are reported alongside and
    are the thing to trust.
"""

from __future__ import annotations

import gc
import logging
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ..data.pca_space import local_model_order
from ..data.vocab import CONTROL_LABEL
from ..models.flow import integrate, residual_counts

SCALE = "low-random_high-1_v10"

# (output name, short column label) for the six scored members, in the order
# the brief lists them.
MEMBERS = (
    ("pds_cosine", "pds_cosine"),
    ("expr_mse_unbiased_capped_norm", "expr_mse"),
    ("de_wilcoxon_direction_fidelity_yield_raw", "dir_fidelity"),
    ("de_wilcoxon_direction_reach_raw", "dir_reach"),
    ("de_wilcoxon_sig_jaccard", "sig_jaccard"),
    ("de_wilcoxon_lfc_nmae", "lfc_nmae"),
)
DERIVED = "expr_mse_unbiased_capped_norm"


def _row_mean(store, rows: np.ndarray) -> np.ndarray:
    """Mean over `rows` (sorted, unique) of the local lognorm expression, computed on the
    sparse rows so a large row set (e.g. 75k control cells) is never densified."""
    return np.asarray(store.csr(rows).mean(axis=0), dtype=np.float32).ravel()


def _counts_from_local(store, rows: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """A context's stored lognorm rows -> integer counts in the global space.

    Works on the CSR triplet directly: remapping `indices` through `gene_idx`
    is the scatter, so nothing is ever densified to 18,533 columns.
    """
    a = store.csr(rows).tocsr()
    lib = np.repeat(store.lib[rows], np.diff(a.indptr))
    # float32 is exact for integer counts < 2^24 and halves the memory of every count matrix
    data = np.rint(np.expm1(a.data.astype(np.float64)) * lib / 1e6).astype(np.float32)
    out = sp.csr_matrix(
        (data, store.gene_idx[a.indices], a.indptr),
        shape=(len(rows), n_genes),
    )
    out.data = np.clip(out.data, 0.0, None)
    out.eliminate_zeros()
    return out


def _to_global(counts_local: sp.csr_matrix, gene_idx: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """Local-panel counts -> the global readout space (column remap, no densify)."""
    c = counts_local.tocsr()
    out = sp.csr_matrix((c.data, np.asarray(gene_idx)[c.indices], c.indptr), shape=(c.shape[0], n_genes))
    out.sort_indices()
    return out


def _adata(blocks: list[sp.csr_matrix], labels: list[str], genes) -> ad.AnnData:
    X = sp.vstack(blocks, format="csr")
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"target": labels}, index=pd.Index([f"c{i}" for i in range(X.shape[0])])
        ),
        var=pd.DataFrame(index=pd.Index(list(genes))),
    )


def _raw(pred: ad.AnnData, real: ad.AnnData, cfg):
    """Run the suite. Returns (the six panel values, the wide frame, a note)."""
    import cell_eval2 as ce

    res = ce.compute_metrics(pred, real, config=cfg)
    names = [n for n, _ in MEMBERS]

    note = None
    try:
        wide = ce.aggregate_metrics_wide(res)
    except ValueError as e:
        if DERIVED not in str(e):
            raise
        # DERIVED is a ratio of sums over the panel, and the reference's own
        # denominator goes non-positive when the measured effect does not clear
        # sampling noise at this cell depth. A property of the REFERENCE, so it
        # is the same for every method -- drop the member, not the run.
        note = "expr_mse unavailable: reference effect below sampling noise"
        wide = ce.aggregate_metrics_wide(res, metrics=[n for n in names if n != DERIVED])

    mean_row = wide.filter(wide["statistic"] == "mean")
    raw = {
        n: float(mean_row[n][0]) if n in wide.columns and len(mean_row) else float("nan")
        for n in names
    }

    return raw, wide, note


def _scale(raw: dict[str, float], wide, common: list[str]) -> dict[str, float]:
    """Place the members on the shipped 0 = no skill, 1 = perfect axis.

    `common` is the member set that is defined for the REFERENCE -- finite for
    both baselines -- so a member the data makes undefined (the reference's own
    effect is below sampling noise) is dropped for every method at once and the
    averages compare like with like. A member that is defined for the baselines
    but not for the model is the model's failure, and is scored as one: the
    library reads a non-finite model value as a degenerate output and takes the
    metric's floor, rather than the member quietly leaving the model's average.
    """
    import cell_eval2 as ce
    from cell_eval2 import scales
    from cell_eval2.scoring import score_one

    names = [n for n, _ in MEMBERS]
    if len(common) == len(names) and all(np.isfinite(raw[n]) for n in names):
        sc = ce.score_metrics(wide, scale=SCALE)
        col = sc.columns[-1]
        return {m: float(v) for m, v in zip(sc["metric"].to_list(), sc[col].to_list())}

    # score_metrics needs every member the scale names, and reads an absent one
    # as a degenerate MODEL output -- it takes clamp_low (-6) and drags the
    # average down. So score the members one at a time through the library's own
    # per-metric arithmetic and the same shipped constants.
    entries = scales.SCALES[SCALE].entries
    out = {n: float("nan") for n in [*names, "avg_score"]}
    for n in common:
        e = entries[n]
        out[n] = float(score_one(raw[n], e.base, e.scoring))
    vals = [out[n] for n in common if np.isfinite(out[n])]
    out["avg_score"] = float(np.mean(vals)) if vals else float("nan")
    return out


METHODS = ("identity", "mean_shift", "flow")
SHORT = [s for _, s in MEMBERS]


class _ToLog(logging.Handler):
    def __init__(self, write, label: str) -> None:
        super().__init__(level=logging.WARNING)
        self.write, self.label = write, label

    def emit(self, record: logging.LogRecord) -> None:
        self.write(f"[{self.label}] {record.name}: {record.getMessage()}")


@contextmanager
def captured(write, label: str):
    """Send cell-eval2's log records to `write(str)` instead of the terminal.

    cell-eval2 reports through Python's standard `logging` and configures no
    handler of its own, so its warnings reach the terminal only through logging's
    fallback for unhandled records. Attaching a handler to its logger for the
    duration of a scoring call redirects them, text unchanged. This only routes
    messages: the library's files and its computation are untouched, and the
    handler is removed again afterwards.
    """
    lg = logging.getLogger("cell_eval2")
    handler, was_propagating = _ToLog(write, label), lg.propagate
    lg.addHandler(handler)
    lg.propagate = False
    try:
        yield
    finally:
        lg.removeHandler(handler)
        lg.propagate = was_propagating


class ValScorer:
    """Scores identity / mean_shift / flow on the held-out context(s) with the VCC2026 suite.

    Everything that does not depend on the model is built ONCE: the real cells as counts,
    the source control cells drawn for each perturbation, each group's fingerprint row,
    and the two baselines. Scoring the model again after another epoch recomputes only
    the flow's predictions.

    mean_shift is the average shift of that perturbation in the TRAINING contexts ("if I
    knew what this knockdown does in the other lines"), applied to the model genes only,
    like the model. A perturbation no training context carries is not scored at all
    (`training/train.py` restricts the held-out dataset to trained perturbations).
    """

    def __init__(self, cfg, train_contexts, val_ds, device, space, n_steps=20, min_cells=5,
                 threads=-1, verbose=True, log=None, max_cells=0, cap=1e6):
        """`space`: pca_space.Space (the basis); `log`: optional write(str) for detail lines
        and cell-eval2's own warnings (see `captured`). `cap`: per-cell total count cap, the
        same one predict.py applies (the portal's max_counts_per_cell), so what is scored is
        what would be submitted -- a cell above it is scaled down to it."""
        self.cap = float(cap) if cap else None
        from ..models.linear_genes import outside_pcs

        self.outside = outside_pcs(cfg)          # model.outside_pcs: linear (see linear_genes.outside_pcs)
        import cell_eval2 as ce

        processed = Path(cfg["data"]["processed_dir"])
        self.genes = pd.read_csv(processed / "gene_vocab.csv").iloc[:, 0].to_numpy()
        pert_names = pd.read_csv(processed / "pert_vocab.csv").iloc[:, 0].to_numpy()
        self.G, self.device, self.n_steps = len(self.genes), device, n_steps
        self.loadings = space.loadings
        self.L = torch.from_numpy(space.loadings).to(device)
        self.mu = torch.from_numpy(space.mu).to(device)
        self._cache: dict = {}
        self._log = log
        say = log if log is not None else (print if verbose else (lambda s: None))
        self._say = say
        rng = np.random.default_rng(int(cfg.get("seed", 0)))

        def make_cfg(labels: list[str]) -> "ce.EvalConfig":
            # a target that is unmeasured in this context cannot resolve to a feature; the
            # pert vocab and the gene vocab are the same csv, so say so explicitly
            return ce.EvalConfig(
                metrics="vcc2026",
                pert_col="target",
                control=CONTROL_LABEL,
                input_type="counts",
                target_gene_map={g: g for g in labels if g != CONTROL_LABEL},
                num_threads=int(threads),
                device="cpu",
            )

        plan = []
        for pos, c in enumerate(val_ds.contexts):
            rows = val_ds.rows[pos]
            groups = []
            for p in np.unique(c.pert[rows]):
                tgt = rows[c.pert[rows] == p]
                if len(tgt) >= min_cells:
                    if max_cells and len(tgt) > max_cells:
                        tgt = np.sort(rng.choice(tgt, max_cells, replace=False))
                    groups.append((int(p), tgt))
            n_pert = sum(1 for p, _ in groups if p != 0)
            if n_pert < 2:
                say(f"  {c.name}: only {n_pert} scorable perturbation(s), skipped")
                continue
            plan.append((pos, c, groups, n_pert))

        wanted = {p for _, _, groups, _ in plan for p, _ in groups if p != 0}
        shift_sum, shift_cnt = self._cross_line_shift(train_contexts, wanted, self.G)

        self.ctx = []
        for pos, c, groups, n_pert in plan:
            pool = val_ds.control_pool[pos]
            order = local_model_order(c.model_col)
            model_global = c.gene_idx[order]
            real_blocks, real_lab, fixed = [], [], []
            for p, tgt in groups:
                real_blocks.append(_counts_from_local(c, tgt, self.G))
                real_lab += [str(pert_names[p])] * len(tgt)
                src = rng.choice(pool, size=len(tgt))
                shift = None
                if p != 0:
                    shift = (shift_sum[p][model_global] / np.maximum(shift_cnt[p][model_global], 1)).astype(np.float32)
                fixed.append(dict(p=p, n=len(src), src=src, lib0=c.lib[src], shift=shift,
                                  fp=torch.from_numpy(c.fp_rows[c.fp_lookup[p]][None, :].astype(np.float32)),
                                  state=None if c.state is None else torch.from_numpy(c.state_of(src)),
                                  anchor=None if c.anchor_s is None else torch.from_numpy(np.concatenate(
                                      [c.anchor_s[c.fp_lookup[p]], c.anchor_m])[None, :].astype(np.float32))))
            real = _adata(real_blocks, real_lab, self.genes)
            del real_blocks
            say(f"  {c.name}: {n_pert} perturbations, {real.n_obs} reference cells")
            self.ctx.append(dict(name=c.name, store=c, gene_idx=c.gene_idx, order=order, n_pert=n_pert,
                                 real=real, real_lab=real_lab, cfg=make_cfg(sorted(set(real_lab))),
                                 groups=fixed))
        del plan
        gc.collect()
        self.n_contexts = len(self.ctx)
        self.n_perts = sum(c["n_pert"] for c in self.ctx)
        self.n_cells = sum(c["real"].n_obs for c in self.ctx)

    @staticmethod
    def _cross_line_shift(train_contexts, wanted: set, G: int):
        """Per perturbation, the shift (perturbed mean - control mean, log1p(CPM)) averaged
        over the training contexts that carry it, in the GLOBAL gene space; each gene is
        averaged over the contexts that measured it."""
        shift_sum = {p: np.zeros(G, dtype=np.float64) for p in wanted}
        shift_cnt = {p: np.zeros(G, dtype=np.float64) for p in wanted}
        for t in train_contexts:
            present = wanted & set(np.unique(t.pert).tolist())
            if not present:
                continue
            ctrl_mean = _row_mean(t, np.sort(t.control_rows.astype(np.int64)))
            order = np.argsort(t.pert, kind="stable")
            sorted_p = t.pert[order]
            for p in present:
                lo, hi = np.searchsorted(sorted_p, [p, p + 1])
                delta = _row_mean(t, np.sort(order[lo:hi])) - ctrl_mean
                shift_sum[p][t.gene_idx] += delta
                shift_cnt[p][t.gene_idx] += 1
        return shift_sum, shift_cnt

    def perts(self) -> dict[str, list[int]]:
        """{context: the knockdowns scored there} (controls excluded) -- what `mean_dz` must cover."""
        return {c["name"]: [g["p"] for g in c["groups"] if g["p"] != 0] for c in self.ctx}

    # -- pieces ---------------------------------------------------------------
    def _score_pred(self, c, pred, label=""):
        sink = captured(self._log, label) if self._log is not None else nullcontext()
        with warnings.catch_warnings(), sink:
            warnings.simplefilter("ignore")
            return _raw(pred, c["real"], c["cfg"])

    def _predict(self, c, method: str, model=None, other_shift: dict | None = None,
                 mean_dz: dict | None = None, gene_pred: dict | None = None) -> ad.AnnData:
        """Every group's predicted counts in the global space: identity (no shift),
        mean_shift (cross-line shift on the model genes), flow (the model's PCA shift),
        flow_linear (flow on the model genes + `other_shift[pert]`, a log1p shift over the
        context's local genes, on every OTHER gene -- e.g. the linear population-mean model),
        mean / mean_linear (the MEAN MODEL alone: every source cell shifted by the same
        predicted average `mean_dz[pert]` [K], + `other_shift` for mean_linear).

        `mean_dz` given with flow / flow_linear = the TWO-STAGE model: the flow is centered
        (its own average over the group is removed) and the mean model's average is added,
        so the population mean comes from the mean model alone.

        THE CONTROL GROUP IS ALWAYS THE REAL, UNSHIFTED CONTROL CELLS, for every method: a
        submission holds no control cells (the portal supplies the real ones), so pushing
        controls through the model scored something that is never submitted (fix 2026-09-30;
        before, flow methods integrated the control group too).

        GENE MODEL (models/gene_model.py), `gene_pred` = {"mu": control mean per local gene,
        pert: (shift, lsr) per local gene}: "gene" = the per-gene density shift (move + stretch),
        "gene_mean" = move only, "gene_flow" = move + a CENTERED flow's scatter (stage 2)."""
        if method in ("gene", "gene_mean", "gene_flow"):
            return self._predict_gene(c, method, model, gene_pred)
        blocks = []
        for g in c["groups"]:
            x0 = c["store"].dense(g["src"])                       # [n, n_local] log1p(CPM)
            dz, shift, other = None, None, None
            if g["p"] == 0:
                pass                                              # real controls, see above
            elif method == "mean_shift":
                shift = g["shift"]
            elif method in ("mean", "mean_linear"):
                dz = np.broadcast_to(mean_dz[g["p"]][None, :], (g["n"], len(mean_dz[g["p"]])))
                if method == "mean_linear":
                    other = (other_shift or {}).get(g["p"])
            elif method in ("flow", "flow_linear"):
                if method == "flow_linear":
                    other = (other_shift or {}).get(g["p"])
                with torch.no_grad():
                    xm = torch.from_numpy(x0[:, c["order"]]).to(self.device)
                    z0 = (xm - self.mu) @ self.L.T
                    st = None if g["state"] is None else g["state"].to(self.device)
                    an = None if g["anchor"] is None else g["anchor"].to(self.device).expand(g["n"], -1)
                    z1 = integrate(model, z0, g["fp"].to(self.device).expand(g["n"], -1), n_steps=self.n_steps,
                                   state=st, anchor=an)
                    dz = (z1 - z0).cpu().numpy()
                if mean_dz is not None:
                    dz = dz - dz.mean(0, keepdims=True) + mean_dz[g["p"]][None, :]
            if self.outside and other is not None and method != "mean_shift":
                # the linear shift's part OUTSIDE the PCs, on the model genes (model.outside_pcs)
                from ..models.linear_genes import outside_part

                shift = outside_part(other[c["order"]], self.loadings)
            counts = residual_counts(x0, c["order"], dz, self.loadings, g["lib0"], shift=shift,
                                     other_shift=other, cap=self.cap)
            blocks.append(_to_global(counts, c["gene_idx"], self.G))
            del x0
        return _adata(blocks, c["real_lab"], self.genes)

    def _predict_gene(self, c, method, model, gene_pred) -> ad.AnnData:
        from ..models.gene_model import transform_counts

        blocks = []
        for g in c["groups"]:
            x0 = c["store"].dense(g["src"])
            if g["p"] == 0:
                counts = residual_counts(x0, c["order"], None, self.loadings, g["lib0"], cap=self.cap)
            else:
                shift, lsr = gene_pred[g["p"]]
                dz = None
                if method == "gene_flow":
                    with torch.no_grad():
                        z0 = (torch.from_numpy(x0[:, c["order"]]).to(self.device) - self.mu) @ self.L.T
                        st = None if g["state"] is None else g["state"].to(self.device)
                        z1 = integrate(model, z0, g["fp"].to(self.device).expand(g["n"], -1), n_steps=self.n_steps,
                                       state=st)
                        dz = (z1 - z0).cpu().numpy()
                    dz = dz - dz.mean(0, keepdims=True)
                counts = transform_counts(x0, shift, lsr if method == "gene" else None, gene_pred["mu"], g["lib0"],
                                          cap=self.cap, dz=dz, order=c["order"], loadings=self.loadings)
            blocks.append(_to_global(counts, c["gene_idx"], self.G))
            del x0
        return _adata(blocks, c["real_lab"], self.genes)

    def _baseline(self, c, method, tag=""):
        key = (c["name"], method)
        if key not in self._cache:
            base = self._predict(c, method)
            self._cache[key] = self._score_pred(c, base, f"{tag}{method} | {c['name']}")
            del base
        return self._cache[key]

    def _flow(self, c, model, tag="", method="flow", other_shift=None, mean_dz=None, gene_pred=None):
        return self._score_pred(c, self._predict(c, method, model, other_shift, mean_dz, gene_pred),
                                f"{tag}{method} | {c['name']}")

    def _common(self, c, tag=""):
        """Members defined for the reference: finite for both baselines."""
        if "common" not in c:
            rs = [self._baseline(c, m, tag)[0] for m in ("identity", "mean_shift")]
            names = [n for n, _ in MEMBERS]
            c["common"] = [n for n in names if all(np.isfinite(r[n]) for r in rs)]
            if len(c["common"]) < len(names):
                gone = ", ".join(s for n, s in MEMBERS if n not in c["common"])
                self._say(f"    {c['name']}: dropped for every method: {gone}  "
                          f"(avg over {len(c['common'])}/6)")
        return c["common"]

    # -- the one public call ----------------------------------------------------
    def score(self, model, methods=METHODS, tag="", other_shift: dict | None = None,
              mean_dz: dict | None = None, gene_pred: dict | None = None) -> pd.DataFrame:
        """One row per (context, method): the six raw members, their scaled values and the
        average. Baselines are scored once and cached. `other_shift` ({context: {pert: shift
        over its local genes}}) is what methods "flow_linear" / "mean_linear" put on the
        non-model genes. `mean_dz` ({context: {pert: [K] predicted average shift}}): the
        mean model's output -- used alone by "mean" / "mean_linear", and as the average of
        the centered flow by "flow" / "flow_linear" (see `_predict`)."""
        rows = []
        for c in self.ctx:
            common = self._common(c, tag)
            for m in methods:
                if m in ("flow", "flow_linear", "mean", "mean_linear", "gene", "gene_mean", "gene_flow"):
                    raw, wide, note = self._flow(c, model, tag, m, (other_shift or {}).get(c["name"]),
                                                 None if mean_dz is None else mean_dz[c["name"]],
                                                 None if gene_pred is None else gene_pred[c["name"]])
                else:
                    raw, wide, note = self._baseline(c, m, tag)
                scaled = _scale(raw, wide, common)
                row = {"context": c["name"], "method": m, "n_perts": c["n_pert"],
                       "n_members": len(common)}
                for name, short in MEMBERS:
                    row[short] = raw[name]
                    row[f"s_{short}"] = scaled[name]
                row["avg_score"] = scaled["avg_score"]
                row["note"] = note or ""
                rows.append(row)
        return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, method: str) -> dict[str, float]:
    """A method's mean over contexts: raw members, scaled members, avg_score."""
    d = df[df.method == method]
    cols = SHORT + [f"s_{s}" for s in SHORT] + ["avg_score"]
    out = {k: float(d[k].mean()) for k in cols}
    # a member is missing from a context when its reference is undefined there
    # (see `_common`), so say how many contexts each number is an average over
    out.update({f"n_{s}": int(d[f"s_{s}"].notna().sum()) for s in SHORT})
    out["n_contexts"] = int(d["context"].nunique())
    return out


def format_report(s: dict, title: str, ref_avg: float | None = None, note: str = "") -> str:
    """The six members and ONE average, as a small table for the terminal.

    `raw` is the metric as cell-eval2 defines it (lower is better for expr_mse and
    lfc_nmae); `scaled` puts all six on one axis (0 = no skill, 1 = perfect);
    `contexts` is how many contexts the number averages over. AVERAGE is the mean,
    over contexts, of each context's average over the members it has.
    """
    nc = s["n_contexts"]
    rows = [f"  {title}", f"    {'':13s}{'raw':>9s}{'scaled':>9s}{'contexts':>10s}"]
    for _, short in MEMBERS:
        rows.append(f"    {short:13s}{s[short]:9.3f}{s['s_' + short]:+9.3f}{s['n_' + short]:>7d}/{nc}")
    tail = f"    {'AVERAGE':13s}{'':9s}{s['avg_score']:+9.3f}"
    if ref_avg is not None:
        tail += f"     (mean_shift {ref_avg:+.3f})"
    if note:
        tail += f"  {note}"
    rows.append(tail)
    return "\n".join(rows)


def format_overview(flow: dict, ref: dict, title: str, note: str = "") -> str:
    """After every cell-eval: the model next to the two baselines, scaled scores per member
    (the same layout as `format_reference`, with the model as an extra column). A model that
    beats mean_shift on the AVERAGE row is doing better than the trivial baselines."""
    cols = (("model", flow), *((_label(m), d) for m, d in ref.items()))
    rows = [f"  {title}", f"    {'scaled':13s}" + "".join(f"{n:>12s}" for n, _ in cols)]
    for short in SHORT:
        rows.append(f"    {short:13s}" + "".join(f"{d['s_' + short]:+12.3f}" for _, d in cols))
    rows.append(f"    {'AVERAGE':13s}" + "".join(f"{d['avg_score']:+12.3f}" for _, d in cols))
    if note:
        rows.append(f"    {note}")
    return "\n".join(rows)


def _label(method: str) -> str:
    """Column label of a reference method (the mean model's methods read as `mean_model`)."""
    return {"mean": "mean_model", "mean_linear": "mean_model", "gene": "gene_model",
            "gene_mean": "gene_shift"}.get(method, method)


def format_reference(ref: dict) -> str:
    """The two baselines' scaled scores, per member, printed once at the start."""
    rows = ["  reference scores, scaled (0 = no skill, 1 = perfect)",
            f"    {'':13s}" + "".join(f"{_label(m):>12s}" for m in ref)]
    for short in SHORT:
        rows.append(f"    {short:13s}" + "".join(f"{d['s_' + short]:+12.3f}" for d in ref.values()))
    rows.append(f"    {'AVERAGE':13s}" + "".join(f"{d['avg_score']:+12.3f}" for d in ref.values()))
    if any(m.startswith("mean") and m != "mean_shift" for m in ref):
        rows.append("  mean_model = the frozen MEAN MODEL alone (every control cell + its predicted average shift):")
        rows.append("  the two-stage model only earns its flow if it beats this column.")
    rows.append("  AVERAGE = mean over contexts of each context's average over the members it has. A member is")
    rows.append("  missing in a context where the real data cannot support it (too few cells), so in the tables")
    rows.append("  below AVERAGE is not the plain mean of the six rows; `contexts` shows how many each covers.")
    return "\n".join(rows)
