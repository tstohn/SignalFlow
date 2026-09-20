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
    Predicted cells reuse their SOURCE control cell's library size, which is
    what `flow.to_counts` does and the same caveat applies: a perturbation
    that shifts sequencing depth is not modelled.

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

import logging
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ..data.vocab import CONTROL_LABEL
from ..models.flow import integrate, mask_from_gene_idx

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


def _counts_from_local(store, rows: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """A context's stored lognorm rows -> integer counts in the global space.

    Works on the CSR triplet directly: remapping `indices` through `gene_idx`
    is the scatter, so nothing is ever densified to 18,533 columns.
    """
    a = store.X[rows].tocsr()
    lib = np.repeat(store.lib[rows], np.diff(a.indptr))
    data = np.rint(np.expm1(a.data.astype(np.float64)) * lib / 1e6)
    out = sp.csr_matrix(
        (data, store.gene_idx[a.indices], a.indptr),
        shape=(len(rows), n_genes),
    )
    out.data = np.clip(out.data, 0.0, None)
    out.eliminate_zeros()
    return out


def _counts_from_global(x_lognorm: np.ndarray, lib: np.ndarray) -> sp.csr_matrix:
    """A predicted lognorm block -> integer counts, at the source cell's depth."""
    counts = np.rint(np.expm1(x_lognorm.astype(np.float64)) * lib[:, None] / 1e6)
    return sp.csr_matrix(np.clip(counts, 0.0, None))


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

    Everything that does not depend on the model is built ONCE, here: the real
    cells as counts, the source control cells drawn for each perturbation, and
    the two baselines' predictions. Scoring the model again after another epoch
    therefore recomputes only the flow's predictions -- so the numbers move when
    the model does and for no other reason, and each epoch costs one flow pass
    plus cell-eval2's own scoring, not a rebuild of the reference.

    The held-out context is a whole cell line the model never trained on, so its
    `mean_shift` baseline cannot come from its own cells. It is instead the average
    shift of that perturbation in the TRAINING contexts ("if I knew what this
    knockout does in the other lines"): a genuinely informative baseline, and an
    honest one. A perturbation that no training context carries has no such
    baseline and is not scored at all -- `training/train.py` restricts the held-out
    dataset to perturbations the training contexts have seen, because the model
    cannot be expected to know an unseen one.

    Used by `training/train.py`, after every `train.cell_eval_every` epochs.
    """

    def __init__(self, cfg, train_contexts, val_ds, device, n_steps=20, min_cells=5,
                 threads=-1, verbose=True, log=None):
        """`log`, if given, is a `write(str)` that receives everything this class
        would otherwise print, plus cell-eval2's own warnings (see `captured`), so
        the terminal can stay quiet. Without it, behaviour is as before: detail
        lines are printed and the library's warnings show up on the terminal."""
        import cell_eval2 as ce

        processed = Path(cfg["data"]["processed_dir"])
        self.genes = pd.read_csv(processed / "gene_vocab.csv").iloc[:, 0].to_numpy()
        pert_names = pd.read_csv(processed / "pert_vocab.csv").iloc[:, 0].to_numpy()
        self.G, self.device, self.n_steps = val_ds.n_genes, device, n_steps
        self._cache: dict = {}
        self._log = log
        say = log if log is not None else (print if verbose else (lambda s: None))
        self._say = say
        G = self.G
        rng = np.random.default_rng(int(cfg.get("seed", 0)))

        def make_cfg(labels: list[str]) -> "ce.EvalConfig":
            # Target resolution runs AFTER the 5-CPM gate, so a target gene that
            # is unmeasured (or silent) in this context is not a feature and
            # cannot be resolved -- and cell_eval2 refuses to score when nothing
            # resolves, to catch construct-ID-vs-symbol mismatches. There is no
            # mismatch here: the pert vocab and the gene vocab are literally the
            # same csv. So say so explicitly. A target that is present gets
            # excluded as usual; one that is not excludes nothing, which is the
            # correct answer.
            return ce.EvalConfig(
                metrics="vcc2026",
                pert_col="target",
                control=CONTROL_LABEL,
                input_type="counts",
                target_gene_map={g: g for g in labels if g != CONTROL_LABEL},
                num_threads=int(threads),
                device="cpu",
            )

        # which (context, perturbation) groups are large enough to score
        plan = []
        for pos, c in enumerate(val_ds.contexts):
            rows = val_ds.rows[pos]
            groups = []
            for p in np.unique(c.pert[rows]):
                tgt = rows[c.pert[rows] == p]
                if len(tgt) >= min_cells:
                    groups.append((int(p), tgt))
            n_pert = sum(1 for p, _ in groups if p != 0)
            if n_pert < 2:
                say(f"  {c.name}: only {n_pert} scorable perturbation(s), skipped")
                continue
            plan.append((pos, c, groups, n_pert))

        wanted = {p for _, _, groups, _ in plan for p, _ in groups if p != 0}
        shift_sum, shift_cnt = self._cross_line_shift(train_contexts, wanted, G)

        self.ctx = []
        for pos, c, groups, n_pert in plan:
            pool = val_ds.control_pool[pos]

            real_blocks, real_lab, fixed = [], [], []
            base = {"identity": [], "mean_shift": []}
            for p, tgt in groups:
                label = str(pert_names[p])
                real_blocks.append(_counts_from_local(c, tgt, G))
                real_lab += [label] * len(tgt)

                src = rng.choice(pool, size=len(tgt))
                x0_local = c.dense(src)
                lib0 = c.lib[src]

                if p == 0:
                    shift = np.zeros(len(c.gene_idx), dtype=np.float32)
                else:
                    shift = (shift_sum[p][c.gene_idx] / np.maximum(shift_cnt[p][c.gene_idx], 1)).astype(np.float32)
                x0_g = np.zeros((len(src), G), dtype=np.float32)
                x0_g[:, c.gene_idx] = x0_local
                ms_g = np.zeros_like(x0_g)
                ms_g[:, c.gene_idx] = x0_local + shift[None, :]
                base["identity"].append(_counts_from_global(x0_g, lib0))
                base["mean_shift"].append(_counts_from_global(ms_g, lib0))
                # the perturbation's per-cell-line correlation row (data/gene_corr.py),
                # scattered into the global gene space once and reused for every cell of
                # this group -- it describes the perturbation in this line, not the cell
                pc_g = np.zeros((1, G), dtype=np.float32)
                row = c.corr_row(p) if p != 0 else None
                if row is not None:
                    pc_g[0, c.gene_idx] = row
                fixed.append(
                    dict(p=p, n=len(src), lib0=lib0,
                         x0=torch.from_numpy(x0_g),
                         pcorr=torch.from_numpy(pc_g),
                         pcorr_ok=torch.full((1, 1), 0.0 if row is None else 1.0),
                         state=torch.from_numpy(c.state[src]))
                )

            real = _adata(real_blocks, real_lab, self.genes)
            say(f"  {c.name}: {n_pert} perturbations, {real.n_obs} reference cells")
            self.ctx.append(
                dict(
                    name=c.name, gene_idx=c.gene_idx, n_pert=n_pert, real=real,
                    real_lab=real_lab, cfg=make_cfg(sorted(set(real_lab))), groups=fixed,
                    base={m: _adata(b, real_lab, self.genes) for m, b in base.items()},
                )
            )
        self.n_contexts = len(self.ctx)
        self.n_perts = sum(c["n_pert"] for c in self.ctx)
        self.n_cells = sum(c["real"].n_obs for c in self.ctx)

    @staticmethod
    def _cross_line_shift(train_contexts, wanted: set, G: int):
        """Per perturbation, the shift (perturbed mean - control mean) averaged over the
        training contexts that carry it, in the GLOBAL gene space. Genes a context did
        not measure contribute nothing, so each gene is averaged over the contexts that
        measured it (and is 0 where none did)."""
        shift_sum = {p: np.zeros(G, dtype=np.float64) for p in wanted}
        shift_cnt = {p: np.zeros(G, dtype=np.float64) for p in wanted}
        for t in train_contexts:
            present = wanted & set(np.unique(t.pert).tolist())
            if not present:
                continue
            ctrl_mean = t.dense(t.control_rows.astype(np.int64)).mean(0)
            order = np.argsort(t.pert, kind="stable")
            sorted_p = t.pert[order]
            for p in present:
                lo, hi = np.searchsorted(sorted_p, [p, p + 1])
                delta = t.dense(order[lo:hi]).mean(0) - ctrl_mean
                shift_sum[p][t.gene_idx] += delta
                shift_cnt[p][t.gene_idx] += 1
        return shift_sum, shift_cnt

    # -- pieces ---------------------------------------------------------------
    def _score_pred(self, c, pred, label=""):
        sink = captured(self._log, label) if self._log is not None else nullcontext()
        with warnings.catch_warnings(), sink:
            warnings.simplefilter("ignore")
            return _raw(pred, c["real"], c["cfg"])

    def _baseline(self, c, method, tag=""):
        key = (c["name"], method)
        if key not in self._cache:
            self._cache[key] = self._score_pred(c, c["base"][method], f"{tag}{method} | {c['name']}")
        return self._cache[key]

    def _flow(self, c, model, tag=""):
        blocks = []
        for g in c["groups"]:
            with torch.no_grad():
                out = integrate(
                    model,
                    g["x0"].to(self.device),
                    torch.full((g["n"],), g["p"], dtype=torch.long, device=self.device),
                    g["state"].to(self.device),
                    mask_from_gene_idx(c["gene_idx"], self.G, g["n"]).to(self.device),
                    n_steps=self.n_steps,
                    pert_corr=g["pcorr"].to(self.device).expand(g["n"], self.G),
                    pert_corr_ok=g["pcorr_ok"].to(self.device).expand(g["n"], 1),
                ).cpu().numpy()
            blocks.append(_counts_from_global(out, g["lib0"]))
        return self._score_pred(c, _adata(blocks, c["real_lab"], self.genes), f"{tag}flow | {c['name']}")

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
    def score(self, model, methods=METHODS, tag="") -> pd.DataFrame:
        """One row per (context, method): the six raw members, their scaled values
        and the average. Baselines are scored once and cached. `tag` (e.g.
        "epoch 5 | ") prefixes the label on any log lines this call produces."""
        rows = []
        for c in self.ctx:
            common = self._common(c, tag)
            for m in methods:
                raw, wide, note = self._flow(c, model, tag) if m == "flow" else self._baseline(c, m, tag)
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


def format_reference(ref: dict) -> str:
    """The two baselines' scaled scores, per member, printed once at the start."""
    ident, shift = ref["identity"], ref["mean_shift"]
    rows = ["  reference scores, scaled (0 = no skill, 1 = perfect)",
            f"    {'':13s}{'identity':>10s}{'mean_shift':>12s}"]
    for short in SHORT:
        rows.append(f"    {short:13s}{ident['s_' + short]:+10.3f}{shift['s_' + short]:+12.3f}")
    rows.append(f"    {'AVERAGE':13s}{ident['avg_score']:+10.3f}{shift['avg_score']:+12.3f}")
    rows.append("  AVERAGE = mean over contexts of each context's average over the members it has. A member is")
    rows.append("  missing in a context where the real data cannot support it (too few cells), so in the tables")
    rows.append("  below AVERAGE is not the plain mean of the six rows; `contexts` shows how many each covers.")
    return "\n".join(rows)
