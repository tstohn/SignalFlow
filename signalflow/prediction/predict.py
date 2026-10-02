"""Predict perturbed cells for ANY unperturbed cells, as raw UMI counts.

    python -m signalflow.prediction.predict --config configs/prototype.yaml \
        --checkpoint runs/prototype/full/last.pt \
        --input <file.h5ad | folder of .h5ad> \
        --perts <perturbations.csv> \
        --out runs/prototype/predictions.h5ad

`--input` is one `.h5ad` or a folder of them (every `.h5ad` in the folder is
read). Each holds unperturbed cells -- one cell line, or several if the file
has a `--context-col` column -- and every cell line becomes one context in the
output. For each perturbation in `--perts` and each context, the flow is run
from `--cells-per-pert` of that context's cells and the predicted lognorm is
converted back to raw UMI counts.

THE FILE THIS WRITES, AND WHAT HAPPENS NEXT
    An `.h5ad`: raw counts in `.X` (sparse, float32, whole numbers),
    `.obs[--pert-col]` and `.obs[--context-col]`, and `.var` holding gene
    symbols in the model's vocabulary order. That is exactly what `vcc prep`
    takes as input; prep validates it and packages it into the `.vcc` the
    challenge portal accepts. `submit_vcc26.py` runs those steps. This script
    never uploads anything.

    float32 and the exact gene order are deliberate: they are the case in which
    `vcc prep` needs the least RAM (no cast, no reordering copy).

OPTIONAL MANIFEST
    `--manifest manifest.json` (the VCC26 controls folder ships one) supplies
    defaults instead of flags: `cells_per_pert`, `pert_col`, `context_col`,
    `control_label`, `max_counts_per_cell` if present, and `n_genes`, checked
    against the model's vocabulary. An explicit flag always overrides the
    manifest. Its `contexts` list is also enforced: the input must provide
    exactly those contexts, checked from `.obs` alone BEFORE any prediction
    starts, and again on the finished file. It does not carry the perturbation
    list, so `--perts` is still required.

WHAT THE INPUT MUST BE
    Raw integer counts in `.X`, gene SYMBOLS in `.var_names`. Both are checked:
    log-normalized input would pass through `_lognorm` and come out looking
    plausible and being wrong. If `.obs[--pert-col]` exists only its
    `non-targeting` rows are used as starting cells -- a perturbed cell is not
    a valid source, since at inference the model only ever starts from a
    control. With no such column every cell is taken to be unperturbed.

    Genes the model's vocabulary does not know are dropped; a cell line that
    measures only part of the vocabulary is fine -- the rest is masked, in the
    model's input and output alike.

THE PCA SPACE COMES FROM THE CHECKPOINT'S OWN FOLDER
    `train` copies the basis (`pca.npz`), the model genes and the delta-fingerprint
    table (`fingerprints.npz`) next to every checkpoint, so they always travel with the
    model they belong to. `--checkpoint` is therefore required, and nothing is read from
    the processed directory's PCA artifacts.

NO CONTEXT MAPPING
    The model has no notion of "which cell line": each input cell line supplies its own
    control cells (projected into the frozen PCA space) and its own covariance
    fingerprints. It must measure every model gene.

THE OUTPUT'S GUARANTEES
    The VCC2026 portal's file rules, the strictest format this repo scores
    against. Each is a parameter, so any other dataset gets the same guarantees:

    1.  contexts labelled as in the input (`--context-col`, else file name);
        with `--manifest`, exactly the contexts it lists
    2.  a column named `--pert-col` (default `target_gene`) holding gene
        symbols, not construct ids (`ADNP`, not `ADNP-1`)
    3.  exactly the perturbations in `--perts` -- no extras, none missing
    4.  exactly `--cells-per-pert` cells (default 400) for every perturbation
        in every context
    5.  `.var` is the model's readout vocabulary, in its order.
        `--reference-genes` additionally asserts that order equals a given
        gene list (the portal's `gene_names.csv`)
    6.  raw counts in `.X`: non-negative, whole, finite. Never `normalize_total`
        or `log1p` -- scoring runs in counts space
    7.  NO non-targeting rows. Control cells are model INPUT only
    8.  at most `--max-stored-entries` (default 4.75e9) stored entries, so `.X`
        is sparse with no explicitly-stored zeros
    9.  no cell above `--max-counts-per-cell` (default 1e6) total counts
    10. at most `--max-cells` (default 400,000) cells in total

    Label rules (1-5, 7, 10) are checked before any prediction runs -- they
    follow from the arguments, so a wrong one fails in seconds, not after the
    run. Value rules (6, 8, 9) are checked on every block as it is written and
    again on what landed on disk.

MEMORY
    The output is written one perturbation block at a time straight into the
    file, so memory does not grow with its size. A full VCC26 panel is ~2
    billion stored entries, ~17 GB if held in RAM; here it is one block (~20
    MB) plus one input context at a time. The file is written to
    `<out>.partial` and moved into place only once every check has passed, so
    a failed or interrupted run never leaves a plausible-looking file at `--out`.

GETTING BACK TO COUNTS
    `models.flow.residual_counts`, the same function the training-time scorer uses: the
    source control cell's log1p(CPM) plus the back-projected predicted PCA shift on the
    model genes, un-logged and scaled to the source cell's library size; every other
    gene keeps the source cell's raw count. Clipped to the per-cell cap and rounded.

THE CELL STATE
    A model trained with data.pca.state_knn = k > 0 also conditions on each source
    cell's state: the mean PC score of its k nearest cells among THIS input line's
    unperturbed cells (itself included) -- `pca_space.knn_mean`, the same function
    `prepare` runs on the training lines' controls. Whether the checkpoint uses it is
    read from the checkpoint itself, so older checkpoints still load.

THE GENES THE FLOW DOES NOT PREDICT (model.other_genes of the --config you pass)
    source  they keep the source control cell's raw counts
    linear  they are shifted by the linear population-mean model (models/linear_genes.py):
            coefficients + the training lines' mean effects from the run's linear_genes.npz
            (fit on the run's training lines and saved there on first use if the run
            predates the option), slopes from THIS input line's unperturbed cells.

THE TWO-STAGE MODEL (mean model + centered flow)
    --checkpoint a mean.pt (runs/<config>/mean/full/mean.pt): the MEAN MODEL ALONE -- every
        source cell is shifted by its knockdown's predicted average (no flow).
    --checkpoint a flow trained with model.centered: true: the mean model beside it (the mean.pt
        train copied into the flow's folder) gives the average, the flow the scatter around it
        (its own average over each knockdown's cells removed).
    The mean model's view of a line is computed HERE from THIS input line's control cells --
    its average control cell, the slope on each target gene, the target gene's expression --
    exactly as for the training lines. Nothing per cell line is stored in any model, so a line
    no model has seen gets the same treatment as any other.

THE PERTURBATION FINGERPRINTS
    delta: the knockdown's mean effect over the TRAINING lines (fingerprints.npz); a
    perturbation no training line measured has none. cov: the covariance of the
    knocked-out gene with the model genes over THIS input line's control cells,
    computed here -- it needs no training data, but it needs the knocked-out gene to be
    measured (and to vary) in this line. The run prints how many perturbations get
    each; one with neither gets only the model's generic "unknown knockdown" response.

GENE VOCABULARIES MUST MATCH, WITH NO OVERRIDE
    The model's readout vocabulary is what the output's `.var` is built from.
    If it disagrees with the manifest's `n_genes`, or with `--reference-genes`
    in count or in order, that is a hard error: there is no flag to write
    anyway, because such a file is unusable rather than merely poor. Fix the
    cause (retrain with `data.gene_vocab_csv` pointing at the right list).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

from ..data import pca_space
from ..data.vocab import CONTROL_LABEL, GeneVocab
from ..models import linear_genes
from ..models.build import build_model, pick_device
from ..models.flow import integrate, residual_counts
from ..submission.file_rules import check_layout
from .counts_writer import CountsWriter

CHUNK = 2048
PROGRESS_EVERY = 50


def _lognorm(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Raw counts -> CPM + log1p, the space the model lives in."""
    x = counts.astype(np.float64).tocsr()
    tot = np.asarray(x.sum(axis=1)).ravel()
    inv = np.divide(1e6, tot, out=np.zeros_like(tot), where=tot > 0)
    x.data *= np.repeat(inv, np.diff(x.indptr))
    x.data = np.log1p(x.data)
    return x.astype(np.float32)


def _check_counts(X: sp.csr_matrix, where: str) -> None:
    """Refuse anything but raw integer counts, on a spread of rows."""
    step = max(1, X.shape[0] // 2000)
    d = X[np.arange(0, X.shape[0], step)].data
    if d.size and (d.min() < 0 or not np.array_equal(d, np.rint(d))):
        raise SystemExit(
            f"{where}: .X must hold raw integer counts (non-negative, whole numbers); "
            f"this looks normalized or log-transformed"
        )


def _input_files(path: Path) -> list[Path]:
    if not path.exists():
        raise SystemExit(f"--input {path} does not exist")
    files = sorted(path.glob("*.h5ad")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no .h5ad files in {path}")
    return files


def _plan(obs: pd.DataFrame, name: str, pert_col: str, context_col: str, control: str):
    """Which rows are usable starting cells, and the context each belongs to.

    Shared by `peek_contexts` and `iter_contexts` so the labels announced up
    front are exactly the ones later produced.
    """
    if pert_col in obs:
        unperturbed = (obs[pert_col].astype(str) == control).to_numpy()
        if not unperturbed.any():
            raise SystemExit(
                f"{name}: .obs[{pert_col!r}] has no {control!r} cells; "
                f"the model only ever starts from unperturbed cells"
            )
    else:
        unperturbed = np.ones(len(obs), dtype=bool)
    labels = (
        obs[context_col].astype(str).to_numpy()
        if context_col in obs
        else np.full(len(obs), name)
    )
    return unperturbed, labels


def peek_contexts(path: Path, pert_col: str, context_col: str, control: str) -> list[str]:
    """The context labels the input will produce, IN THE ORDER they are produced,
    from `.obs` alone.

    Opened backed, so no matrix is loaded. That is what lets everything that
    depends only on the labels -- a manifest's context list, the total cell
    count, the output's `.obs` -- be known and checked before a prediction that
    takes far longer than this.
    """
    out: list[str] = []
    for f in _input_files(path):
        a = ad.read_h5ad(f, backed="r")
        unperturbed, labels = _plan(a.obs, f.name, pert_col, context_col, control)
        for label in dict.fromkeys(labels[unperturbed]):
            if label in out:
                raise SystemExit(f"context label {label!r} appears in more than one input")
            out.append(label)
        a.file.close()
    return out


def iter_contexts(path: Path, pert_col: str, context_col: str, control: str):
    """Yield (label, raw counts, var_names) per context, one file at a time.

    One file at a time on purpose: a 18,400-cell control file is ~0.9 GB
    sparse, and holding every context at once is a multiple of that.
    """
    for f in _input_files(path):
        a = ad.read_h5ad(f)
        unperturbed, labels = _plan(a.obs, f.name, pert_col, context_col, control)
        X = sp.csr_matrix(a.X)
        var_names = a.var_names.astype(str).tolist()
        for label in dict.fromkeys(labels[unperturbed]):
            block = X[unperturbed & (labels == label)]
            _check_counts(block, f"{f.name} [{label}]")
            yield label, block, var_names


def read_perts(path: Path, pert_col: str) -> list[str]:
    """Perturbation symbols from a csv WITH a header: a `pert_col` column, or
    a single column of any name."""
    df = pd.read_csv(path)
    if pert_col in df.columns:
        col = pert_col
    elif df.shape[1] == 1:
        col = df.columns[0]
    else:
        raise SystemExit(
            f"{path}: expected a {pert_col!r} column or a single column, got {list(df.columns)}"
        )
    return list(dict.fromkeys(df[col].astype(str)))


def preflight(
    want_perts, genes, meta, reference_genes, expected_n_genes=None, trained_perts=None
) -> list[str]:
    """Raise on a vocabulary mismatch; return warnings about the model's reach.

    A vocabulary mismatch is never acceptable and has no override. Untrained
    perturbations are different: the file is well-formed, the model just cannot
    speak to them -- so that is a loud warning, returned for the caller to print.
    """
    warnings: list[str] = []

    if expected_n_genes is not None and int(expected_n_genes) != len(genes):
        raise SystemExit(
            f"gene vocabulary mismatch: the model's readout vocabulary has {len(genes)} "
            f"genes; the manifest expects {expected_n_genes}. Retrain with "
            f"data.gene_vocab_csv pointing at the right list."
        )

    if reference_genes is not None and list(reference_genes) != list(genes):
        raise SystemExit(
            f"gene vocabulary mismatch: the model's readout vocabulary ({len(genes)} genes) "
            f"is not --reference-genes ({len(reference_genes)} genes) in the same order. "
            f"Retrain with data.gene_vocab_csv pointing at that file."
        )

    if trained_perts is None:
        trained_perts = set()
        for c in meta["contexts"]:
            trained_perts.update(c["perts"])
    unseen = [p for p in want_perts if p not in trained_perts]
    if unseen:
        warnings.append(
            f"{len(unseen)}/{len(want_perts)} perturbations were never trained "
            f"(e.g. {', '.join(unseen[:4])}). They have no delta fingerprint; only the "
            f"covariance fingerprint from the input controls speaks for them, and only "
            f"where this line measures the knocked-out gene"
        )
    return warnings


# ---- the GENE MODEL (models/gene_model.py) ---------------------------------------------------------

def _gene_sources(run_dir: Path, G: int, pert_index: dict):
    """The training lines' per-knockdown statistics over all genes (gene_tables.npz, written by
    train_gene next to gene.pt) as the gene model's only source."""
    from ..models import gene_model as gm

    tab = np.load(run_dir / "gene_tables.npz")
    src = gm.Sources(G, len(pert_index))
    src.add("train", [pert_index[str(n)] for n in tab["pert_names"]], tab["d"], tab["v"], np.arange(G),
            ok=tab["ok"], gen_d=tab["gen_d"], gen_v=tab["gen_v"])
    return src


def _gene_line_predictions(gmodel, src, label, x_log, gene_idx, want_perts, pert_index, pert_names_vocab,
                           gene_names, device, seed, chunk: int = 16):
    """(control mean per local gene, {j: [n_local, 2] (shift, log sd ratio)}) for every wanted
    knockdown j of ONE input line -- everything about the line from its own control cells."""
    from ..models import gene_model as gm

    mu, sd = gm.line_stats_from_controls(x_log)
    local = {str(gene_names[g]): j for j, g in enumerate(gene_idx)}
    has = [j for j, p in enumerate(want_perts) if p in local]
    slope, s_perts = np.zeros((0, len(gene_idx)), np.float16), []
    if has:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(x_log.shape[0], size=min(linear_genes.CTRL_CELLS, x_log.shape[0]), replace=False))
        sl, ok = linear_genes.slope_matrix(np.asarray(x_log[rows].todense(), dtype=np.float32),
                                           np.array([local[want_perts[j]] for j in has]), device)
        slope = sl[ok].astype(np.float16)
        s_perts = [pert_index[want_perts[j]] for k, j in enumerate(has) if ok[k]]
    line = gm.Line(label, gene_idx, mu, sd, s_perts, slope, gene_names)
    idx = np.array([pert_index[p] for p in want_perts], dtype=np.int64)
    out = {}
    for lo in range(0, len(idx), chunk):
        P = gm.predict(gmodel, line.features(src, ["train"], idx[lo:lo + chunk], pert_names_vocab), device)
        out.update({lo + i: P[i] for i in range(len(P))})
    return mu, out


def _predict_gene_only(ckpt, ck, args, cfg, genes, gene_vocab, G, contexts, want_perts, pert_index, pert_col,
                       ctx_col, control, per_pert, cap, seed, obs, expect_labels, processed) -> None:
    """--checkpoint a gene.pt: every control cell of each input line moved (and stretched) gene by
    gene -- the gene model alone, no PCA, no flow."""
    from ..models import gene_model as gm

    device = pick_device(cfg["train"].get("device", "auto"))
    gmodel, _ = gm.load(ckpt, device)
    spread = bool(gm.gene_cfg(ck["config"])["spread"])
    pert_names_vocab = pd.read_csv(processed / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    print(f"  model: GENE MODEL ONLY (per-gene density shift, {'move + stretch' if spread else 'move only'}, "
          f"all genes the input measures)")
    src = _gene_sources(ckpt.parent, G, pert_index)
    out_path = Path(args.out or ckpt.parent / "predictions.h5ad")
    writer = CountsWriter(out_path, obs=obs, var=pd.DataFrame(index=pd.Index(genes)),
                          uns={"signalflow": {"checkpoint": str(ckpt), "input": str(args.input), "seed": seed,
                                              "partial": bool(args.dry_run_perts)}},
                          n_genes=G, cap=cap, max_stored=args.max_stored_entries)
    rng = np.random.default_rng(seed)
    for k, (label, counts, var_names) in enumerate(iter_contexts(Path(args.input), pert_col, ctx_col, control)):
        if label != contexts[k]:
            writer._fail(f"internal error: context order changed ({label!r} != {contexts[k]!r})")
        known = np.array([g in gene_vocab for g in var_names])
        if not known.all():
            counts = counts[:, known]
        gene_idx = np.array(gene_vocab.indices([g for g, kn in zip(var_names, known) if kn]), dtype=np.int64)
        x_log = _lognorm(counts)
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        t0 = time.time()
        mu, preds = _gene_line_predictions(gmodel, src, label, x_log, gene_idx, want_perts, pert_index,
                                           pert_names_vocab, genes, device, seed)
        print(f"  context {label}: {counts.shape[0]:,} unperturbed cells, {len(gene_idx):,} genes -> "
              f"{len(want_perts)} perturbations  [gene model {time.time() - t0:.0f}s]")
        for j in range(len(want_perts)):
            srcr = rng.choice(counts.shape[0], size=per_pert, replace=per_pert > counts.shape[0])
            x0 = np.asarray(x_log[srcr].todense(), dtype=np.float32)
            local_counts = gm.transform_counts(x0, preds[j][:, 0], preds[j][:, 1] if spread else None, mu,
                                               lib[srcr], cap=cap)
            block = sp.csr_matrix((local_counts.data, gene_idx[local_counts.indices], local_counts.indptr),
                                  shape=(per_pert, G))
            block.sort_indices()
            writer.append(block)
    writer.finish(genes, pert_col, ctx_col, expect_labels)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    if args.dry_run_perts:
        print("  PARTIAL file (--dry-run-perts): the portal would reject it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="the .pt to predict with. Its run folder must also hold pca.npz and "
                         "fingerprints.npz (train writes them)")
    ap.add_argument("--input", required=True, help="an .h5ad, or a folder of .h5ad, of unperturbed cells")
    ap.add_argument("--perts", required=True, help="csv (with header) of the perturbations to predict")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None, help="manifest.json supplying defaults for the four options below, and the contexts to expect")
    ap.add_argument("--cells-per-pert", type=int, default=None, help="default 400")
    ap.add_argument("--pert-col", default=None, help="default target_gene")
    ap.add_argument("--context-col", default=None, help="default context")
    ap.add_argument("--max-counts-per-cell", type=float, default=None, help="default 1e6")
    ap.add_argument("--max-stored-entries", type=int, default=4_750_000_000)
    ap.add_argument("--max-cells", type=int, default=400_000)
    ap.add_argument("--reference-genes", default=None, help="assert the output gene order equals this list")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--dry-run-perts",
        type=int,
        default=0,
        help="only predict the first N perturbations; the file is PARTIAL, which "
        "the portal would reject, for testing the pipeline",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # a flag you pass beats the manifest, which beats the built-in default
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else {}

    def pick(cli, key, default):
        return cli if cli is not None else manifest.get(key, default)

    per_pert = int(pick(args.cells_per_pert, "cells_per_pert", 400))
    pert_col = pick(args.pert_col, "pert_col", "target_gene")
    ctx_col = pick(args.context_col, "context_col", "context")
    cap = float(pick(args.max_counts_per_cell, "max_counts_per_cell", 1e6))
    control = manifest.get("control_label", CONTROL_LABEL)
    expect_contexts = manifest.get("contexts")
    if args.manifest:
        print(
            f"  manifest {args.manifest}: contexts {expect_contexts}, {per_pert} cells "
            f"per perturbation, columns {pert_col!r}/{ctx_col!r}, control {control!r}"
        )
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))

    processed = Path(cfg["data"]["processed_dir"])
    meta = json.loads((processed / "meta.json").read_text())
    gene_vocab = GeneVocab.from_csv(processed / "gene_vocab.csv")
    genes = list(gene_vocab.names)
    G = meta["n_genes"]

    # the PCA space and the delta fingerprints belong to the MODEL: read from its own folder
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"--checkpoint {ckpt} does not exist")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    trained_cfg = ck.get("config") or cfg
    gene_only = ck.get("kind") == "gene_model"          # a gene.pt: the gene model alone, no PCA at all
    for f in () if gene_only else ("pca.npz", "fingerprints.npz"):
        if not (ckpt.parent / f).is_file():
            raise SystemExit(f"{ckpt.parent / f} is missing: point --checkpoint at a checkpoint of a "
                             f"PCA-space run (train writes {f} beside it)")
    space, fpz, delta_fp, sc = None, None, {}, None
    if not gene_only:
        space = pca_space.Space.load(ckpt.parent / "pca.npz")
        fpz = np.load(ckpt.parent / "fingerprints.npz")
        delta_fp = dict(zip(fpz["pert_names"].astype(str).tolist(), fpz["fp"]))
        sc = {"delta": float(fpz["scale_delta"]), "cov": float(fpz["scale_cov"])}
    pcfg = pca_space.pca_cfg(trained_cfg)
    run_info = ckpt.parent / "run.json"
    trained_perts = None
    if run_info.is_file():
        info = json.loads(run_info.read_text())
        trained_perts = set(info.get("trained_perts") or [])
        print(f"  checkpoint {ckpt}  ({info.get('mode', '?')} run, "
              f"{len(info.get('train_contexts', []))} training context(s), "
              f"epoch {info.get('epochs_run', '?')})")
    pert_index = {
        p: i
        for i, p in enumerate(
            pd.read_csv(processed / "pert_vocab.csv").iloc[:, 0].astype(str)
        )
    }

    want_perts = read_perts(Path(args.perts), pert_col)
    if control in want_perts:
        raise SystemExit(
            f"--perts contains {control!r}; predictions are for perturbations, "
            f"unperturbed cells are the model's input"
        )
    reference_genes = (
        pd.read_csv(args.reference_genes).iloc[:, 0].astype(str).tolist()
        if args.reference_genes
        else None
    )

    for w in preflight(want_perts, genes, meta, reference_genes, manifest.get("n_genes"), trained_perts):
        print(f"  warning: {w}")

    if args.dry_run_perts:
        want_perts = want_perts[: args.dry_run_perts]
        print(f"  DRY RUN: {len(want_perts)} perturbation(s) only -- file is PARTIAL\n")

    missing = [p for p in want_perts if p not in pert_index]
    if missing:
        raise SystemExit(f"{len(missing)} perturbation(s) absent from the pert vocab, e.g. {missing[:3]}")

    # ---- everything that depends only on labels, checked before any compute ----
    contexts = peek_contexts(Path(args.input), pert_col, ctx_col, control)
    n_block = len(want_perts) * per_pert
    obs = pd.DataFrame(
        {
            pert_col: np.tile(np.repeat(want_perts, per_pert), len(contexts)),
            ctx_col: np.repeat(contexts, n_block),
        },
        index=pd.Index([f"cell_{i}" for i in range(len(contexts) * n_block)]),
    )
    expect_labels = dict(
        want_perts=want_perts, pert_col=pert_col, ctx_col=ctx_col, per_pert=per_pert,
        control=control, expect_contexts=expect_contexts, max_cells=args.max_cells,
    )
    check_layout(obs, **expect_labels)
    print(
        f"  plan: {len(contexts)} contexts {contexts} x {len(want_perts)} perturbations x "
        f"{per_pert} cells = {len(obs):,} cells"
    )

    if gene_only:
        _predict_gene_only(ckpt, ck, args, cfg, genes, gene_vocab, G, contexts, want_perts, pert_index, pert_col,
                           ctx_col, control, per_pert, cap, seed, obs, expect_labels, processed)
        return

    device = pick_device(cfg["train"].get("device", "auto"))
    K = space.n_pcs
    from ..models import mean_model as mm

    # the TWO-STAGE model: a mean.pt alone (mean-only), or a centered flow with its mean.pt beside it
    # (model.stage1: mean) or its gene.pt + gene_tables.npz beside it (model.stage1: gene)
    mean_only = ck.get("kind") == mm.KIND
    centered = not mean_only and bool((trained_cfg.get("model") or {}).get("centered"))
    gene_stage = centered and (trained_cfg.get("model") or {}).get("stage1", "mean") == "gene"
    mean_net, mean_cfg_, gene_net, gene_src = None, None, None, None
    if gene_stage:
        from ..models import gene_model as gm

        if not (ckpt.parent / "gene.pt").is_file():
            raise SystemExit(f"{ckpt} is a flow on a gene model but {ckpt.parent / 'gene.pt'} is missing")
        gene_net, _ = gm.load(ckpt.parent / "gene.pt", device)
        gene_src = _gene_sources(ckpt.parent, G, pert_index)
        pert_names_vocab = pd.read_csv(processed / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy()
    if mean_only or (centered and not gene_stage):
        mpath = ckpt if mean_only else ckpt.parent / "mean.pt"
        if not mpath.is_file():
            raise SystemExit(f"{ckpt} is a centered flow but {mpath} is missing (train copies it there)")
        mean_net, mck = mm.load(mpath, device)
        mean_cfg_ = mm.mean_cfg(mck["config"])
        if "generic" not in fpz:
            raise SystemExit(f"{ckpt.parent / 'fingerprints.npz'} has no generic response: re-run the training stage")
    use_state = not mean_only and any(k.startswith("state_enc.") for k in ck["model"])
    state_k = int(pcfg["state_knn"]) if use_state else 0
    use_anchor = not mean_only and any(k.startswith("gain_head.") for k in ck["model"])
    if use_anchor and "generic" not in fpz:
        raise SystemExit(f"{ckpt.parent / 'fingerprints.npz'} has no generic response: an anchored checkpoint "
                         f"needs a run folder written by the current `train`")
    # anchored: gain0 / scale_delta are buffers, restored from the checkpoint by load_state_dict
    model = None
    if not mean_only:
        model = build_model(K, space.pc_sd, trained_cfg["model"], state=use_state,
                            anchor={"gain0": [0.0, 0.0, 0.0], "scale_delta": 1.0} if use_anchor else None).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
    L_t = torch.from_numpy(space.loadings).to(device)
    mu_t = torch.from_numpy(space.mu).to(device)
    other_genes = linear_genes.mode(cfg)
    use_outside = linear_genes.outside_pcs(cfg)
    lin_tab = ckpt.parent / "linear_genes.npz"
    if other_genes == "linear" and not lin_tab.is_file():
        # a run from before model.other_genes existed: fit the linear model on the run's own
        # training lines now (same function training uses) and keep it next to the checkpoint
        from ..data.dataset import load_contexts

        info = json.loads((ckpt.parent / "run.json").read_text())
        print(f"  {lin_tab.name} missing: fitting the linear model on this run's "
              f"{len(info['train_contexts'])} training lines (once; saved next to the checkpoint)")
        _, all_c = load_contexts(processed)
        tr = [c for c in all_c if c.name in set(info["train_contexts"])]
        coef, D_l, _ = linear_genes.fit_for_run(cfg, tr, tr, G, device, say=lambda s: print("  " + s))
        linear_genes.save_for_run(ckpt.parent, tr, D_l, coef, G,
                                  pd.read_csv(processed / "pert_vocab.csv").iloc[:, 0].astype(str).to_numpy())
        del D_l, all_c, tr
    n_delta = sum(p in delta_fp for p in want_perts)
    print(f"  model: {len(space.gene_idx):,} model genes, {K} PCs; delta fingerprints for "
          f"{n_delta}/{len(want_perts)} perturbations; cell state "
          + (f"= mean of {state_k} nearest input cells" if use_state else "off")
          + ("; ANCHORED (v = g_d*D + g_s*S + g_m*M + r)" if use_anchor else "")
          + ("; MEAN MODEL ONLY (every cell + its knockdown's predicted average)" if mean_only else "")
          + ("; TWO-STAGE (gene model average + centered flow)" if gene_stage else
             "; TWO-STAGE (mean model average + centered flow)" if centered else "")
          + f"; non-model genes: {'linear model' if other_genes == 'linear' else 'copied from the source cell'}"
          + ("; + the linear shift OUTSIDE the PCs on the model genes" if use_outside else ""))

    out_path = Path(args.out or Path(ckpt).parent / "predictions.h5ad")
    writer = CountsWriter(
        out_path,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes)),
        uns={
            "signalflow": {
                "checkpoint": str(ckpt),
                "input": str(args.input),
                "n_steps": args.n_steps,
                "seed": seed,
                "partial": bool(args.dry_run_perts),
            }
        },
        n_genes=G,
        cap=cap,
        max_stored=args.max_stored_entries,
    )

    rng = np.random.default_rng(seed)
    t_start = time.time()
    n_done = 0
    for k, (label, counts, var_names) in enumerate(
        iter_contexts(Path(args.input), pert_col, ctx_col, control)
    ):
        if label != contexts[k]:
            writer._fail(f"internal error: context order changed ({label!r} != {contexts[k]!r})")
        known = np.array([g in gene_vocab for g in var_names])
        if not known.any():
            writer._fail(
                f"context {label}: none of its {len(var_names)} genes are in the model's "
                f"vocabulary -- are .var_names gene symbols?"
            )
        if not known.all():
            counts = counts[:, known]
            print(f"  context {label}: dropping {int((~known).sum())} genes not in the model's vocabulary")
        gene_idx = np.array(gene_vocab.indices([g for g, kn in zip(var_names, known) if kn]), dtype=np.int32)
        if len(gene_idx) < G:
            print(
                f"  context {label}: measures {len(gene_idx)}/{G} vocabulary "
                f"genes; the rest are written as zero"
            )

        x_log = _lognorm(counts)
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        mcol = pca_space.model_cols(gene_idx, space.gene_idx, f"context {label}")
        order = pca_space.local_model_order(mcol)
        print(f"  context {label}: {counts.shape[0]:,} unperturbed cells -> {len(want_perts)} perturbations")

        # covariance fingerprint, from THIS line's own control cells (the same function
        # `prepare` runs on the training lines)
        local = {str(genes[g]): j for j, g in enumerate(gene_idx)}
        has = [j for j, p in enumerate(want_perts) if p in local]
        cov_of: dict[int, np.ndarray] = {}
        slope_of: dict[int, np.ndarray] = {}
        if has:
            rows = np.sort(rng.choice(x_log.shape[0], size=min(int(pcfg["cov_cells"]), x_log.shape[0]),
                                      replace=False))
            cov, ok, var = pca_space.cov_fingerprint(x_log[rows], mcol,
                                                     np.array([local[want_perts[j]] for j in has]), space)
            cov_of = {j: cov[k] for k, j in enumerate(has) if ok[k]}
            slope_of = {j: cov[k] / var[k] for k, j in enumerate(has) if ok[k] and var[k] > 1e-8}
        fp = np.stack([pca_space.make_row(K, delta_fp.get(p), cov_of.get(j), sc) for j, p in enumerate(want_perts)])
        print(f"    covariance fingerprints for {len(cov_of)}/{len(want_perts)} perturbations; "
              f"neither fingerprint for {int(((fp[:, K] == 0) & (fp[:, 2 * K + 1] == 0)).sum())}")

        # the mean model's inputs for THIS line, from its own control cells (nothing stored per line)
        mean_hat = None
        if mean_net is not None:
            crow = np.sort(rng.choice(x_log.shape[0], size=min(int(mean_cfg_["ctrl_cells"]), x_log.shape[0]),
                                      replace=False))
            tcols = np.array([local.get(p, -1) for p in want_perts], dtype=np.int64)
            C_line, e_line = mm.control_features(x_log[crow], mcol, space, tcols)
            zero = np.zeros(K, dtype=np.float32)
            inp = {"D": np.stack([delta_fp.get(p, zero) for p in want_perts]).astype(np.float32),
                   "d_ok": np.array([p in delta_fp for p in want_perts], dtype=np.float32),
                   "S": np.stack([slope_of.get(j, zero) for j in range(len(want_perts))]).astype(np.float32),
                   "s_ok": np.array([j in slope_of for j in range(len(want_perts))], dtype=np.float32),
                   "M": fpz["generic"].astype(np.float32), "C": C_line, "e": e_line}
            mean_hat = mm.predict(mean_net, inp, device)
            print(f"    mean model: predicted averages for {len(want_perts)} perturbations "
                  f"(median ||average|| {np.median(np.linalg.norm(mean_hat, axis=1)):.2f} PC units)")

        gene_mu, gene_preds = None, None
        if gene_stage:
            gene_mu, gene_preds = _gene_line_predictions(gene_net, gene_src, label, x_log, gene_idx.astype(np.int64),
                                                         want_perts, pert_index, pert_names_vocab, genes, device, seed)
            print(f"    gene model: shifts for {len(want_perts)} perturbations over {len(gene_idx):,} genes")

        other_of: dict[int, np.ndarray] = {}
        if other_genes == "linear":
            t_l = time.time()
            other_of = linear_genes.input_shifts(lin_tab, x_log, gene_idx, [str(genes[g]) for g in gene_idx],
                                                 want_perts, str(device), seed)
            print(f"    linear model for the {len(gene_idx) - len(order):,} non-model genes  [{time.time() - t_l:.0f}s]")

        # cell state of every input cell: the mean of its k nearest input cells in the PCA
        # space -- the same definition `prepare` used on the training lines' controls
        state_all = None
        if use_state and not mean_only:
            t_s = time.time()
            Z = np.empty((x_log.shape[0], K), dtype=np.float32)
            for i in range(0, x_log.shape[0], 4096):
                Z[i : i + 4096] = space.project(np.asarray(x_log[i : i + 4096][:, order].todense(), dtype=np.float32))
            state_all = pca_space.knn_mean(Z, state_k, str(device))
            del Z
            print(f"    cell state for {x_log.shape[0]:,} cells  [{time.time() - t_s:.0f}s]")

        for j, p in enumerate(want_perts):
            src = rng.choice(counts.shape[0], size=per_pert, replace=per_pert > counts.shape[0])
            x0 = np.asarray(x_log[src].todense(), dtype=np.float32)
            if mean_only:
                dz = np.broadcast_to(mean_hat[j][None, :], (per_pert, K))
            else:
                with torch.no_grad():
                    z0 = (torch.from_numpy(x0[:, order]).to(device) - mu_t) @ L_t.T
                    st = None if state_all is None else torch.from_numpy(state_all[src]).to(device)
                    an = None
                    if use_anchor:
                        a_row = np.concatenate([slope_of.get(j, np.zeros(K, np.float32)), fpz["generic"]]).astype(np.float32)
                        an = torch.from_numpy(a_row[None, :]).to(device).expand(per_pert, -1)
                    z1 = integrate(model, z0, torch.from_numpy(fp[j : j + 1]).to(device).expand(per_pert, -1),
                                   n_steps=args.n_steps, state=st, anchor=an)
                    dz = (z1 - z0).cpu().numpy()
                if gene_stage:
                    dz = dz - dz.mean(0, keepdims=True)
                elif centered:
                    dz = dz - dz.mean(0, keepdims=True) + mean_hat[j][None, :]
            if gene_stage:
                # the gene model moves every gene (all genes, no PCA); the centered flow adds its scatter
                local_counts = gm.transform_counts(x0, gene_preds[j][:, 0], None, gene_mu, lib[src], cap=cap,
                                                   dz=dz, order=order, loadings=space.loadings)
                block = sp.csr_matrix((local_counts.data, gene_idx[local_counts.indices], local_counts.indptr),
                                      shape=(per_pert, G))
                block.sort_indices()
                writer.append(block)
                n_done += 1
                continue
            outside = None
            if use_outside and other_of.get(j) is not None:
                # the linear shift's part OUTSIDE the PCs, on the model genes (model.outside_pcs)
                outside = linear_genes.outside_part(other_of[j][order], space.loadings)
            local_counts = residual_counts(x0, order, dz, space.loadings, lib[src], cap=cap,
                                           other_shift=other_of.get(j), shift=outside)
            block = sp.csr_matrix((local_counts.data, gene_idx[local_counts.indices], local_counts.indptr),
                                  shape=(per_pert, G))
            block.sort_indices()
            writer.append(block)
            n_done += 1
            if (j + 1) % PROGRESS_EVERY == 0 or j + 1 == len(want_perts):
                total = len(contexts) * len(want_perts)
                rate = (time.time() - t_start) / n_done
                print(
                    f"    {label}: {j + 1}/{len(want_perts)} perturbations  "
                    f"[{n_done}/{total} overall, ~{rate * (total - n_done) / 60:.0f} min left, "
                    f"{writer.nnz:,} stored entries]"
                )

    writer.finish(genes, pert_col, ctx_col, expect_labels)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    if args.dry_run_perts:
        print("  PARTIAL file (--dry-run-perts): the portal would reject it")
    else:
        print("  next: python -m signalflow.submission.submit_vcc26 package --pred", out_path, "(packages it; uploads nothing)")


if __name__ == "__main__":
    main()
