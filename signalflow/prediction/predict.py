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

THE PCA BASIS COMES FROM THE CHECKPOINT'S OWN FOLDER
    Each training run fits its own shared PCA basis (on that run's training lines
    only) and saves `pca_shared.npz` beside its checkpoints, so the basis always
    travels with the model it was fit for. `--checkpoint` is therefore required,
    and the basis is read from its folder -- never from the processed directory,
    whose contents may since have changed.

NO CONTEXT MAPPING
    The model has no notion of "which cell line" at all -- see
    `models/encoders.py` and `data/shared_pca.py` -- so an input needs no
    relationship to any context seen in training. Each cell line supplies its
    own cell state, via the frozen shared PCA basis, and its own gene mask,
    from which vocabulary genes it actually measures.

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
    The model predicts lognorm (CPM + log1p). `rint(expm1(x) * lib / 1e6)`
    inverts it, with `lib` the library size of the control cell the trajectory
    started from -- the same choice `flow.to_counts` makes, and the same
    caveat: a perturbation that shifts sequencing depth is not modelled. Cells
    are then clipped to the per-cell cap and rounded, which is what makes them
    whole numbers rather than a float matrix that looks like counts.

WHAT THIS STILL CANNOT FIX
    A one-hot `PertEncoder` has one row per perturbation, trained only from
    cells carrying it. A perturbation absent from training keeps its random
    init, so the model cannot generalise to it -- not badly, at all (README
    §6). `preflight()` counts these and WARNS -- it does not stop the run: a
    file that passes every rule above and contains noise for that reason is the
    expensive failure here, so read that warning. The cell-line side of this
    problem is gone (that was `ContextEncoder`); the perturbation side is still
    open.

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

from ..data import shared_pca
from ..data.vocab import CONTROL_LABEL, GeneVocab
from ..models.build import build_model, pick_device
from ..models.flow import integrate, mask_from_gene_idx
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


def _cell_state(
    x_log: sp.csr_matrix,
    counts: sp.csr_matrix,
    gene_idx: np.ndarray,
    loadings: np.ndarray,
    mu: np.ndarray,
):
    """The conditioning vector, projected into the SAME shared space training
    data uses -- `shared_pca.project()` against the frozen basis, not a fresh
    fit. This is what makes a cell line absent from training embeddable at
    all: see `data/shared_pca.py` and `models/encoders.py`.

    Chunked, matching `data/prepare.py::_cell_state` -- the two must stay in
    sync, or these cells' state would land in a different space than the
    training data's does.
    """
    n = x_log.shape[0]
    n_pcs = loadings.shape[0]
    pcs = np.empty((n, n_pcs), dtype=np.float32)
    mean_log = np.empty(n, dtype=np.float32)
    for i in range(0, n, CHUNK):
        block = np.asarray(x_log[i : i + CHUNK].todense(), dtype=np.float32)
        pcs[i : i + CHUNK] = shared_pca.project(gene_idx, block, loadings, mu)
        mean_log[i : i + CHUNK] = block.mean(axis=1)

    lib = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    n_det = np.diff(counts.tocsr().indptr).astype(np.float32)
    scalars = np.stack([np.log1p(lib), np.log1p(n_det), mean_log], axis=1)
    state = np.concatenate([pcs, scalars], axis=1).astype(np.float32)

    # z-scored against THIS context's own cells -- they are all unperturbed,
    # so unlike prepare.py there is no separate "control subset" to restrict to.
    mu_s, sd_s = state.mean(axis=0), state.std(axis=0)
    sd_s[sd_s < 1e-6] = 1.0
    return ((state - mu_s) / sd_s).astype(np.float32), lib


def _to_counts(x_lognorm: np.ndarray, lib: np.ndarray, cap: float) -> sp.csr_matrix:
    """Predicted lognorm -> whole, non-negative counts under the per-cell cap."""
    counts = np.rint(np.expm1(x_lognorm.astype(np.float64)) * lib[:, None] / 1e6)
    np.clip(counts, 0.0, None, out=counts)
    total = counts.sum(axis=1)
    hot = total > cap
    if hot.any():
        # Rescale then re-round, so the cap holds on the integers actually written.
        counts[hot] = np.rint(counts[hot] * (cap / total[hot])[:, None])
    out = sp.csr_matrix(counts.astype(np.float32))
    out.eliminate_zeros()
    return out


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
            f"(e.g. {', '.join(unseen[:4])}). A one-hot PertEncoder leaves these "
            f"at their random init -- the model cannot generalise to them at all"
        )
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="the .pt to predict with. Its run folder must also hold the "
                         "pca_shared.npz that model was trained with (train writes both)")
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

    # The PCA basis belongs to the MODEL, not to the processed data: each run fits its
    # own (on that run's training lines only) and saves it beside the checkpoint. Using
    # another run's basis would feed the model cell states from a different space --
    # silently, and with no error -- so it is read from the checkpoint's own folder.
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"--checkpoint {ckpt} does not exist")
    basis = ckpt.parent / "pca_shared.npz"
    if not basis.is_file():
        raise SystemExit(
            f"{basis} is missing: {ckpt.name} has no PCA basis beside it. `train` writes "
            f"pca_shared.npz into every run folder -- point --checkpoint at a checkpoint "
            f"from such a run (a pre-{'2026-09'} run kept the basis in the processed dir)."
        )
    loadings, pca_mu = shared_pca.load(basis)
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

    device = pick_device(cfg["train"].get("device", "auto"))
    # the state width follows the basis this checkpoint was trained with
    n_state = int(loadings.shape[0]) + int(meta.get("n_scalars", 3))
    model = build_model(meta["n_genes"], meta["n_perts"], n_state, cfg["model"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

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
                f"genes; the rest are forced to zero in both the model's "
                f"input and its output"
            )

        x_log = _lognorm(counts)
        state, lib = _cell_state(x_log, counts, gene_idx, loadings, pca_mu)
        print(f"  context {label}: {counts.shape[0]:,} unperturbed cells -> {len(want_perts)} perturbations")

        for j, p in enumerate(want_perts):
            src = rng.choice(counts.shape[0], size=per_pert, replace=per_pert > counts.shape[0])
            x0_local = np.asarray(x_log[src].todense(), dtype=np.float32)
            x0_g = np.zeros((per_pert, G), dtype=np.float32)
            x0_g[:, gene_idx] = x0_local
            with torch.no_grad():
                pred = integrate(
                    model,
                    torch.from_numpy(x0_g).to(device),
                    torch.full((per_pert,), pert_index[p], dtype=torch.long, device=device),
                    torch.from_numpy(state[src]).to(device),
                    mask_from_gene_idx(gene_idx, G, per_pert).to(device),
                    n_steps=args.n_steps,
                ).cpu().numpy()
            writer.append(_to_counts(pred, lib[src], cap))
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
