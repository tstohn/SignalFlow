# SignalFlow v0 — conditional flow matching for perturbation response

Predicts what single cells look like after a CRISPR knockout, by learning a
**velocity field** that transports control cells onto perturbed cells.

Deliberately the simplest thing that is still the *right shape*: every piece
that will need to get smarter is an isolated, swappable class with its upgrade
path written next to it.

```bash
cd /Users/timstohn/Desktop/SignalFlow

python -m signalflow.data.prepare --config configs/prototype.yaml   # ~40 s
python -m signalflow.training.train        --config configs/prototype.yaml   # ~14 s/epoch (MPS)
python -m signalflow.evaluation.evaluate     --config configs/prototype.yaml --split test
python -m signalflow.evaluation.evaluate     --config configs/prototype.yaml --split test --vcc
```

The `Makefile` wraps these: `make training` (prepare → train → evaluate), `make prediction`,
and `make submission` (package + upload in one call). `make -n <target>` previews without running.

Everything is driven by `configs/prototype.yaml`. Run from the repo root
(`signalflow/` is not installed as a package).

---

## 1. The idea in six lines

```
x0  ~ control cells of a context           # independent coupling — no pairing
x1  ~ perturbed cells, same context, target gene p
t   ~ U(0,1)
x_t = (1-t)·x0 + t·x1                      # straight-line interpolant
u   = x1 - x0                              # its velocity
loss = masked MSE( v_θ(x_t, t | p, s(x0), mask),  u )
```

**Inference:** start at a *real control cell*, Euler-integrate `t: 0→1`. The
endpoint is a predicted perturbed cell in lognorm space.

So the model's **output is a velocity field**; the **deliverable is cells**.

### Why the loss is on the velocity, not on `x1`

Regressing `x1` directly gives you the conditional *mean* — one point per
perturbation, cell-to-cell heterogeneity gone. Under the flow-matching loss
the optimum is `v*(x_t,t,c) = E[x1 - x0 | x_t, c]`, and integrating that field
transports the **whole** control distribution onto the **whole** perturbed
distribution. The spread comes out for free — which is what the VCC-style
distributional metrics actually score.

It also makes the direction × magnitude factorisation a config flag:
`model.head: dirmag` gives `v = softplus(m)·normalize(d)`.

### How other methods do it

| Method | Output | Loss |
|---|---|---|
| GEARS, scGPT, CPA, biolord | post-pert expression (usually δ from control mean) | MSE, + MSE on top-DE genes |
| scGen | latent shift, then decode | VAE ELBO + latent arithmetic |
| CellOT | OT map control→perturbed | ICNN dual OT objective |
| CellFlow / OT-CFM family | velocity field, ODE integrate | flow-matching MSE on velocity |
| STATE (Arc, VCC25) | perturbed *set* from control *set* | MSE + distributional (energy/MMD) |

The first row collapses each perturbation to a point. That is the thing being
avoided here.

---

## 2. Two index spaces, kept apart (`vocab.py`)

| | Space | Size | Source |
|---|---|---|---|
| `GeneVocab` | **readout** — what we predict expression for | 18,533 | `data/VCC26/controls/gene_names.csv` |
| `PertVocab` | **perturbation** — what can be knocked out, index 0 = `non-targeting` | 18,534 | same file, + the control slot |

Same csv, two different roles. They must stay separate: `selected_genes.csv`
is the *readout* panel (300 HVG + 200 random per dataset, 1,213 unique), and
only **4 of the 80** prototype perturbations appear in it. The perturbation
vocabulary has to be the full symbol list or most knockouts are unrepresentable.

Swapping either later = point `data.gene_vocab_csv` / `data.pert_vocab_csv` at
a different csv. A one-column file, or one with a `gene_name`/`gene`/`symbol`
column, both work (`vocab.py:30`).

---

## 3. Preprocessing (`data/prepare.py`)

`.h5ad` → one compact `.npz` per context. Reads `.layers["lognorm"]`
(CPM + log1p) as the model space and `.X` (raw UMIs) for library size.

**Nothing is zero-padded on disk.** Each context keeps its own compact matrix
(630–1,147 columns) plus `gene_idx`, the map from local column → global
readout index. The scatter into the 18,533-wide space happens per batch. That
is what keeps this workable as the panel grows.

### Cell state — one shared basis, not one per context (`shared_pca.py`, `prepare.py:63`)

35 dims = **32 PCs + 3 scalars**:

- The PCs come from **one PCA basis fit across every context's control
  cells**, not a fresh fit per context. Fitting alternates two least-squares
  solves (EM/ALS) over only the (cell, gene) pairs a context actually
  measured — the same masking discipline as the flow-matching loss (§4),
  applied to fitting a PCA instead of a velocity field. See `shared_pca.py`.
- **Why shared, not per-context:** a per-context basis has no row for a
  context absent at prep time — exactly the failure mode that made a
  per-context `ContextEncoder` (removed, see §6) unusable on a held-out cell
  line. The shared basis is frozen after `prepare.py` and reused unmodified:
  embedding a NEW cell line is one small least-squares solve against it
  (`shared_pca.project()`), needing no access to the training data.
- scalars: `log1p(total UMI)`, `log1p(genes detected)`, `mean lognorm`
- z-scored against **that context's own** control-cell statistics — this step
  stays per-context; it's derivable from any dataset's own controls and
  carries no training-time identity
- `pca_shared.npz` (top-level, ONE file) holds `loadings`/`mu`; per-context
  `state_mu`/`state_sd` no longer exist as separate files, they're folded into
  `state` at prep time the same way they always were

### Splits (`prepare.py:93`)

80/10/10 **on cells**, stratified by perturbation. *Not* on perturbations —
see §6.

### What lands on disk

```
data/processed/prototype/
  meta.json            vocab paths, per-context summary, prep config
  gene_vocab.csv       readout space   (row order = index)
  pert_vocab.csv       perturbation space, row 0 = "non-targeting"
  splits.json          train/val/test CELL indices, per context
  pca_shared.npz        ONE shared PCA basis: loadings, mu
  contexts/<name>.npz
      X_data/X_indices/X_indptr/X_shape   lognorm expression, CSR
      gene_idx      int32 [n_local]       local column -> global gene index
      pert          int32 [n_cells]       -> pert vocab (0 = control)
      is_control    bool  [n_cells]
      state         f32   [n_cells, 35]   cell-state conditioning
      lib           f32   [n_cells]       total UMI (from raw .X)
      control_rows  int32 [n_control]
```

8 contexts, 31,210 cells, 35 MB.

---

## 4. Masking — the part that makes heterogeneous panels work

An unmeasured gene is **not** a gene measured as zero, and the model is told
which is which. `FlowDataset.collate` scatters each batch's genes into the
global space and builds a `mask [B, n_genes]` tensor alongside it
(`dataset.py`), passed to the model as an **argument**, every call — never
looked up from a trained context id (there is no such lookup; see §6). It
enters in **three** places:

| Where | Code | Why |
|---|---|---|
| input | `torch.cat([x_t * m, m], -1)` — `velocity.py` | the model sees the panel explicitly, so zero-because-unmeasured ≠ zero-because-silent |
| output | `return v * m` — `velocity.py` | the field never moves in directions the data cannot speak to |
| loss | `((v-u)²·m).sum() / m.sum()` — `flow.py` | **never `mean()` over G** — contexts measure different gene counts, and an unmasked mean makes their losses incomparable |

`evaluate.py`, `cell_eval.py` and `predict.py` all build this same mask directly
from a context's `gene_idx` (`flow.mask_from_gene_idx`) rather than from any
identity lookup — which is what makes them work on data whose context never
appeared during training.

---

## 5. Batching (`data/dataset.py`)

A training item is one **perturbed cell** `x1`. Its partner `x0` is a control
cell drawn at random from the same context — independent coupling.

Two things that are easy to get wrong and are handled explicitly:

- **Conditioning always comes from the source cell `x0`** (`dataset.py:135`).
  `x1`'s state must never leak in; at inference it does not exist.
- **Control cells appear as targets too**, with pert index 0. The model learns
  that `non-targeting` means near-zero net displacement, with the real
  control-to-control spread still in it. Without that anchor nothing pins the
  magnitude scale.

`ContextBatchSampler` (`dataset.py:143`) draws each batch from a single
context — cheap scatter, one shared mask — and shuffles context order every
epoch so the gradient doesn't walk through datasets in blocks. `ctx` still
rides along in every batch dict, but **only** as bookkeeping for per-source
reporting in `evaluate.py`/`cell_eval.py` — the model itself never receives it.

A batch is `{x0, x1, pert, mask, state, lib0, ctx}`.

---

## 6. Model (`models/velocity.py`, `models/encoders.py`)

```
                 pert  ──► PertEncoder    (nn.Embedding, 64)   ─┐
                 state ──► StateEncoder   (2-layer MLP, 64)     ├─► cond (160)
                 t     ──► TimeEncoder    (Fourier, 32)        ─┘
                                                                 │  (additive)
  [x_t·m, m] ──► Linear(2G→512) ──► 3× pre-norm ResBlock ──► LayerNorm ──► head ──► ·m
```

31.5M params at G=18,533. Output layer is zero-initialised, so the model
starts as the identity map (predict no change) and has to earn every
deviation.

**There is no cell-line encoder.** A learned lookup keyed by "which
dataset/context" cannot represent one absent at training time — the same
generalisation gap `PertEncoder` has for perturbations (below), for cell
lines instead. `ContextEncoder` was removed for exactly that reason; cell-line
identity is now entirely subsumed by `StateEncoder`'s input, which comes from
the **shared** PCA basis (§3), reusable on a context the model never trained
on. What genes a cell's context measures still matters, but that's the `mask`
argument — a property of the data, not a trained embedding of an identity.

**`PertEncoder` is one-hot.** `nn.Embedding(V, d)` is exactly `one_hot(p) @ W`,
without materialising the V-wide row. The control slot is initialised to
exactly zero.

> **The limit, stated plainly.** A one-hot encoder has one free row per
> perturbation, learned only from cells carrying it. A perturbation never seen
> in training keeps its random init — so this model cannot generalise to
> unseen perturbations. Not badly: *at all*. In the prototype data **no
> perturbation is shared between any two of the eight files** (10 each, 80
> unique, zero overlap), so a held-out-perturbation split would score pure
> noise. That is why splits are on cells.

Heads (`velocity.py:41`): `plain` (free vector, default) or `dirmag`
(`v = softplus(m)·normalize(d)`).

---

## 7. Evaluation (`evaluate.py`)

Three methods, always compared:

| | |
|---|---|
| `identity` | predict no change, `x1_hat = x0`. The floor. |
| `mean_shift` | `x1_hat = x0 + ` mean δ of that (context, pert) on the **train** split. Deliberately strong — it is handed the answer's first moment. |
| `flow` | Euler-integrate the learned field from `x0`. |

Three metrics, per (context, perturbation), over that context's panel genes:

- **`delta_r`** — Pearson r between predicted and true mean shift from the
  control mean. **The headline number.** Plain expression correlation sits
  near 1.0 for everything including `identity`, and tells you nothing.
- **`mae`** — mean |predicted mean − true mean| per gene.
- **`energy`** — energy distance between predicted and true cell clouds,
  `2E|X−Y| − E|X−X'| − E|Y−Y'|`. Zero iff the distributions match. This is the
  one that punishes collapsing to a point, and the reason for a flow rather
  than a regressor.

`mean_shift` is in the harness because most published gains in this field
evaporate against it.

### How the data is split, and what each part is used for

`prepare.py` splits **cells**, 80 / 10 / 10, stratified per (context,
perturbation) — so every perturbation appears in all three parts (a one-hot
`PertEncoder` cannot handle a held-out perturbation, §6). Every group keeps at
least one training cell.

| split | used by | for |
|---|---|---|
| `train` | `train.py` | the gradient updates |
| `val` | `train.py`, each epoch | val loss (which picks `best.pt`) and, if `train.cell_eval_every` > 0, the cell-eval2 VCC26 metrics. Its source control cells come from a held-out val pool too, so the flow never starts from cells it memorised |
| `test` | `evaluate.py` | one final look, after training |

One caveat: the shared PCA basis and each context's state z-scoring are fit on
the control cells of *all three* splits. It is unsupervised and controls only,
but val/test controls do influence the basis.

### Cell-eval metrics during training (`train.cell_eval_every`)

Two output channels. The **terminal** gets the summary: one line per epoch (train
loss, val loss, variance explained, time) and, every `train.cell_eval_every`
epochs *and at the last epoch*, a table of the six cell-eval2 VCC26 members with
ONE average. `0` turns cell-eval off. The **log file** `<out_dir>/train.log` gets
all of that plus what is kept off the terminal: per-context cell-eval detail and
cell-eval2's own warnings, each tagged with the epoch, method and context.

```
epoch   5  train 7.9032  val 7.7602  (identity 8.2235, explained   5.6%)  11.4s
  cell-eval (val) after epoch 5   [18s]
                       raw   scaled  contexts
    pds_cosine       0.536   +0.072      8/8
    expr_mse         1.424   -0.424      4/8
    dir_fidelity     0.096   -0.808      8/8
    dir_reach        0.353   +0.353      8/8
    sig_jaccard      0.472   +0.472      8/8
    lfc_nmae         1.137   -0.137      6/8
    AVERAGE                  -0.032     (mean_shift +0.328)  best -0.032 at epoch 5
```

`raw` is each metric as cell-eval2 defines it (lower is better for `expr_mse` and
`lfc_nmae`); `scaled` puts all six on one axis, 0 = no skill and 1 = perfect;
`mean_shift` is the number to beat. `contexts` is how many of the val contexts a
row averages over: a metric is missing in a context where the real data cannot
support it (too few cells), which is also why AVERAGE — the mean over contexts of
each context's average over the members it has — is not the plain mean of the six
rows. At the start the two baselines are printed once, per member, and at the end
the average by epoch is listed. Everything also lands in `history.json` as `vcc_*`
keys. The reference (real cells, source cells, both baselines) is built once, so
each scoring recomputes only the flow's predictions (~15–25 s here).

The repeated notices cell-eval2 prints, such as "de_lfc_nmae: omitted N
perturbation(s) for an empty gate", are ordinary Python `logging` records: they
mean the metric had nothing to score for those perturbations because too few real
genes were significant (the val split has few cells per perturbation). They go to
`train.log` (and, for `evaluate --vcc`, to `evaluate_vcc_<split>.log` beside the
results) instead of the terminal. **cell-eval2 itself is never modified**: a
handler is attached to its logger from our side (`evaluation/cell_eval.captured`)
only while a scoring call runs, and removed afterwards.

**Stopping, and what is saved.** Ctrl-C is safe: `train.py` writes after *every*
epoch, so an interrupt costs at most the epoch in progress.

| file | holds |
|---|---|
| `best.pt` | lowest val **loss** so far |
| `best_vcc.pt` | highest cell-eval **avg score** so far (only when cell-eval is on) |
| `last.pt` | the most recent completed epoch |
| `history.json` | every epoch's numbers |
| `train.log` | everything printed, plus the cell-eval detail and notices kept off the terminal (appended to on each run) |

The two "best" checkpoints need not agree — the loss is a proxy, the cell-eval
score is what the challenge measures — so pass `--checkpoint runs/.../best_vcc.pt`
to `predict.py` / `evaluate.py` to use the latter.

**Plateau stopping** (`training/stopping.py`). The cell-eval score is noisy, so
"stop when it has not set a new maximum" is a poor rule: one lucky epoch sets a bar
that may never be cleared again, and a wobble upward of 0.001 resets the clock.
`train.early_stop_*` adds three things, all off by default (`early_stop_patience: 0`):

| key | meaning |
|---|---|
| `early_stop_window` | average the last W measurements before comparing, so one lucky or unlucky scoring decides nothing |
| `early_stop_min_delta` | progress means the smoothed score beats the best by at least this (0..1 scale; loss units if cell-eval is off) |
| `early_stop_patience` | N measurements in a row without progress, then stop |

A measurement is one scoring (every `cell_eval_every` epochs) or, with cell-eval off,
one epoch's val loss. A reasonable start with `cell_eval_every: 5` is patience 3,
window 3, min_delta 0.005. The table shows a `plateau watch k/N` counter so you can
see how close it is. A window dilutes a one-off lucky spike but does not remove it;
a wider window or a larger patience reduces that further. `best_vcc.pt` still follows
the *raw* score (the peak), while the stop decision uses the *smoothed* one. The
cosine LR schedule is sized for `epochs`, so stopping early leaves the learning rate
un-annealed.

### `--vcc` — the VCC2026 competition suite (`cell_eval.py`)

The same three methods, scored on the six members the 2026 Virtual Cell
Challenge scores, by `cell-eval2` itself. No metric is re-implemented here:
this side builds the two AnnData objects the competition defines and the
library does the arithmetic.

| member | is | good |
|---|---|---|
| `pds_cosine` | is a predicted profile nearest its own measured one | high, 0.5 = no information |
| `expr_mse_unbiased_capped_norm` | expression error, sampling noise removed | low, 1.0 = emit the control |
| `de_wilcoxon_direction_fidelity_yield_raw` | are called genes moved the right way | high, 0.5 = random signs |
| `de_wilcoxon_direction_reach_raw` | how deep the ranking stays 90 % correct | high |
| `de_wilcoxon_sig_jaccard` | do the responding-gene sets agree | high |
| `de_wilcoxon_lfc_nmae` | fold-change size | low, 1.0 = predict no change |

Two things this side has to get right, both written up in `cell_eval.py`:

- **Counts, not lognorm.** The competition scores raw counts, and the group-sum
  profile behind `pds_cosine` / `expr_mse_*` cannot be formed from lognorm --
  declaring `input_type="lognorm"` silently swaps in a different profile.
  `prepare.py` stores CPM+log1p *and* `lib`, so `rint(expm1(x)·lib/1e6)` gives
  the original integers back to within 8e-4. Predicted cells reuse their source
  cell's depth, with the same caveat as `flow.to_counts`.
- **The full 18,533-gene space, not the context's panel.** Five members exclude
  the perturbation's own target gene, and `cell-eval2` refuses to score when no
  target resolves to a feature. Four of the eight contexts have *zero* target
  genes in their own panel, so panel-space scoring fails on half the data.
  Unmeasured genes are structurally zero on both sides, so they change nothing.

**Scaling.** Official scoring is `s = (u - b)/(r - b)` against reference bundles
measured on the competition's own contexts; those constants do not describe this
data. `--vcc` uses the scale `cell-eval2` ships instead, `0` = no skill and
`1` = perfect, so the six land on one axis. Raw values print alongside and are
the ones to trust.

### Predicting on new data, and submitting it (`predict.py` → `submit_vcc26.py`)

`--vcc` scores in memory against ground truth you already have. Data with *no*
known answer — the real VCC26 contexts — goes through two scripts instead.

```
predict.py ──► predictions.h5ad ──(vcc prep)──► prediction.vcc ──(vcc submit)──► portal
              raw counts, sparse            validates + packages            uploads + scores
```

```bash
# 1. predict: unperturbed cells + a perturbation list -> predicted raw counts
python -m signalflow.prediction.predict --config configs/prototype.yaml \
    --input data/VCC26/controls \
    --manifest data/VCC26/controls/manifest.json \
    --perts data/VCC26/controls/pert_counts.csv \
    --reference-genes data/VCC26/controls/gene_names.csv \
    --out runs/prototype/predictions.h5ad

# 2. package: predictions.h5ad -> prediction.vcc   (local; sends NOTHING)
python -m signalflow.submission.submit_vcc26 package --pred runs/prototype/predictions.h5ad

# 3. upload: prediction.vcc -> the portal            (run by you; asks first)
python -m signalflow.submission.submit_vcc26 upload --file runs/prototype/predictions.vcc -m "my model v1" --wait

# ...or steps 2 and 3 together, in one call
python -m signalflow.submission.submit_vcc26 submit --pred runs/prototype/predictions.h5ad -m "my model v1" --wait
```

**How the portal wants data.** One `.vcc` file covering all three contexts,
with the labels A/B/C reused exactly as in the control files (D/E/F in the final
phase, never carried over). It holds the 300 perturbations from
`pert_counts.csv` × exactly 400 cells × 3 contexts = 360,000 cells, over all
18,533 genes of `gene_names.csv`; raw whole-number counts, non-negative and
finite, ≤ 1e6 per cell; **no** `non-targeting` rows; ≤ 400,000 cells and ≤
4.75e9 stored entries in total (a density limit — store it sparse, never
dense). You never build a `.vcc` by hand: `vcc prep` turns the `.h5ad` into one.

**`predict.py`** takes `--input` (one `.h5ad` or a folder; raw counts, gene
symbols) and `--perts` (a csv with a header). `--manifest` is optional: it
supplies `cells_per_pert`, `pert_col`, `context_col` and `control_label` as
defaults (a flag you pass still wins), checks `n_genes`, and requires the input
to provide exactly its `contexts` — checked from `.obs` alone before any
prediction starts. If `.obs["target_gene"]` exists only its `non-targeting` rows
are used as starting cells. Contexts are labelled by `.obs["context"]` if
present, else by file name. There is no context mapping to supply: the model has
no notion of "which cell line" (§6), so an input needs no relationship to
anything seen in training.

All label rules — contexts, the exact perturbation set, 400 cells each, gene
order, no controls, the 400,000-cell cap — are checked **before any prediction
runs**, from the arguments alone. The output is written **one perturbation block
at a time straight into the file**, so memory does not grow with its size (a full
panel is ~2 billion stored entries, ~17 GB if held in RAM). Each block is checked
as it is written (whole, finite, non-negative counts; the per-cell cap; no
explicit zeros; the density cap), the file is written to `<out>.partial`, and it
is only moved into place after being re-read from disk. A failed run never leaves
a plausible-looking file at `--out`.

**Gene vocabularies must match, with no override.** If the model's vocabulary
disagrees with the manifest's `n_genes`, or with `--reference-genes` in count or
order, `predict.py` stops with an error; there is no flag to write anyway. An
*untrained perturbation* (which keeps its random init, §6) is different: the file
is well-formed, so it is a loud **warning**, not a refusal. Read it: with the
current checkpoint it says 298 of 300 perturbations were never trained, and the
file would pass every format check while being mostly noise. `--dry-run-perts N`
writes a deliberately partial file for testing, which `submit_vcc26.py` refuses.

**`submit_vcc26.py`** drives the `vcc` command-line tool (installed as a
`uv tool`, so it is found next to this Python, on `PATH`, or in `~/.local/bin`;
`--vcc-bin` overrides). It has three commands, and each flag belongs to exactly one:

| command | does | flags |
|---|---|---|
| `package` | `predictions.h5ad` → `.vcc`, on this machine. **Sends nothing.** | `--pred` (input), `--out`, `--controls`, `--dry-run`, `--ignore-memory`, `--force` |
| `upload` | sends an *existing* `.vcc` to the portal. Never packages. | `--file` (the `.vcc`), `-m`, `-d`, `--wait`, `--yes` |
| `submit` | `package` then `upload`, in one call | both sets of flags |

`--pred` is only ever the input `.h5ad`; it does not itself create or send
anything. They are separate commands because they cost different things:
packaging is the slow, RAM-hungry, free and repeatable step (~22 GiB for a full
file), while uploading is quick but spends a daily submission. Split, you package
once, inspect, and upload later (or from another machine) without packaging
twice; `submit` is for when you don't need that. `upload` and `submit` send data,
so they ask you to type `submit` first (`--yes` skips it), and they are meant to be
run by you, never by a script or an agent. Log in first with `vcc login` (or
`export VCC_TOKEN=...`).

Its pre-flight matters because **`vcc prep` loads the whole `.h5ad` into RAM
before checking anything** and only warns about memory afterwards; on too small a
machine it is killed by the OS with no message. So before prep runs, the script
opens only the labels and the matrix header and checks the layout, the
stored-entry count, and whether packaging fits in this machine's RAM. Packaging
needs roughly `1.3 × 8 bytes × stored entries + 2 GiB` for the float32,
in-gene-order file `predict.py` writes: a full panel at ~5,800 entries per cell
(2.1 billion) needs **~22 GiB**, more than a 16 GB machine has. Run it on a
larger machine (the `.h5ad` copies as-is) or have the model predict fewer
expressed genes per cell.

**On this data the suite is thin.** VCC2026 assumes 300 perturbations × 400
cells per context; the prototype test split has 5–24 cells across ≤10
perturbations. `expr_mse` needs the *reference's* effect to clear its own
sampling-noise correction and often cannot at that depth; when a member is
unavailable it is dropped for every method at once, and the run says which and
how many of the six remain.

---

## 8. Where it stands

**Smoke run only — 2 epochs, not a trained model.** Test split, 70
perturbations, 8 contexts:

| method | delta_r (mean) | MAE | energy |
|---|---|---|---|
| identity (no change) | 0.036 | 0.733 | 16.87 |
| **flow (2 epochs)** | **0.064** | 0.731 | 16.77 |
| mean_shift baseline | 0.267 | 0.710 | 16.18 |

For reference, an earlier **40-epoch** run on the smaller 1,213-gene readout
space reached `delta_r = 0.122` — still below `mean_shift`. Treat both as
"the pipeline runs end to end", not as model quality. Nothing here has been
trained to convergence.

Two symptoms already visible and worth watching:

- **Controls move too much** (energy 1.97 vs 1.31 for identity) — the
  `non-targeting = no motion` anchor is not being fully respected.
- **`frac_var_explained` plateaus around 4–9%** — most of `x1 − x0` is
  irreducible cell-to-cell noise, so this is not directly alarming, but the
  systematic signal is thin at this data scale.

Also note: with the full 18,533-gene readout space, **17,320 output slots are
measured by no dataset in the prototype**, so those rows never receive
gradient (~8.9M of 31.6M params are dead). Harmless, and the price of
checkpoint compatibility when richer datasets arrive. Set
`gene_vocab_csv: null` to fall back to the union of the per-file panels
(1,213 genes, 4.9M params, ~2.6 s/epoch) when iterating fast.

---

## 9. How to continue — highest leverage first

1. **`PertEncoder` → feature-based.** The only remaining change that makes
   unseen perturbations possible *at all* (unseen cell lines are handled — see
   §6, §9 item 4 below, done). Project features of the knocked-out gene: its
   own expression profile across control cells, an OmniPath/STRING network
   embedding, a DepMap essentiality vector. Same output shape — nothing
   downstream changes. Highest leverage remaining; everything else is refinement.
2. **Add a distributional term to the loss** (energy / MMD on integrated
   endpoints, as STATE does). The current loss is pointwise; the metric is not.
3. **OT coupling instead of independent coupling** — pair `x0`/`x1` by
   minibatch optimal transport. Straighter paths, fewer integration steps,
   lower gradient variance.
4. ~~`StateEncoder` → shared across contexts~~ **Done.** `ContextEncoder` is
   removed; cell state comes from one PCA basis fit across every context and
   reusable on one that never existed at fit time (`shared_pca.py`). The
   EM/ALS fit is currently plain numpy — fine at this scale (18,533 genes,
   tens of thousands of cells), but worth profiling once real data brings
   many more cells.
5. **`head: dirmag`** — direction × magnitude, already wired, one config flag.
6. **Library-size modelling.** `flow.to_counts` reuses the source cell's depth;
   perturbations shift it.

Then scale the data: the prototype is 29k cells over 8 files, which is thin
for 80 perturbations.

---

## 10. Layout — where things live

One folder per stage. Each holds the script you run plus its helpers; nothing
imports a runnable script, and code shared between stages lives in `data/` or
`models/`. File names are unique across the package, so the rest of this document
refers to them by their short name.

| Folder | File | Lines | What it holds |
|---|---|---|---|
| `data/` | `vocab.py` | 106 | the two index spaces |
| | `shared_pca.py` | 130 | ONE masked/EM PCA basis, fit across contexts; frozen, reusable on new ones |
| | `dataset.py` | 197 | pairing, gene scatter, per-batch mask, context sampler |
| | `prepare.py` | 291 | **run:** h5ad → per-context npz, shared PCA fit + project, splits |
| `models/` | `encoders.py` | 106 | pert / state / time encoders + upgrade notes (no context encoder) |
| | `velocity.py` | 128 | the field, both heads; mask is an argument, not a lookup |
| | `flow.py` | 122 | CFM loss, Euler sampler, lognorm→counts, mask-from-gene_idx helper |
| | `build.py` | 37 | `build_model`, `pick_device` — shared by training, evaluation and prediction |
| `training/` | `train.py` | 279 | **run:** loop, AdamW + cosine, per-epoch cell-eval, checkpoints, early stop |
| | `stopping.py` | 55 | plateau rule: smoothing + minimum improvement + patience |
| `evaluation/` | `evaluate.py` | 182 | **run:** metrics + `identity` / `mean_shift` baselines, `--vcc` |
| | `cell_eval.py` | 385 | the six VCC2026 members via `cell-eval2` (`ValScorer`), used by `evaluate` and `train` |
| | `metrics.py` | 30 | energy distance, Pearson r |
| `prediction/` | `predict.py` | 531 | **run:** model + any unperturbed `.h5ad` → predicted counts |
| | `counts_writer.py` | 138 | streams blocks into an `.h5ad`, checking each, then moves it into place |
| `submission/` | `submit_vcc26.py` | 330 | **run:** `package` (pre-flight + `vcc prep`), `upload`, `submit`; by you, never by a script |
| | `file_rules.py` | 72 | the portal's label rules (contexts, perturbation set, 400 cells, gene order) |

Run everything as `python -m signalflow.<folder>.<script>` from the repo root, e.g.
`python -m signalflow.training.train --config configs/prototype.yaml`.
