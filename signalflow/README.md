# SignalFlow v0 — conditional flow matching for perturbation response

Predicts what single cells look like after a CRISPR knockout, by learning a
**velocity field** that transports control cells onto perturbed cells.

Deliberately the simplest thing that is still the *right shape*: every piece
that will need to get smarter is an isolated, swappable class with its upgrade
path written next to it.

```bash
cd /Users/timstohn/Desktop/SignalFlow

python -m signalflow.data.prepare --config configs/prototype.yaml   # ~40 s
python -m signalflow.train        --config configs/prototype.yaml   # ~14 s/epoch (MPS)
python -m signalflow.evaluate     --config configs/prototype.yaml --split test
```

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
loss = masked MSE( v_θ(x_t, t | p, s(x0), ctx),  u )
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

### Cell state — dumb on purpose (`prepare.py:55`)

35 dims = **32 PCs + 3 scalars**:

- PCA fit on that context's **control cells only** — at inference we only ever
  start a trajectory from a control, so that is the distribution the basis has
  to cover
- scalars: `log1p(total UMI)`, `log1p(genes detected)`, `mean lognorm`
- z-scored against control-cell statistics
- `pca_components` / `pca_mean` / `state_mu` / `state_sd` are saved so new
  cells can be projected the same way

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
  contexts/<name>.npz
      X_data/X_indices/X_indptr/X_shape   lognorm expression, CSR
      gene_idx      int32 [n_local]       local column -> global gene index
      pert          int32 [n_cells]       -> pert vocab (0 = control)
      is_control    bool  [n_cells]
      state         f32   [n_cells, 35]   cell-state conditioning
      lib           f32   [n_cells]       total UMI (from raw .X)
      control_rows  int32 [n_control]
      pca_components / pca_mean / state_mu / state_sd
```

8 contexts, 31,210 cells, 35 MB.

---

## 4. Masking — the part that makes heterogeneous panels work

An unmeasured gene is **not** a gene measured as zero, and the model is told
which is which. `FlowDataset` builds `gene_mask [n_contexts, n_genes]`
(`dataset.py:76`) and it enters in **three** places:

| Where | Code | Why |
|---|---|---|
| input | `torch.cat([x_t * m, m], -1)` — `velocity.py:125` | the model sees the panel explicitly, so zero-because-unmeasured ≠ zero-because-silent |
| output | `return v * m` — `velocity.py:136` | the field never moves in directions the data cannot speak to |
| loss | `((v-u)²·m).sum() / m.sum()` — `flow.py:60` | **never `mean()` over G** — contexts measure different gene counts, and an unmasked mean makes their losses incomparable |

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
epoch so the gradient doesn't walk through datasets in blocks.

A batch is `{x0, x1, pert, ctx, state, lib0}`.

---

## 6. Model (`models/velocity.py`, `models/encoders.py`)

```
                 pert  ──► PertEncoder    (nn.Embedding, 64)   ─┐
                 state ──► StateEncoder   (2-layer MLP, 64)     ├─► cond (192)
                 ctx   ──► ContextEncoder (nn.Embedding, 32)    │
                 t     ──► TimeEncoder    (Fourier, 32)        ─┘
                                                                 │  (additive)
  [x_t·m, m] ──► Linear(2G→512) ──► 3× pre-norm ResBlock ──► LayerNorm ──► head ──► ·m
```

31.6M params at G=18,533. Output layer is zero-initialised, so the model
starts as the identity map (predict no change) and has to earn every
deviation.

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

1. **`PertEncoder` → feature-based.** The only change that makes unseen
   perturbations possible *at all*. Project features of the knocked-out gene:
   its own expression profile across control cells, an OmniPath/STRING network
   embedding, a DepMap essentiality vector. Same output shape — nothing
   downstream changes. Do this first; everything else is refinement.
2. **Add a distributional term to the loss** (energy / MMD on integrated
   endpoints, as STATE does). The current loss is pointwise; the metric is not.
3. **OT coupling instead of independent coupling** — pair `x0`/`x1` by
   minibatch optimal transport. Straighter paths, fewer integration steps,
   lower gradient variance.
4. **`StateEncoder` → shared across contexts**, so cell states are comparable
   between cell lines and a *new* line is representable. Currently the PCA
   basis is context-local, which the context embedding papers over.
5. **`head: dirmag`** — direction × magnitude, already wired, one config flag.
6. **Library-size modelling.** `flow.to_counts` reuses the source cell's depth;
   perturbations shift it.

Then scale the data: the prototype is 29k cells over 8 files, which is thin
for 80 perturbations.

---

## 10. File map

| File | Lines | What it holds |
|---|---|---|
| `vocab.py` | 106 | the two index spaces |
| `data/prepare.py` | 252 | h5ad → per-context npz, PCA, splits |
| `data/dataset.py` | 187 | pairing, gene scatter, mask, context sampler |
| `models/encoders.py` | 112 | pert / state / context / time encoders + upgrade notes |
| `models/velocity.py` | 136 | the field, both heads |
| `flow.py` | 109 | CFM loss, Euler sampler, lognorm→counts |
| `train.py` | 157 | loop, AdamW + cosine, checkpointing |
| `evaluate.py` | 178 | metrics + `identity` / `mean_shift` baselines |
