# SignalFlow v0 — conditional flow matching for perturbation response

Deliberately the simplest thing that is still the *right shape*. Every
component that will need to get smarter is a separate swappable class with its
upgrade path written next to it.

```bash
python -m signalflow.data.prepare --config configs/prototype.yaml
python -m signalflow.train        --config configs/prototype.yaml
python -m signalflow.evaluate     --config configs/prototype.yaml --split test
```

---

## What the model outputs, and why

**A per-cell velocity field. Cells come out of integrating it.** Not either/or.

```
x0  ~ control cells of a context            (independent coupling — no pairing)
x1  ~ perturbed cells, same context, target gene p
t   ~ U(0,1)
x_t = (1-t)·x0 + t·x1
u   = x1 - x0
loss = masked MSE( v_θ(x_t, t | p, s(x0), ctx),  u )
```

Inference: start at a **real control cell**, Euler-integrate `t: 0→1`. The
endpoint is a predicted perturbed cell in lognorm space; `flow.to_counts`
takes it back to UMIs using the source cell's library size.

**Why the loss is on the velocity and not on `x1` directly.** Regressing `x1`
gives you the conditional *mean* — one point per perturbation, cell-to-cell
heterogeneity gone. Under the flow-matching loss the optimum is
`v*(x_t,t,c) = E[x1 - x0 | x_t, c]`, and integrating that field transports the
*whole* control distribution onto the *whole* perturbed distribution. The
spread comes out for free, which is exactly what the VCC-style distributional
metrics score.

It also makes your `direction × magnitude` factorisation a one-line change:
`model.head: dirmag` gives `v = softplus(m)·normalize(d)`.

### How other methods do it

| Method | Output | Loss |
|---|---|---|
| GEARS, scGPT, CPA, biolord | post-pert expression (usually δ from control mean) | MSE, + MSE on top-DE genes |
| scGen | latent shift, then decode | VAE ELBO + latent arithmetic |
| CellOT | OT map control→perturbed | ICNN dual OT objective |
| CellFlow / OT-CFM family | velocity field, ODE integrate | flow-matching MSE on velocity |
| STATE (Arc, VCC25) | perturbed *set* from control *set* | MSE + distributional (energy/MMD) |

The first row collapses each perturbation to a point. That is the thing to
avoid, and the reason for choosing flow matching here.

---

## Two corrections to the original spec

**1. `selected_genes.csv` is not the perturbation vocabulary.** It is the
*readout* panel (1,213 unique genes = 300 HVG + 200 random per dataset). Only
**4 of the 80** prototype perturbations appear in it. The perturbation
vocabulary is `data/VCC26/controls/gene_names.csv` (18,533 symbols), which
covers all 80. Two separate index spaces, kept apart in `vocab.py`:

- `GeneVocab` — readout space, what we predict expression for
- `PertVocab` — perturbation space, index 0 reserved for `non-targeting`

**2. No perturbation is shared between any two of the eight files.** 10 each,
80 unique, zero overlap. With a one-hot encoder a held-out perturbation keeps
its random-init embedding row, so generalisation to unseen perturbations is
impossible — not poor, *absent*. Splits are therefore on **cells**, stratified
by perturbation. Swapping `PertEncoder` for a feature-based one (control
expression profile of the knocked-out gene / OmniPath / DepMap) is the single
change that unlocks unseen perturbations.

---

## Masking

Each context has its own gene panel (630–1,147 genes here, union 1,213). An
unmeasured gene is **not** a gene measured as zero, and the model is told
which is which. The mask enters in three places:

1. input is `[x_t · mask, mask]` — the model sees the panel explicitly;
2. output is multiplied by the mask — the field never moves in directions the
   data cannot speak to;
3. the loss averages over **measured entries only** — never `mean()` over `G`,
   or loss magnitudes stop being comparable across contexts.

Nothing is zero-padded on disk. Each context stores a compact matrix plus
`gene_idx` (local column → global readout index); the scatter happens per
batch. That is what keeps this workable at 18,533 genes — verified: prep and
training both run unchanged with `gene_vocab_csv` pointed at the VCC26 list
(31.6M params, 15 s/epoch on MPS).

---

## Cell-state conditioning

Dumb on purpose: **32 PCs + 3 scalars = 35 dims**, precomputed in `prepare.py`.

- PCA fit on that context's **control cells only** — at inference we only ever
  start from a control, so that is the distribution the basis must cover
- scalars: `log1p(total UMI)`, `log1p(genes detected)`, `mean lognorm`
- z-scored against control-cell statistics

Conditioning always comes from the **source** cell `x0`. `x1`'s state must
never leak in — at inference it does not exist.

This basis is context-local, which is fine for v0 because the model also gets
a context embedding. A shared cross-context encoder (or scFoundation /
scBaseCount coordinates) is the v1 upgrade; `StateEncoder`'s interface does
not change.

---

## Data structure

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
      pca_components / pca_mean / state_mu / state_sd   (project new cells)
```

Control cells appear as targets too, with pert index 0, so the model learns
that "non-targeting" means near-zero net displacement — with the real
control-to-control spread still in it. Without that anchor nothing pins the
magnitude scale.

`ContextBatchSampler` draws each batch from a single context (cheap scatter,
one shared mask), shuffling context order every epoch.

---

## Results, prototype subset (40 epochs, ~2.6 s/epoch on MPS)

Test split, 70 perturbations, 8 contexts. `delta_r` = Pearson r between
predicted and true mean shift from the control mean — the headline number,
because plain expression correlation sits near 1.0 for everything including
`identity` and tells you nothing.

| method | delta_r (mean) | MAE | energy |
|---|---|---|---|
| identity (no change) | 0.036 | 0.733 | 16.87 |
| **flow (this model)** | **0.122** | **0.719** | **16.41** |
| mean_shift baseline | 0.267 | 0.710 | 16.18 |

**Read this honestly: the flow beats the do-nothing floor by ~3×, and loses to
the mean-shift baseline.** `mean_shift` is handed the per-perturbation first
moment from the train split, so it is a hard baseline by construction — and
most published gains in this field evaporate against it, which is why it is in
the harness. With ~25k cells, 500 readout genes per context and as few as 5
test cells per perturbation, v0 losing here is the expected starting point,
not a bug. The scaffolding is the deliverable; this row is the number to beat.

Two diagnosable v0 symptoms already visible:

- controls move too much (energy 2.42 vs 1.31 for identity) — the model is not
  fully respecting the "non-targeting = no motion" anchor
- `frac_var_explained` plateaus near 9% — most of `x1 - x0` is irreducible
  cell-to-cell noise, so this is not directly alarming, but it does mean the
  signal is thin at this data scale

---

## Upgrade order (highest leverage first)

1. **`PertEncoder` → feature-based.** The only change that makes unseen
   perturbations possible at all. Everything else is refinement.
2. **Add a distributional term to the loss** (energy / MMD on integrated
   endpoints, as STATE does). The current loss is pointwise; the metric is not.
3. **OT coupling instead of independent coupling** — pair `x0`/`x1` by
   minibatch OT. Straighter paths, fewer integration steps, lower variance.
4. **`StateEncoder` → shared across contexts**, so cell states are comparable
   between cell lines and a new line is representable.
5. **`head: dirmag`** — direction × magnitude, already wired.
6. **Library-size modelling.** `to_counts` currently reuses the source cell's
   depth; perturbations shift it.

## Files

| File | What it holds |
|---|---|
| `vocab.py` | the two index spaces |
| `data/prepare.py` | h5ad → per-context npz, PCA, splits |
| `data/dataset.py` | pairing, gene scatter, mask, context sampler |
| `models/encoders.py` | pert / state / context / time encoders + upgrade notes |
| `models/velocity.py` | the field, both heads |
| `flow.py` | CFM loss, Euler sampler, lognorm→counts |
| `train.py` | loop |
| `evaluate.py` | metrics + `identity` / `mean_shift` baselines |
