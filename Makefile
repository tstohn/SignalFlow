# SignalFlow -- one-word entry points for the pipeline.
#
#   make prepare      raw .h5ad -> data/processed/   (run once per data change)
#   make holdout      train, keeping ONE cell line out      -- everyday model changes
#   make crossval     hold each line out in turn            -- reliable comparison + epoch count
#   make train-full   train on EVERY line, no validation    -- the final model
#   make prediction   predict the VCC26 contexts -> $(PRED)
#   make submission   package AND upload in one call (asks you to confirm first)
#
# Override any variable on the command line, e.g.
#   make holdout CONFIG=configs/other.yaml
#   make submission MODEL_NAME="SignalFlow_0.1"
#
# Preview what a target would run WITHOUT running it:   make -n submission

CONFIG     ?= configs/prototype.yaml
PY         ?= .venv/bin/python
CONTROLS   ?= data/VCC26/controls
RUNS       ?= runs/prototype
CKPT       ?= $(RUNS)/full/last.pt
PRED       ?= $(RUNS)/predictions.h5ad
MODEL_NAME ?= SignalFlow_0.0

# Run choices (held-out line, epochs, early stopping) are NOT here: they live in the config's
# `train:` block, the one place they are set. The targets below only pick the mode.

.DEFAULT_GOAL := help
.PHONY: help prepare holdout crossval train-full holdout-vcc25 prediction submission tensorboard
.NOTPARALLEL:

help:
	@echo "make prepare      raw .h5ad -> data/processed/   (once per data change)"
	@echo ""
	@echo "make holdout      train, ONE cell line held out   -- everyday model changes"
	@echo "make holdout-vcc25  hold out ALL VCC25 lines together as validation"
	@echo "make crossval     hold each line out in turn      -- comparison + epoch count"
	@echo "make train-full   train on EVERY line             -- the final model (train.full_epochs)"
	@echo ""
	@echo "make tensorboard  training curves of every run (loss, val loss, cell-eval)"
	@echo ""
	@echo "make prediction   predict VCC26 -> $(PRED)   (CKPT=$(CKPT))"
	@echo "make submission   package + upload in one call; you confirm before anything is sent"
	@echo ""
	@echo "preview a target without running it:  make -n <target>"

# ---- data --------------------------------------------------------------------

prepare:
	$(PY) -m signalflow.data.prepare --config $(CONFIG)

# ---- training ----------------------------------------------------------------

holdout:
	$(PY) -m signalflow.training.train --config $(CONFIG) --mode holdout

# validate on ALL the VCC25__* lines together (the flag replaces train.holdout)
holdout-vcc25:
	$(PY) -m signalflow.training.train --config $(CONFIG) --mode holdout --vcc25

crossval:
	$(PY) -m signalflow.training.train --config $(CONFIG) --mode crossval

# trains exactly train.full_epochs (from the config); crossval recommends the value
train-full:
	$(PY) -m signalflow.training.train --config $(CONFIG) --mode full

# ---- monitoring --------------------------------------------------------------
# curves of every run under train.out_dir (loss, val loss, cell-eval members); open the printed URL

tensorboard:
	$(PY) -m tensorboard.main --logdir $(RUNS)

# ---- prediction --------------------------------------------------------------
# CKPT defaults to the `full` run's last.pt; its folder must hold the pca_shared.npz
# that model was trained with (every run writes one).

prediction:
	$(PY) -m signalflow.prediction.predict --config $(CONFIG) \
	    --checkpoint $(CKPT) \
	    --input $(CONTROLS) \
	    --manifest $(CONTROLS)/manifest.json \
	    --perts $(CONTROLS)/pert_counts.csv \
	    --reference-genes $(CONTROLS)/gene_names.csv \
	    --out $(PRED)

# ---- submission --------------------------------------------------------------
# package + upload in ONE call. It does not run `prediction` first, and it asks you
# to type `submit` before anything leaves this machine. Needs ~22 GiB of RAM for a
# full-size file, so it will refuse on a 16 GB machine.

submission:
	$(PY) -m signalflow.submission.submit_vcc26 submit \
	    --pred $(PRED) --controls $(CONTROLS) -m "$(MODEL_NAME)" --wait
