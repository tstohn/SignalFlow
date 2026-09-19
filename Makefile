# SignalFlow -- one-word entry points for the pipeline.
#
#   make training     prepare -> train -> evaluate on the test split
#   make prediction   predict the VCC26 contexts -> $(PRED)
#   make submission   package AND upload in one call (asks you to confirm first)
#
# Override any variable on the command line, e.g.
#   make submission MODEL_NAME="SignalFlow_0.1"
#   make prediction CHECKPOINT=runs/prototype/best_vcc.pt
#
# Preview what a target would run WITHOUT running it:   make -n submission

CONFIG     ?= configs/prototype.yaml
PY         ?= .venv/bin/python
CONTROLS   ?= data/VCC26/controls
PRED       ?= runs/prototype/predictions.h5ad
MODEL_NAME ?= SignalFlow_0.0
# empty = each script's own default, best.pt in the config's out_dir
CHECKPOINT ?=

CKPT_ARG = $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT))

.DEFAULT_GOAL := help
.PHONY: help training prediction submission prepare train evaluate
# `training` is three steps that must run in order, even under `make -j`
.NOTPARALLEL:

help:
	@echo "make training     prepare -> train -> evaluate (test split)"
	@echo "make prediction   predict VCC26 -> $(PRED)"
	@echo "make submission   package + upload in one call; you confirm before anything is sent"
	@echo ""
	@echo "parts of training, if you only want one:  make prepare | make train | make evaluate"
	@echo "preview any target without running it:    make -n <target>"

# ---- 1. training -------------------------------------------------------------

prepare:
	$(PY) -m signalflow.data.prepare --config $(CONFIG)

train:
	$(PY) -m signalflow.training.train --config $(CONFIG)

evaluate:
	$(PY) -m signalflow.evaluation.evaluate --config $(CONFIG) --split test --vcc $(CKPT_ARG)

training: prepare train evaluate

# ---- 2. prediction -----------------------------------------------------------

prediction:
	$(PY) -m signalflow.prediction.predict --config $(CONFIG) \
	    --input $(CONTROLS) \
	    --manifest $(CONTROLS)/manifest.json \
	    --perts $(CONTROLS)/pert_counts.csv \
	    --reference-genes $(CONTROLS)/gene_names.csv \
	    $(CKPT_ARG) \
	    --out $(PRED)

# ---- 3. submission -----------------------------------------------------------
# package + upload in ONE call. It does not run `prediction` first, and it asks you
# to type `submit` before anything leaves this machine. Needs ~22 GiB of RAM for a
# full-size file, so it will refuse on a 16 GB machine.

submission:
	$(PY) -m signalflow.submission.submit_vcc26 submit \
	    --pred $(PRED) --controls $(CONTROLS) -m "$(MODEL_NAME)" --wait
