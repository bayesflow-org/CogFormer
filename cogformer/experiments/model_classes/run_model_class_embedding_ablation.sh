#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Model embedding ablation: train model class with and without
#   model identity embedding on DDM + RDM + CDM jointly.
# Outputs: cogformer/experiments/checkpoints/fm/model_class/
#          cogformer/experiments/figures/fm/model_class/
# ============================================================

EPOCHS=5000
STEPS_PER_EPOCH=100
TRAIN_BATCH=64
VAL_BATCH=100
LR=1e-4
DROPOUT=0.05
LAYER_DROPOUT=0.05
FM_SAMPLE_STEPS=200
FM_NUM_SAMPLES=100

run_model_class() {
    local EXTRA_ARGS=$1
    local LABEL=$2
    echo "=== Training model class: ${LABEL} ==="
    python -m cogformer.experiments.model_classes.model_class_cf_train \
        --epochs           "$EPOCHS" \
        --steps_per_epoch  "$STEPS_PER_EPOCH" \
        --train_batch_size "$TRAIN_BATCH" \
        --val_batch_size   "$VAL_BATCH" \
        --lr               "$LR" \
        --dropout          "$DROPOUT" \
        --layer_dropout    "$LAYER_DROPOUT" \
        --fm_sample_steps  "$FM_SAMPLE_STEPS" \
        --fm_num_samples   "$FM_NUM_SAMPLES" \
        $EXTRA_ARGS
    echo "=== Done: ${LABEL} ==="
}

run_model_class "" "with model embedding (baseline)"
run_model_class "--no_model_embedding" "without model embedding (ablation)"

echo "=== Model embedding ablation complete ==="
