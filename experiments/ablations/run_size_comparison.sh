#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# ============================================================
# Size comparison ablation: s → m → l → xl
# Outputs: outputs/checkpoints/ablations/size/ + outputs/data/predictions/ablations/size/
#          outputs/figures/ablations/size/
# ============================================================

USE_WANDB=--use_wandb   # set USE_WANDB=--use_wandb to enable

EPOCHS=1000
STEPS_PER_EPOCH=100
TRAIN_BATCH=64
VAL_BATCH=200
NUM_OBS=500
MIN_OBS=200
MAX_OBS=500
LR=1e-4
DROPOUT=0.05
LAYER_DROPOUT=0.05
FM_SAMPLE_STEPS=200
FM_NUM_SAMPLES=200

run_size() {
    local SIZE=$1
    echo "=== Training CogFormer-${SIZE} ==="
    python experiments/ablations/cf_size_comparison_train.py \
        --size             "$SIZE" \
        --epochs           "$EPOCHS" \
        --steps_per_epoch  "$STEPS_PER_EPOCH" \
        --train_batch_size "$TRAIN_BATCH" \
        --val_batch_size   "$VAL_BATCH" \
        --num_obs          "$NUM_OBS" \
        --min_num_obs      "$MIN_OBS" \
        --max_num_obs      "$MAX_OBS" \
        --lr               "$LR" \
        --dropout          "$DROPOUT" \
        --layer_dropout    "$LAYER_DROPOUT" \
        --fm_sample_steps  "$FM_SAMPLE_STEPS" \
        --fm_num_samples   "$FM_NUM_SAMPLES" \
        ${USE_WANDB}
    echo "=== Done: CogFormer-${SIZE} ==="
}

run_size s
run_size m
run_size l
run_size xl

echo "=== All size comparison runs complete ==="
