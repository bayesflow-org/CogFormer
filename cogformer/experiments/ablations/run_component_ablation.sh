#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Component ablation study: baseline → no_sab → no_mab →
#   no_film → no_fourier on DDM interaction case.
# Outputs: cogformer/experiments/ablations/component_ablation_data/
#          cogformer/experiments/ablations/component_ablation_figures/
# ============================================================

EPOCHS=1000
STEPS_PER_EPOCH=100
TRAIN_BATCH=64
VAL_BATCH=200
NUM_OBS=500
MIN_OBS=200
MAX_OBS=500
LR=1e-4
DROPOUT=0.05
FM_SAMPLE_STEPS=200
FM_NUM_SAMPLES=200

# Architecture (matches 'l' size from size comparison)
NUM_LAYERS=8
NUM_HEADS=8
PROJ_DIM=256
NUM_SEEDS=32
SEED_DIM=64
TIME_EMB_DIM=32
POS_EMB_DIM=32

run_ablation() {
    local ABLATION=$1
    echo "=== Running component ablation: ${ABLATION} ==="
    python -m cogformer.experiments.ablations.cf_component_ablation_train \
        --ablation         "$ABLATION" \
        --epochs           "$EPOCHS" \
        --steps_per_epoch  "$STEPS_PER_EPOCH" \
        --train_batch_size "$TRAIN_BATCH" \
        --val_batch_size   "$VAL_BATCH" \
        --num_obs          "$NUM_OBS" \
        --min_num_obs      "$MIN_OBS" \
        --max_num_obs      "$MAX_OBS" \
        --lr               "$LR" \
        --dropout          "$DROPOUT" \
        --fm_sample_steps  "$FM_SAMPLE_STEPS" \
        --fm_num_samples   "$FM_NUM_SAMPLES" \
        --num_layers       "$NUM_LAYERS" \
        --num_heads        "$NUM_HEADS" \
        --proj_dim         "$PROJ_DIM" \
        --num_seeds        "$NUM_SEEDS" \
        --seed_dim         "$SEED_DIM" \
        --time_embedding_dim "$TIME_EMB_DIM" \
        --pos_embedding_dim  "$POS_EMB_DIM"
    echo "=== Done: ${ABLATION} ==="
}

run_ablation baseline
run_ablation no_sab
run_ablation no_mab
run_ablation no_film
run_ablation no_fourier

echo "=== All component ablation runs complete ==="
