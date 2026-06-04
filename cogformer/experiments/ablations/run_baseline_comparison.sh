#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BayesFlow baseline comparison: coupling_flow, diffusion_model,
#   flow_matching, stable_consistency on DDM interaction case.
# Outputs: cogformer/experiments/ablations/baselines_data/
#          cogformer/experiments/ablations/baselines_figures/
# ============================================================

EPOCHS=1000
STEPS_PER_EPOCH=100
BATCH_SIZE=64
VAL_BATCH_SIZE=200
NUM_SAMPLES=200

run_baseline() {
    local NETWORK=$1
    echo "=== Training BayesFlow baseline: ${NETWORK} ==="
    python -m cogformer.experiments.ablations.bf_baseline_comparison_train \
        --network        "$NETWORK" \
        --epochs         "$EPOCHS" \
        --steps_per_epoch "$STEPS_PER_EPOCH" \
        --batch_size     "$BATCH_SIZE" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --num_samples    "$NUM_SAMPLES"
    echo "=== Done: ${NETWORK} ==="
}

run_baseline coupling_flow
run_baseline diffusion_model
run_baseline flow_matching
run_baseline stable_consistency

echo "=== All baseline comparison runs complete ==="
