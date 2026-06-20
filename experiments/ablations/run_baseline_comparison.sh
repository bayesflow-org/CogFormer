#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# ============================================================
# BayesFlow baseline comparison: coupling_flow, diffusion_model,
#   flow_matching, stable_consistency on DDM interaction case.
# Outputs: experiments/ablations/baselines_data/
#          experiments/ablations/baselines_figures/
# ============================================================

EPOCHS=1000
STEPS_PER_EPOCH=100
BATCH_SIZE=64
VAL_BATCH_SIZE=200
NUM_SAMPLES=200

run_baseline() {
    local NETWORK=$1
    echo "=== Training BayesFlow baseline: ${NETWORK} ==="
    python experiments/ablations/bf_baseline_comparison_train.py \
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
