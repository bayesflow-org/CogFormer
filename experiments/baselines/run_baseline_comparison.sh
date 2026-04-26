#!/usr/bin/env bash
# Train and validate all 4 inference networks for the DDM interaction baseline,
# then produce a cross-network comparison.
#
# Usage:
#   bash experiments/baselines/run_baseline_comparison.sh
#   bash experiments/baselines/run_baseline_comparison.sh --epochs 2000 --batch_size 128
#
# Any flags after the script name are forwarded to the train step only.
# Run from the project root directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

MODULE="experiments.baselines.baseline_inference_network_comparison"
NETS=("coupling_flow" "flow_matching" "diffusion_model" "stable_consistency")
TRAIN_ARGS=("$@")

PASSED=()
FAILED=()

echo "============================================================"
echo " Baseline Inference Network Comparison — DDM Interaction"
echo " Networks   : ${NETS[*]}"
echo " Train args : ${TRAIN_ARGS[*]:-(defaults)}"
echo "============================================================"

for net in "${NETS[@]}"; do

    echo ""
    echo "------------------------------------------------------------"
    echo " TRAINING    :  $net"
    echo "------------------------------------------------------------"
    if ! python -m "$MODULE" --mode train --net "$net" "${TRAIN_ARGS[@]}"; then
        echo "  [FAIL] Training failed — $net"
        FAILED+=("train/$net")
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " VALIDATION  :  $net"
    echo "------------------------------------------------------------"
    if python -m "$MODULE" --mode validate --net "$net"; then
        PASSED+=("$net")
    else
        echo "  [FAIL] Validation failed — $net"
        FAILED+=("validate/$net")
    fi

done

echo ""
echo "------------------------------------------------------------"
echo " COMPARISON"
echo "------------------------------------------------------------"
python -m "$MODULE" --mode compare

echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
echo " Passed (${#PASSED[@]}) : ${PASSED[*]:-(none)}"
echo " Failed (${#FAILED[@]}) : ${FAILED[*]:-(none)}"
echo "============================================================"
