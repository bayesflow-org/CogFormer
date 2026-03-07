#!/bin/bash
# Train and validate all DDM BayesFlow cases sequentially.
#
# Usage:
#   bash bayesgpt/experiments/model_families/run_ddm_bf_all.sh
#   bash bayesgpt/experiments/model_families/run_ddm_bf_all.sh --epochs 500 --batch_size 128
#
# All flags after the script name are forwarded to the train script only.
# Validation always uses its own defaults (batch_size=200, num_samples=200).

set -e

CASES=("intercept_only" "fixed" "regressed" "fixed_regressed" "interaction" "full")
TRAIN_ARGS="$@"

echo "=========================================="
echo " DDM BayesFlow: train + validate all cases"
echo " Train args: ${TRAIN_ARGS:-<defaults>}"
echo "=========================================="

for case in "${CASES[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo " TRAINING: $case"
    echo "------------------------------------------"
    python -m bayesgpt.experiments.model_families.ddm_family_bf_train \
        --case "$case" $TRAIN_ARGS

    echo ""
    echo "------------------------------------------"
    echo " VALIDATING: $case"
    echo "------------------------------------------"
    python -m bayesgpt.experiments.model_families.ddm_family_bf_validate \
        --case "$case"
done

echo ""
echo "=========================================="
echo " All cases complete."
echo "=========================================="
