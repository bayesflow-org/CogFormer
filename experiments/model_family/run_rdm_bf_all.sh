#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
# Train and validate all RDM BayesFlow cases sequentially.
#
# Usage:
#   bash experiments/model_family/run_rdm_bf_all.sh
#   bash experiments/model_family/run_rdm_bf_all.sh --epochs 500 --batch_size 128
#
# All flags after the script name are forwarded to the train script only.
# Validation always uses its own defaults (batch_size=200, num_samples=200).

CASES=("intercept_only" "fixed" "regressed" "fixed_regressed" "interaction" "full")
TRAIN_ARGS="$@"
SKIPPED=()

echo "=========================================="
echo " RDM BayesFlow: train + validate all cases"
echo " Train args: ${TRAIN_ARGS:-<defaults>}"
echo "=========================================="

for case in "${CASES[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo " CHECKING SIMULATOR: $case"
    echo "------------------------------------------"
    if ! python experiments/model_family/rdm_family_bf_train.py \
            --case "$case" --test; then
        echo "  !! Simulator check failed for '$case' — skipping."
        SKIPPED+=("$case")
        continue
    fi

    echo ""
    echo "------------------------------------------"
    echo " TRAINING: $case"
    echo "------------------------------------------"
    python experiments/model_family/rdm_family_bf_train.py \
        --case "$case" $TRAIN_ARGS
    if [ $? -ne 0 ]; then
        echo "  !! Training failed for '$case' — skipping validation."
        SKIPPED+=("$case")
        continue
    fi

    echo ""
    echo "------------------------------------------"
    echo " VALIDATING: $case"
    echo "------------------------------------------"
    python experiments/model_family/rdm_family_bf_validate.py \
        --case "$case"
done

echo ""
echo "=========================================="
echo " All cases complete."
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo " Skipped cases: ${SKIPPED[*]}"
fi
echo "=========================================="
