#!/bin/bash
# Train and validate all CDM BayesFlow cases sequentially.
#
# Usage:
#   bash bayesgpt/experiments/model_families/run_cdm_bf_all.sh
#   bash bayesgpt/experiments/model_families/run_cdm_bf_all.sh --epochs 500 --batch_size 128
#
# All flags after the script name are forwarded to the train script only.
# Validation always uses its own defaults (batch_size=200, num_samples=200).

CASES=("intercept_only" "fixed" "regressed" "fixed_regressed" "interaction" "full")
TRAIN_ARGS="$@"
SKIPPED=()

echo "=========================================="
echo " CDM BayesFlow: train + validate all cases"
echo " Train args: ${TRAIN_ARGS:-<defaults>}"
echo "=========================================="

for case in "${CASES[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo " CHECKING SIMULATOR: $case"
    echo "------------------------------------------"
    if ! python -m bayesgpt.experiments.model_families.cdm_family_bf_train \
            --case "$case" --test; then
        echo "  !! Simulator check failed for '$case' — skipping."
        SKIPPED+=("$case")
        continue
    fi

    echo ""
    echo "------------------------------------------"
    echo " TRAINING: $case"
    echo "------------------------------------------"
    python -m bayesgpt.experiments.model_families.cdm_family_bf_train \
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
    python -m bayesgpt.experiments.model_families.cdm_family_bf_validate \
        --case "$case"
done

echo ""
echo "=========================================="
echo " All cases complete."
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo " Skipped cases: ${SKIPPED[*]}"
fi
echo "=========================================="
