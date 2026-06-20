#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
# Train and validate all BayesFlow cases for a given model family.
#
# Usage:
#   bash experiments/model_family/run_family_bf_all.sh --model_family ddm
#   bash experiments/model_family/run_family_bf_all.sh --model_family rdm --epochs 500 --batch_size 128
#
# --model_family is required. All remaining flags are forwarded to the train script only.
# Validation always uses its own defaults (batch_size=200, num_samples=200).

MODEL_FAMILY=""
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_family)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL_FAMILY" ]]; then
    echo "Error: --model_family is required (ddm, rdm, or cdm)"
    exit 1
fi

CASES=("intercept_only" "fixed" "regressed" "fixed_regressed" "interaction" "full")
SKIPPED=()

echo "=========================================="
echo " ${MODEL_FAMILY^^} BayesFlow: train + validate all cases"
echo " Train args: ${TRAIN_ARGS[*]:-<defaults>}"
echo "=========================================="

for case in "${CASES[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo " CHECKING SIMULATOR: $case"
    echo "------------------------------------------"
    if ! python experiments/model_family/family_bf_train.py \
            --model_family "$MODEL_FAMILY" --case "$case" --test; then
        echo "  !! Simulator check failed for '$case' — skipping."
        SKIPPED+=("$case")
        continue
    fi

    echo ""
    echo "------------------------------------------"
    echo " TRAINING: $case"
    echo "------------------------------------------"
    python experiments/model_family/family_bf_train.py \
        --model_family "$MODEL_FAMILY" --case "$case" "${TRAIN_ARGS[@]}"
    if [ $? -ne 0 ]; then
        echo "  !! Training failed for '$case' — skipping validation."
        SKIPPED+=("$case")
        continue
    fi

    echo ""
    echo "------------------------------------------"
    echo " VALIDATING: $case"
    echo "------------------------------------------"
    python experiments/model_family/family_bf_validate.py \
        --model_family "$MODEL_FAMILY" --case "$case"
done

echo ""
echo "=========================================="
echo " All cases complete."
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo " Skipped cases: ${SKIPPED[*]}"
fi
echo "=========================================="
