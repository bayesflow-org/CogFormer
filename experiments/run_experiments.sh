#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# ============================================================
# CogFormer — full experiment pipeline
#
# Covers:
#   1. Model family experiments  (DDM, RDM, CDM)
#      a. BayesFlow baseline training + validation (5 cases each)
#      b. CogFormer training + validation
#   2. Model class experiment    (DDM + RDM + CDM jointly)
#      a. CogFormer training + validation
#   3. Component ablation study  (DDM family)
#      a. Ablation training (no_sab, no_mab, no_film, no_fourier)
#      b. Evaluation across all 5 cases
#      c. Comparison tables (LaTeX)
#
# Outputs:
#   Checkpoints : outputs/checkpoints/
#   Figures     : outputs/figures/
#   Data        : outputs/data/predictions/
#   Ablations   : outputs/{checkpoints,tables,figures}/ablations/component/
#
# Usage:
#   bash experiments/run_experiments.sh
#   (run from the repository root with the venv activated)
# ============================================================

# ── Hyperparameters ─────────────────────────────────────────
EPOCHS=5000
STEPS_PER_EPOCH=100
TRAIN_BATCH=64
VAL_BATCH=200
LR=1e-4
DROPOUT=0.05
LAYER_DROPOUT=0.05
NUM_OBS=500
MIN_OBS=200
MAX_OBS=500
FM_SAMPLE_STEPS=200
FM_NUM_SAMPLES=200

NUM_LAYERS=8
NUM_HEADS=8
PROJ_DIM=256
NUM_SEEDS=32
SEED_DIM=64
TIME_EMB_DIM=32
POS_EMB_DIM=32

CASES="intercept_only fixed regressed fixed_regressed interaction"

DATA_DIR="./outputs/data/predictions"

# ── Checkpoint name helpers ──────────────────────────────────
family_cf_checkpoint() {
    local STEM=$1
    echo "./outputs/checkpoints/model_family/${2}/${STEM}_l${NUM_LAYERS}_h${NUM_HEADS}_p${PROJ_DIM}_s${NUM_SEEDS}_d${SEED_DIM}_o${NUM_OBS}_b${TRAIN_BATCH}_e${EPOCHS}_t${STEPS_PER_EPOCH}.pt"
}

model_class_checkpoint() {
    echo "./outputs/checkpoints/model_class/cogformer_model_class_fm_l${NUM_LAYERS}_h${NUM_HEADS}_p${PROJ_DIM}_s${NUM_SEEDS}_d${SEED_DIM}_b${TRAIN_BATCH}_e${EPOCHS}_t${STEPS_PER_EPOCH}.pt"
}

# ── 1. MODEL FAMILY EXPERIMENTS ──────────────────────────────

run_family() {
    local FAM=$1        # ddm | rdm | cdm
    local STEM=$2       # checkpoint stem (e.g. cogformer_mixed_attn)

    echo ""
    echo "============================================================"
    echo " Model family: ${FAM^^}"
    echo "============================================================"

    # BayesFlow baseline: train + validate (5 cases)
    for CASE in $CASES; do
        echo "--- BF train  [${FAM^^}] case=${CASE} ---"
        python experiments/model_families/family_bf_train.py \
            --model_family "$FAM" \
            --case         "$CASE" \
            --epochs       "$EPOCHS" \
            --steps_per_epoch "$STEPS_PER_EPOCH" \
            --batch_size   "$TRAIN_BATCH"

        echo "--- BF validate [${FAM^^}] case=${CASE} ---"
        python experiments/model_families/family_bf_validate.py \
            --model_family "$FAM" \
            --case         "$CASE" \
            --batch_size   "$VAL_BATCH" \
            --num_samples  "$FM_NUM_SAMPLES" \
            --skip_posteriors
    done

    # CogFormer: train
    echo "--- CF train [${FAM^^}] ---"
    python experiments/model_families/family_cf_train.py \
        --model_family      "$FAM" \
        --epochs            "$EPOCHS" \
        --steps_per_epoch   "$STEPS_PER_EPOCH" \
        --train_batch_size  "$TRAIN_BATCH" \
        --val_batch_size    "$VAL_BATCH" \
        --lr                "$LR" \
        --dropout           "$DROPOUT" \
        --layer_dropout     "$LAYER_DROPOUT" \
        --num_obs           "$NUM_OBS" \
        --min_num_obs       "$MIN_OBS" \
        --max_num_obs       "$MAX_OBS" \
        --decoder_num_layers "$NUM_LAYERS" \
        --encoder_num_layers "$NUM_LAYERS" \
        --decoder_num_heads  "$NUM_HEADS" \
        --encoder_num_heads  "$NUM_HEADS" \
        --projection_dim    "$PROJ_DIM" \
        --num_seeds         "$NUM_SEEDS" \
        --seed_dim          "$SEED_DIM" \
        --time_embedding_dim "$TIME_EMB_DIM" \
        --pos_embedding_dim  "$POS_EMB_DIM" \
        --use_wandb

    # CogFormer: validate
    local CKPT
    CKPT=$(family_cf_checkpoint "$STEM" "$FAM")

    echo "--- CF validate [${FAM^^}] ---"
    python experiments/model_families/family_cf_validate.py \
        --model_family  "$FAM" \
        --checkpoint    "$CKPT" \
        --data_dir      "$DATA_DIR" \
        --batch_size    "$VAL_BATCH" \
        --num_samples   "$FM_NUM_SAMPLES" \
        --num_sample_steps "$FM_SAMPLE_STEPS" \
        --skip_posteriors

    echo "=== Done: ${FAM^^} family ==="
}

run_family ddm "cogformer_mixed_attn"
run_family cdm "cogformer_cdm_mixed_attn"
run_family rdm "cogformer_rdm_mixed_attn"

# ── 2. MODEL CLASS EXPERIMENT ─────────────────────────────────

echo ""
echo "============================================================"
echo " Model class (DDM + RDM + CDM jointly)"
echo "============================================================"

echo "--- CF train [model class] ---"
python experiments/model_classes/model_class_cf_train.py \
    --epochs            "$EPOCHS" \
    --steps_per_epoch   "$STEPS_PER_EPOCH" \
    --train_batch_size  "$TRAIN_BATCH" \
    --val_batch_size    "$VAL_BATCH" \
    --lr                "$LR" \
    --dropout           "$DROPOUT" \
    --layer_dropout     "$LAYER_DROPOUT" \
    --decoder_num_layers "$NUM_LAYERS" \
    --encoder_num_layers "$NUM_LAYERS" \
    --decoder_num_heads  "$NUM_HEADS" \
    --encoder_num_heads  "$NUM_HEADS" \
    --proj_dim          "$PROJ_DIM" \
    --num_seeds         "$NUM_SEEDS" \
    --seed_dim          "$SEED_DIM" \
    --time_embedding_dim "$TIME_EMB_DIM" \
    --pos_embedding_dim  "$POS_EMB_DIM" \
    --use_wandb

MC_CKPT=$(model_class_checkpoint)

echo "--- CF validate [model class] ---"
python experiments/model_classes/model_class_cf_validate.py \
    --checkpoint    "$MC_CKPT" \
    --data_dir      "$DATA_DIR" \
    --batch_size    "$VAL_BATCH" \
    --num_samples   "$FM_NUM_SAMPLES" \
    --num_sample_steps "$FM_SAMPLE_STEPS" \
    --skip_posteriors

echo "=== Done: model class ==="

# ── 3. COMPONENT ABLATION STUDY ───────────────────────────────

echo ""
echo "============================================================"
echo " Component ablation study (DDM family)"
echo "============================================================"

ABL_DATA="./outputs/checkpoints/ablations/component"

run_ablation() {
    local ABLATION=$1
    echo "--- Ablation train: ${ABLATION} ---"
    python experiments/ablations/cf_component_ablation_train.py \
        --ablation          "$ABLATION" \
        --epochs            "$EPOCHS" \
        --steps_per_epoch   "$STEPS_PER_EPOCH" \
        --train_batch_size  "$TRAIN_BATCH" \
        --val_batch_size    "$VAL_BATCH" \
        --num_obs           "$NUM_OBS" \
        --min_num_obs       "$MIN_OBS" \
        --max_num_obs       "$MAX_OBS" \
        --lr                "$LR" \
        --dropout           "$DROPOUT" \
        --fm_sample_steps   "$FM_SAMPLE_STEPS" \
        --fm_num_samples    "$FM_NUM_SAMPLES" \
        --num_layers        "$NUM_LAYERS" \
        --num_heads         "$NUM_HEADS" \
        --proj_dim          "$PROJ_DIM" \
        --num_seeds         "$NUM_SEEDS" \
        --seed_dim          "$SEED_DIM" \
        --time_embedding_dim "$TIME_EMB_DIM" \
        --pos_embedding_dim  "$POS_EMB_DIM" \
        --use_wandb
}

run_ablation no_sab
run_ablation no_mab
run_ablation no_film
run_ablation no_fourier

# Evaluate baseline + all ablations across all 5 cases
DDM_CKPT=$(family_cf_checkpoint "cogformer_mixed_attn" "ddm")

for CONDITION in baseline no_sab no_mab no_film no_fourier; do
    if [ "$CONDITION" = "baseline" ]; then
        CKPT="$DDM_CKPT"
    else
        CKPT="${ABL_DATA}/cogformer_${CONDITION}_l${NUM_LAYERS}_h${NUM_HEADS}_p${PROJ_DIM}_s${NUM_SEEDS}_d${SEED_DIM}_o${NUM_OBS}_b${TRAIN_BATCH}_e${EPOCHS}_t${STEPS_PER_EPOCH}.pt"
    fi

    echo "--- Ablation eval: ${CONDITION} ---"
    python experiments/ablations/cf_component_ablation_eval.py \
        --condition     "$CONDITION" \
        --checkpoint    "$CKPT" \
        --batch_size    "$VAL_BATCH" \
        --num_samples   "$FM_NUM_SAMPLES" \
        --fm_sample_steps "$FM_SAMPLE_STEPS" \
        --num_layers    "$NUM_LAYERS" \
        --num_heads     "$NUM_HEADS" \
        --proj_dim      "$PROJ_DIM" \
        --num_seeds     "$NUM_SEEDS" \
        --seed_dim      "$SEED_DIM" \
        --time_embedding_dim "$TIME_EMB_DIM" \
        --pos_embedding_dim  "$POS_EMB_DIM"
done

echo "--- Ablation comparison tables ---"
python experiments/ablations/cf_component_ablation_compare.py

echo ""
echo "============================================================"
echo " All experiments complete."
echo "============================================================"
