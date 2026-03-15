#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ModelClass: training + validation
# ============================================================

# Architecture
ENCODER_LAYERS=8
DECODER_LAYERS=8
ENCODER_HEADS=8
DECODER_HEADS=8
PROJ_DIM=256
NUM_SEEDS=32
SEED_DIM=128
MODEL_EMBED_DIM=8
TIME_EMB_DIM=32
POS_EMB_DIM=32

# Training schedule
TRAIN_BATCH=64
EPOCHS=1000
STEPS=100
MIN_OBS=200
MAX_OBS=500
LR=1e-4
DROPOUT=0.05
LAYER_DROPOUT=0.05

# Validation
VAL_BATCH=200
VAL_BATCH_TRAIN=100   # val_step batch size during training (lighter than standalone validate)
VAL_NUM_OBS=500
FM_STEPS=200
FM_SAMPLES=200

# Derived checkpoint name (mirrors the naming in model_class_gpt_train.py)
CKPT_NAME="bayesgpt_model_class_fm_l${DECODER_LAYERS}_h${DECODER_HEADS}_p${PROJ_DIM}_s${NUM_SEEDS}_d${SEED_DIM}_b${TRAIN_BATCH}_e${EPOCHS}_t${STEPS}.pt"
CKPT_DIR="./bayesgpt/experiments/checkpoints/fm/model_class"
CKPT="${CKPT_DIR}/${CKPT_NAME}"

OUTDIR="./bayesgpt/experiments/figures/fm/model_class"
PRED_DIR="./bayesgpt/experiments/data/model_class"

# ============================================================
# Step 1: Train
# ============================================================
echo "=== [1/2] Training ModelClass GPT ==="
python -m bayesgpt.experiments.model_classes.model_class_gpt_train \
    --encoder_num_layers  "$ENCODER_LAYERS" \
    --decoder_num_layers  "$DECODER_LAYERS" \
    --encoder_num_heads   "$ENCODER_HEADS" \
    --decoder_num_heads   "$DECODER_HEADS" \
    --projection_dim      "$PROJ_DIM" \
    --num_seeds           "$NUM_SEEDS" \
    --seed_dim            "$SEED_DIM" \
    --model_embed_dim     "$MODEL_EMBED_DIM" \
    --time_embedding_dim  "$TIME_EMB_DIM" \
    --pos_embedding_dim   "$POS_EMB_DIM" \
    --train_batch_size    "$TRAIN_BATCH" \
    --epochs              "$EPOCHS" \
    --steps_per_epoch     "$STEPS" \
    --min_num_obs         "$MIN_OBS" \
    --max_num_obs         "$MAX_OBS" \
    --lr                  "$LR" \
    --dropout             "$DROPOUT" \
    --layer_dropout       "$LAYER_DROPOUT" \
    --use_wandb \
    --val_batch_size      "$VAL_BATCH_TRAIN" \
    --val_num_obs         "$VAL_NUM_OBS" \
    --fm_sample_steps     "$FM_STEPS" \
    --fm_num_samples      "$FM_SAMPLES"

# ============================================================
# Step 2: Validate
# ============================================================
echo "=== [2/3] Validating ModelClass GPT ==="
python -m bayesgpt.experiments.model_classes.model_class_gpt_validate \
    --checkpoint          "$CKPT" \
    --outdir              "$OUTDIR" \
    --pred_dir            "$PRED_DIR" \
    --batch_size          "$VAL_BATCH" \
    --num_obs             "$VAL_NUM_OBS" \
    --encoder_num_layers  "$ENCODER_LAYERS" \
    --decoder_num_layers  "$DECODER_LAYERS" \
    --encoder_num_heads   "$ENCODER_HEADS" \
    --decoder_num_heads   "$DECODER_HEADS" \
    --num_seeds           "$NUM_SEEDS" \
    --seed_dim            "$SEED_DIM" \
    --proj_dim            "$PROJ_DIM" \
    --model_embed_dim     "$MODEL_EMBED_DIM" \
    --time_embedding_dim  "$TIME_EMB_DIM" \
    --pos_embedding_dim   "$POS_EMB_DIM" \
    --num_sample_steps    "$FM_STEPS" \
    --num_samples         "$FM_SAMPLES"

ENSEMBLE_OUTDIR="${OUTDIR}/ensemble"

# ============================================================
# Step 3: Ensemble eval (global 8-param space, cross-model)
# ============================================================
echo "=== [3/4] Ensemble eval: ModelClass GPT ==="
python -m bayesgpt.experiments.model_classes.model_class_ensemble_eval \
    --checkpoint          "$CKPT" \
    --outdir              "$ENSEMBLE_OUTDIR" \
    --encoder_num_layers  "$ENCODER_LAYERS" \
    --decoder_num_layers  "$DECODER_LAYERS" \
    --encoder_num_heads   "$ENCODER_HEADS" \
    --decoder_num_heads   "$DECODER_HEADS" \
    --num_seeds           "$NUM_SEEDS" \
    --seed_dim            "$SEED_DIM" \
    --proj_dim            "$PROJ_DIM" \
    --model_embed_dim     "$MODEL_EMBED_DIM" \
    --time_embedding_dim  "$TIME_EMB_DIM" \
    --pos_embedding_dim   "$POS_EMB_DIM" \
    --num_sample_steps    "$FM_STEPS" \
    --num_samples         "$FM_SAMPLES"

# ============================================================
# Step 4: Ensemble metrics heatmap (50 configs, global space)
# ============================================================
echo "=== [4/4] Ensemble metrics: ModelClass GPT ==="
python -m bayesgpt.experiments.model_classes.model_class_ensemble_metrics \
    --checkpoint          "$CKPT" \
    --outdir              "$ENSEMBLE_OUTDIR" \
    --encoder_num_layers  "$ENCODER_LAYERS" \
    --decoder_num_layers  "$DECODER_LAYERS" \
    --encoder_num_heads   "$ENCODER_HEADS" \
    --decoder_num_heads   "$DECODER_HEADS" \
    --num_seeds           "$NUM_SEEDS" \
    --seed_dim            "$SEED_DIM" \
    --proj_dim            "$PROJ_DIM" \
    --model_embed_dim     "$MODEL_EMBED_DIM" \
    --time_embedding_dim  "$TIME_EMB_DIM" \
    --pos_embedding_dim   "$POS_EMB_DIM" \
    --num_sample_steps    "$FM_STEPS" \
    --num_samples         "$FM_SAMPLES"

echo "=== Done ==="
echo "Checkpoint : ${CKPT}"
echo "Figures    : ${OUTDIR}"
echo "Predictions: ${PRED_DIR}"
echo "Ensemble   : ${ENSEMBLE_OUTDIR}"
