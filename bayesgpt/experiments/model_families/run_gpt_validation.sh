#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paths
# ============================================================
DATA_DIR="./bayesgpt/experiments/data"
PRED_DIR="./bayesgpt/experiments/data"

DDM_CKPT="./bayesgpt/experiments/checkpoints/fm/ddm/bayesgpt_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e5000_t100.pt"
RDM_CKPT="./bayesgpt/experiments/checkpoints/fm/rdm/bayesgpt_rdm_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e1000_t100.pt"
CDM_CKPT="./bayesgpt/experiments/checkpoints/fm/cdm/bayesgpt_cdm_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e1000_t100.pt"

DDM_OUTDIR="./bayesgpt/experiments/figures/fm/ddm/"
RDM_OUTDIR="./bayesgpt/experiments/figures/fm/rdm/"
CDM_OUTDIR="./bayesgpt/experiments/figures/fm/cdm/"

C2ST_DDM_OUTDIR="./bayesgpt/experiments/figures/c2st/ddm/"
C2ST_RDM_OUTDIR="./bayesgpt/experiments/figures/c2st/rdm/"
C2ST_CDM_OUTDIR="./bayesgpt/experiments/figures/c2st/cdm/"

# ============================================================
# Step 1: GPT validation (all 6 cases per model, using BF data)
# ============================================================

echo "=== [1/6] GPT validate: DDM ==="
python -m bayesgpt.experiments.model_families.ddm_family_gpt_validate \
    --checkpoint "$DDM_CKPT" \
    --data_dir   "$DATA_DIR" \
    --outdir     "$DDM_OUTDIR" \
    --pred_dir   "$PRED_DIR"

echo "=== [2/6] GPT validate: RDM ==="
python -m bayesgpt.experiments.model_families.rdm_family_gpt_validate \
    --checkpoint "$RDM_CKPT" \
    --data_dir   "$DATA_DIR" \
    --outdir     "$RDM_OUTDIR" \
    --pred_dir   "$PRED_DIR"

echo "=== [3/6] GPT validate: CDM ==="
python -m bayesgpt.experiments.model_families.cdm_family_gpt_validate \
    --checkpoint "$CDM_CKPT" \
    --data_dir   "$DATA_DIR" \
    --outdir     "$CDM_OUTDIR" \
    --pred_dir   "$PRED_DIR"

# ============================================================
# Step 2: C2ST (BF predictions vs GPT predictions)
# ============================================================

echo "=== [4/6] C2ST: DDM ==="
python -m bayesgpt.experiments.model_families.ddm_family_c2st \
    --bf_data_dir  "$DATA_DIR" \
    --gpt_pred_dir "$PRED_DIR" \
    --outdir       "$C2ST_DDM_OUTDIR"

echo "=== [5/6] C2ST: RDM ==="
python -m bayesgpt.experiments.model_families.rdm_family_c2st \
    --bf_data_dir  "$DATA_DIR" \
    --gpt_pred_dir "$PRED_DIR" \
    --outdir       "$C2ST_RDM_OUTDIR"

echo "=== [6/6] C2ST: CDM ==="
python -m bayesgpt.experiments.model_families.cdm_family_c2st \
    --bf_data_dir  "$DATA_DIR" \
    --gpt_pred_dir "$PRED_DIR" \
    --outdir       "$C2ST_CDM_OUTDIR"

echo "=== Done ==="