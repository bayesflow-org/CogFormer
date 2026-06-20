#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# ============================================================
# Paths
# ============================================================
DATA_DIR="./experiments/data"
PRED_DIR="./experiments/data"

DDM_CKPT="./experiments/checkpoints/fm/ddm/cogformer_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e5000_t100.pt"
RDM_CKPT="./experiments/checkpoints/fm/rdm/cogformer_rdm_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e5000_t100.pt"
CDM_CKPT="./experiments/checkpoints/fm/cdm/cogformer_cdm_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e5000_t100.pt"

DDM_OUTDIR="./experiments/figures/fm/ddm/"
RDM_OUTDIR="./experiments/figures/fm/rdm/"
CDM_OUTDIR="./experiments/figures/fm/cdm/"

C2ST_DDM_OUTDIR="./experiments/figures/c2st/ddm/"
C2ST_RDM_OUTDIR="./experiments/figures/c2st/rdm/"
C2ST_CDM_OUTDIR="./experiments/figures/c2st/cdm/"

# ============================================================
# Step 1: CF validation (all 6 cases per model, using BF data)
# ============================================================

echo "=== [1/6] CF validate: DDM ==="
python experiments/model_families/family_cf_validate.py \
    --model_family ddm \
    --checkpoint "$DDM_CKPT" \
    --data_dir   "$DATA_DIR" \
    --outdir     "$DDM_OUTDIR" \
    --pred_dir   "$PRED_DIR"

echo "=== [2/6] CF validate: RDM ==="
python experiments/model_families/family_cf_validate.py \
    --model_family rdm \
    --checkpoint "$RDM_CKPT" \
    --data_dir   "$DATA_DIR" \
    --outdir     "$RDM_OUTDIR" \
    --pred_dir   "$PRED_DIR"

echo "=== [3/6] CF validate: CDM ==="
python experiments/model_families/family_cf_validate.py \
    --model_family cdm \
    --checkpoint "$CDM_CKPT" \
    --data_dir   "$DATA_DIR" \
    --outdir     "$CDM_OUTDIR" \
    --pred_dir   "$PRED_DIR"

# ============================================================
# Step 2: C2ST (BF predictions vs CF predictions)
# ============================================================

echo "=== [4/6] C2ST: DDM ==="
python experiments/model_families/family_c2st.py \
    --model_family ddm \
    --bf_data_dir  "$DATA_DIR" \
    --cf_pred_dir "$PRED_DIR" \
    --outdir       "$C2ST_DDM_OUTDIR"

echo "=== [5/6] C2ST: RDM ==="
python experiments/model_families/family_c2st.py \
    --model_family rdm \
    --bf_data_dir  "$DATA_DIR" \
    --cf_pred_dir "$PRED_DIR" \
    --outdir       "$C2ST_RDM_OUTDIR"

echo "=== [6/6] C2ST: CDM ==="
python experiments/model_families/family_c2st.py \
    --model_family cdm \
    --bf_data_dir  "$DATA_DIR" \
    --cf_pred_dir "$PRED_DIR" \
    --outdir       "$C2ST_CDM_OUTDIR"

echo "=== Done ==="