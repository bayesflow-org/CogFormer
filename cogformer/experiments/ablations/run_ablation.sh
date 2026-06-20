#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/.venv/bin/activate"

echo "=== Starting full ablation suite ==="

echo ""
echo "--- Phase 1: Component ablation (no_sab, no_mab, no_film, no_fourier) ---"
bash "$SCRIPT_DIR/cogformer/experiments/ablations/run_component_ablation.sh"

echo ""
echo "--- Phase 2: Model embedding ablation (with / without model identity embedding) ---"
bash "$SCRIPT_DIR/cogformer/experiments/model_classes/run_model_class_embedding_ablation.sh"

echo ""
echo "=== Full ablation suite complete ==="
