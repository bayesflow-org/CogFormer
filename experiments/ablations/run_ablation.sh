#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# Activate the project virtualenv if present (skip if already active).
if [[ -z "${VIRTUAL_ENV:-}" && -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "=== Starting full ablation suite ==="

echo ""
echo "--- Phase 1: Component ablation (no_sab, no_mab, no_film, no_fourier) ---"
bash "$PROJECT_ROOT/experiments/ablations/run_component_ablation.sh"

echo ""
echo "--- Phase 2: Model embedding ablation (with / without model identity embedding) ---"
bash "$PROJECT_ROOT/experiments/model_classes/run_model_class_embedding_ablation.sh"

echo ""
echo "=== Full ablation suite complete ==="
