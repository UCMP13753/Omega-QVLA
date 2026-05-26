#!/bin/bash
# Script to run Libero evaluation
# Usage: ./run_libero_eval.sh [task_suite_name] [extra args...]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common_paths.sh"

TASK=${1:-libero_10}
shift || true
EXTRA_ARGS=("$@")

HEADLESS_FLAG="no"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
        break
    fi
done

# Activate libero_test environment
quantvla_activate_env libero_test

# Add QuantVLA_GR00T and LIBERO to Python path
quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config

echo "=========================================="
echo "Running Libero evaluation for $TASK"
echo "Headless mode: $HEADLESS_FLAG"
echo "Port: 5556 (GR00T)"
echo "QuantVLA root: $QUANTVLA_ROOT"
echo "LIBERO root: $LIBERO_ROOT"
echo "HF cache: $HF_HOME"
echo "LIBERO config: $LIBERO_CONFIG_PATH"
echo "=========================================="
echo ""
echo "Make sure the inference server is running in another terminal!"
echo "Run: ./run_inference_server.sh $TASK"
echo ""
echo "Results will be saved to:"
echo "  - Log: /tmp/logs/libero_eval_${TASK}.log"
echo "  - Videos: /tmp/logs/rollout_*.mp4"
echo "=========================================="
echo ""

cd "${QUANTVLA_ROOT}/examples/Libero/eval"

python run_libero_eval.py --task_suite_name "$TASK" --port 5556 "${EXTRA_ARGS[@]}"
