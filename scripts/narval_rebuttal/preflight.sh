#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
model="${1:?usage: preflight.sh groot|pi05 (inside a GPU allocation)}"
case "$model" in
  groot) task_env="${GROOT_ENV:?}" ;;
  pi05) task_env="${PI05_ENV:?}" ;;
  *) echo "unknown model $model" >&2; exit 2 ;;
esac
if [[ -f "$task_env/bin/activate" ]]; then source "$task_env/bin/activate"; fi
cd "$REPO_ROOT"
"$task_env/bin/python" -m tools.rebuttal.preflight --model "$model" \
  --checkpoints "$CHECKPOINTS_ROOT" --pi-checkpoint "${OPENPI_CHECKPOINT:-}"
