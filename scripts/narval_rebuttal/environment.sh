#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${REBUTTAL_CONFIG:-${REPO_ROOT}/scripts/narval_rebuttal/config.local.sh}"
[[ -f "$CONFIG" ]] || { echo "Copy scripts/narval_rebuttal/config.example.sh to config.local.sh and edit it." >&2; exit 2; }
source "$CONFIG"
if [[ -n "${REBUTTAL_MODULES:-}" ]]; then
    read -r -a task_modules <<< "$REBUTTAL_MODULES"
    module load "${task_modules[@]}"
fi
: "${LIBERO_ROOT:?Set LIBERO_ROOT}"
: "${CHECKPOINTS_ROOT:?Set CHECKPOINTS_ROOT}"
: "${REBUTTAL_ROOT:?Set REBUTTAL_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_ROOT}:${OPENPI_ROOT:-}/src:${OPENPI_ROOT:-}/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-${REBUTTAL_ROOT}/cache/huggingface}"
export QUANTVLA_CACHE_ROOT="${REBUTTAL_ROOT}/cache"
export TORCH_HOME="${REBUTTAL_ROOT}/cache/torch"
# Per-job config avoids LIBERO writing interactive config files/racing on first import.
export LIBERO_CONFIG_PATH="${REBUTTAL_ROOT}/libero_config/${SLURM_JOB_ID:-local}"
mkdir -p "$LIBERO_CONFIG_PATH" "$HF_HOME" "$TORCH_HOME"
# Locate repository-owned LIBERO asset directories without importing LIBERO (which prompts).
python3 - "$LIBERO_ROOT" "$LIBERO_CONFIG_PATH" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
choices=[root/'libero'/'libero',root/'libero',root]
base=next((p for p in choices if (p/'bddl_files').is_dir()),None)
if base is None:
    raise SystemExit(f'Cannot find bddl_files under {root}; install LIBERO first')
config={'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),
        'init_states':str(base/'init_files'),'datasets':str(root/'datasets'),
        'assets':str(base/'assets')}
# JSON is a YAML subset, accepted by LIBERO yaml.safe_load.
(Path(sys.argv[2])/'config.yaml').write_text(json.dumps(config,indent=2))
PY
