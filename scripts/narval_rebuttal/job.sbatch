#!/bin/bash
#SBATCH --job-name=omega-rebuttal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
: "${REBUTTAL_REPO:?}"
source "${REBUTTAL_REPO}/scripts/narval_rebuttal/environment.sh"
: "${MANIFEST:?}"
: "${JOB_INDEX:?}"
model="$(python3 - "$MANIFEST" "$JOB_INDEX" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['jobs'][int(sys.argv[2])]['config']['model'])
PY
)"
if [[ "$model" == groot ]]; then
    task_env="${GROOT_ENV:?Set GROOT_ENV}"
else
    task_env="${PI05_ENV:?Set PI05_ENV}"
fi
if [[ -f "$task_env/bin/activate" ]]; then
    source "$task_env/bin/activate"
else
    # Conda-created environment prefix, with libraries supplied through its prefix.
    export PATH="$task_env/bin:$PATH"
fi
[[ -x "$task_env/bin/python" ]] || { echo "Missing Python in $task_env" >&2; exit 2; }
cd "$REBUTTAL_REPO"
srun "$task_env/bin/python" -u -m tools.rebuttal.worker --manifest "$MANIFEST" --job "$JOB_INDEX"
