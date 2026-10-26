#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
: "${ACCOUNT:?Set ACCOUNT}"
MANIFEST="${1:?usage: submit.sh /absolute/run/manifest.json [--submit] [--retry-failed]}"
shift
cd "$REPO_ROOT"
python3 -m tools.rebuttal.submit --manifest "$MANIFEST" --account "$ACCOUNT" "$@"
