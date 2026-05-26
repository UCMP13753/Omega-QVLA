#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config

unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1

/work/mingze/miniconda3/envs/groot_test/bin/python \
  "${SCRIPT_DIR}/run_libero_duquant_benchmark_multi_gpu.py"
