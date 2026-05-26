#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUANTVLA_ROOT="${QUANTVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

quantvla_find_conda_root() {
    local candidates=()

    if [[ -n "${CONDA_ROOT:-}" ]]; then
        candidates+=("${CONDA_ROOT}")
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidates+=("$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)")
    fi
    candidates+=("/work/mingze/miniconda3" "${HOME}/miniconda3" "/ceph/workspace/xinyu/miniconda3")

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "Unable to locate conda.sh. Set CONDA_ROOT explicitly." >&2
    return 1
}

quantvla_activate_env() {
    local env_name="$1"
    local conda_root
    conda_root="$(quantvla_find_conda_root)"
    # shellcheck disable=SC1090
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
}

quantvla_default_libero_root() {
    local candidates=()

    if [[ -n "${LIBERO_ROOT:-}" ]]; then
        candidates+=("${LIBERO_ROOT}")
    fi
    candidates+=(
        "${QUANTVLA_ROOT}/../LIBERO"
        "/work/mingze/LIBERO"
        "${HOME}/LIBERO"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "${QUANTVLA_ROOT}/../LIBERO"
}

quantvla_export_pythonpath() {
    local libero_root
    local entries=("${QUANTVLA_ROOT}")

    libero_root="$(quantvla_default_libero_root)"
    export LIBERO_ROOT="${libero_root}"
    if [[ -d "${libero_root}" ]]; then
        entries+=("${libero_root}")
    fi

    if [[ -n "${PYTHONPATH:-}" ]]; then
        entries+=("${PYTHONPATH}")
    fi

    export PYTHONPATH
    PYTHONPATH="$(IFS=:; echo "${entries[*]}")"
}

quantvla_setup_cache_dirs() {
    local cache_root="${QUANTVLA_CACHE_ROOT:-/work/mingze/.cache/quantvla}"
    mkdir -p "${cache_root}/huggingface" "${cache_root}/torch" "${cache_root}/xdg"

    export QUANTVLA_CACHE_ROOT="${cache_root}"
    export HF_HOME="${cache_root}/huggingface"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export TORCH_HOME="${cache_root}/torch"
    export XDG_CACHE_HOME="${cache_root}/xdg"
}

quantvla_setup_libero_config() {
    local config_root="${LIBERO_CONFIG_PATH:-/work/mingze/.libero}"
    mkdir -p "${config_root}"
    export LIBERO_CONFIG_PATH="${config_root}"
}
