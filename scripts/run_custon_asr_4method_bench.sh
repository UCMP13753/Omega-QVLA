#!/bin/bash
#
# Unified 4-method LIBERO benchmark with hold-out eval (init_states[0..39]).
#
# Methods:
#   1. duquant      - DuQuant per-row MSE (no ATM/OHB, no ASPQ)
#   2. quantvla     - DuQuant + ATM + OHB (paper config, runs run_quantvla.sh equivalents)
#   3. pure_gptq    - GPTQ baseline (no ASPQ refinement)
#   4. aspq_gptq    - GPTQ + ASPQ subspace refinement (ours)
#
# Suite:  one of {long, spatial, goal, object}
# Variant: one of {no_dit, with_dit}  - whether DiT attention is quantized
# Bits:    W3A8 or W4A8
#
# Each invocation runs ONE config on a chosen GPU set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

quantvla_activate_env "${QUANTVLA_CONDA_ENV:-custon_asr}"
quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config

CONDA_ROOT="$(quantvla_find_conda_root)"
PYTHON_BIN="${CONDA_ROOT}/envs/${QUANTVLA_CONDA_ENV:-custon_asr}/bin/python"

export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset VIRTUAL_ENV

# Required args
METHOD="${METHOD:?must set METHOD: duquant|quantvla|pure_gptq|aspq_gptq}"
SUITE="${SUITE:?must set SUITE: long|spatial|goal|object}"
VARIANT="${VARIANT:-no_dit}"     # no_dit | with_dit
WBITS="${WBITS:-4}"
ABITS="${ABITS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-5800}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-40}"   # 40 trials → init_states[0..39] (hold-out eval)
NUM_STEPS_WAIT=10
DENOISING_STEPS=8

case "$SUITE" in
    long)    TASK_SUITE="libero_10";     CKPT_NAME="libero-long-posttrain" ;;
    spatial) TASK_SUITE="libero_spatial"; CKPT_NAME="libero-spatial-posttrain" ;;
    goal)    TASK_SUITE="libero_goal";    CKPT_NAME="libero-goal-posttrain" ;;
    object)  TASK_SUITE="libero_object";  CKPT_NAME="libero-object-posttrain" ;;
    *) echo "unknown suite $SUITE" >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-${CHECKPOINTS_ROOT:-/ceph/workspace/xinyu/checkpoints}/gr00t-n1.5-${CKPT_NAME}}"
DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
[ "$SUITE" = "goal" ] && DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"

INCLUDE_NO_DIT='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
EXCLUDE_NO_DIT='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens|attn1)(?:\.|$)'
INCLUDE_WITH_DIT='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*'
EXCLUDE_WITH_DIT='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'

if [ "$VARIANT" = "no_dit" ]; then
    INCLUDE="$INCLUDE_NO_DIT"
    EXCLUDE="$EXCLUDE_NO_DIT"
else
    INCLUDE="$INCLUDE_WITH_DIT"
    EXCLUDE="$EXCLUDE_WITH_DIT"
fi
# Optional caller override (e.g. to run DiT-only INCLUDE with LLM left FP16)
if [ -n "${GR00T_ASPQ_GPTQ_INCLUDE_OVERRIDE:-}" ]; then
    INCLUDE="$GR00T_ASPQ_GPTQ_INCLUDE_OVERRIDE"
fi
if [ -n "${GR00T_ASPQ_GPTQ_EXCLUDE_OVERRIDE:-}" ]; then
    EXCLUDE="$GR00T_ASPQ_GPTQ_EXCLUDE_OVERRIDE"
fi

LABEL="${METHOD}_${SUITE}_${VARIANT}_w${WBITS}a${ABITS}"
# Allow caller-provided OUTPUT_ROOT to override (orchestrator embeds K into path)
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/4method_bench/${LABEL}}"
mkdir -p "${OUTPUT_ROOT}/logs"
LABEL=$(basename "${OUTPUT_ROOT}")

echo "================================================================"
echo "[bench] ${LABEL}  GPU_LIST=${GPU_LIST}  trials=${NUM_TRIALS_PER_TASK}"
echo "  ckpt=${CHECKPOINT}"
echo "  output=${OUTPUT_ROOT}"
echo "================================================================"

# Method-specific env config
case "$METHOD" in
    fp16)
        # No quantization at all — clear all quant env vars so models run pure FP16.
        unset GR00T_DUQUANT_INCLUDE GR00T_DUQUANT_EXCLUDE GR00T_DUQUANT_BLOCK \
              GR00T_DUQUANT_BLOCK_OUT GR00T_DUQUANT_PERMUTE GR00T_DUQUANT_ROW_ROT \
              GR00T_DUQUANT_ACT_PCT GR00T_DUQUANT_CALIB_STEPS GR00T_DUQUANT_LS \
              GR00T_DUQUANT_ASPQ GR00T_ASPQ_GPTQ GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        ;;
    duquant)
        # Pure DuQuant, no ATM/OHB, no ASPQ
        export GR00T_DUQUANT_INCLUDE="$INCLUDE"
        export GR00T_DUQUANT_EXCLUDE="$EXCLUDE"
        export GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64
        export GR00T_DUQUANT_PERMUTE=0 GR00T_DUQUANT_ROW_ROT=restore
        export GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32
        export GR00T_DUQUANT_LS=0.15
        unset GR00T_DUQUANT_ASPQ GR00T_ASPQ_GPTQ GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        ;;
    quantvla)
        # DuQuant + ATM + OHB (paper config)
        export GR00T_DUQUANT_INCLUDE="$INCLUDE"
        export GR00T_DUQUANT_EXCLUDE="$EXCLUDE"
        export GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64
        export GR00T_DUQUANT_PERMUTE=0 GR00T_DUQUANT_ROW_ROT=restore
        export GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32
        export GR00T_DUQUANT_LS=0.15
        export GR00T_ATM_ENABLE=1
        export GR00T_ATM_ALPHA_PATH="${QUANTVLA_ROOT}/atm_alpha_beta_${SUITE}.json"
        export GR00T_ATM_SCOPE=dit
        export GR00T_OHB_ENABLE=1
        export GR00T_OHB_FALLBACK=1.0
        export GR00T_OHB_SCOPE=dit
        unset GR00T_DUQUANT_ASPQ GR00T_ASPQ_GPTQ
        ;;
    pure_gptq|aspq_gptq)
        # Both go through ASPQ-GPTQ codepath; pure_gptq pack has empty ASPQ basis (built with min_eig=1e10).
        # Note: build uses mode names 'aspq' and 'pure_gptq', so map METHOD -> mode_name.
        if [ "$METHOD" = "aspq_gptq" ]; then PACK_MODE="aspq"; else PACK_MODE="pure_gptq"; fi
        # Allow caller to override pack path (e.g. point to aspq_sigma pack for ablation).
        QUANT_PATH="${GR00T_ASPQ_GPTQ_PATH_OVERRIDE:-${QUANTVLA_ROOT}/results/multisuite_packs/${SUITE}_${PACK_MODE}_w${WBITS}a${ABITS}/quantized.pt}"
        if [ ! -f "${QUANT_PATH}" ]; then
            echo "[bench] ERROR: missing pack ${QUANT_PATH}" >&2; exit 1
        fi
        export GR00T_ASPQ_GPTQ=1
        export GR00T_ASPQ_GPTQ_PATH="${QUANT_PATH}"
        export GR00T_ASPQ_GPTQ_INCLUDE="$INCLUDE"
        export GR00T_ASPQ_GPTQ_EXCLUDE="$EXCLUDE"
        export GR00T_ASPQ_GPTQ_WBITS_DEFAULT="$WBITS"
        export GR00T_ASPQ_GPTQ_ABITS="$ABITS"
        export GR00T_ASPQ_GPTQ_MISSING=fallback
        # Optional: runtime topk for ASPQ-GPTQ (default uses pack's stored 64)
        if [ "$METHOD" = "aspq_gptq" ] && [ -n "${GR00T_ASPQ_GPTQ_RUNTIME_TOPK:-}" ]; then
            export GR00T_ASPQ_GPTQ_RUNTIME_TOPK
        fi
        unset GR00T_ATM_ENABLE GR00T_OHB_ENABLE GR00T_DUQUANT_ASPQ
        ;;
    *) echo "unknown method $METHOD" >&2; exit 1 ;;
esac

export TASK_SUITE
export GPU_LIST
export NUM_TRIALS_PER_TASK NUM_STEPS_WAIT DENOISING_STEPS
export WBITS ABITS
export OUTPUT_ROOT
export PACKDIR="${OUTPUT_ROOT}/packdir"
export PORT_BASE
export MODEL_PATH="${CHECKPOINT}"
export BENCHMARK_LABEL="${LABEL}"

"${PYTHON_BIN}" -u "${QUANTVLA_ROOT}/scripts/run_libero_duquant_benchmark_multi_gpu.py" \
    > "${OUTPUT_ROOT}/logs/run.log" 2>&1
echo "[bench ${LABEL}] exit=$?"
