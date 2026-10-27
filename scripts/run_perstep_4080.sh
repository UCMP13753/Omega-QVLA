#!/usr/bin/env bash
# Three routes + original baseline, single GPU. Existing pack is never replaced.
# Usage: bash scripts/run_perstep_4080.sh /path/to/original/quantized.pt
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PACK="${1:?Usage: bash scripts/run_perstep_4080.sh /path/to/original/quantized.pt}"
PACK="$(realpath "$PACK")"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-$PWD/results/perstep_4080/$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo 'Select one GPU with CUDA_VISIBLE_DEVICES for this comparison.' >&2
    exit 1
fi
export GR00T_PERSTEP_ENGINE=triton
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Set SETUP=1 only when this is a fresh environment. Keep an existing model env.
if [[ "${SETUP:-0}" == 1 ]]; then
    "$PYTHON" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    "$PYTHON" -m pip install 'numpy<2' pytest triton==3.1.0
fi
"$PYTHON" -c 'import torch, triton; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(), torch.__version__, triton.__version__)'
"$PYTHON" -m pytest -q tests/test_perstep_backends.py

EXTRA=()
if [[ -n "${CHECKPOINT:-}" ]]; then
    # Capture mode requires your working GR00T/LIBERO environment.
    SUITE="${SUITE:-object}"
    TASK="libero_${SUITE}"
    [[ "$SUITE" == long ]] && TASK=libero_10
    DCFG=examples.Libero.custom_data_config:LiberoDataConfig
    [[ "$SUITE" == goal ]] && DCFG=examples.Libero.custom_data_config:LiberoDataConfigMeanStd
    CAPTURE_ROOT="${OUT}_captures"
    mkdir -p "$CAPTURE_ROOT"
    for SPLIT in calibration heldout; do
        OFFSET=0
        [[ "$SPLIT" == heldout ]] && OFFSET=10
        "$PYTHON" -m tools.capture_perstep_inputs --pack "$PACK" --checkpoint "$CHECKPOINT" \
            --task-suite-name "$TASK" --data-config "$DCFG" --init-offset "$OFFSET" \
            --output "$CAPTURE_ROOT/$SPLIT.pt"
    done
    EXTRA+=(--calibration-captures "$CAPTURE_ROOT/calibration.pt" --captures "$CAPTURE_ROOT/heldout.pt")
else
    echo '[scope] CHECKPOINT unset: synthetic microbench only; no model accuracy claim.'
fi
"$PYTHON" -m tools.compare_perstep_backends --pack "$PACK" --output "$OUT" \
    --group-size "${GROUP_SIZE:-128}" --tokens 32 128 \
    --warmup "${WARMUP:-10}" --repeats "${REPEATS:-30}" "${EXTRA[@]}"

# Optional actual task-success evaluation: existing GR00T conda + LIBERO setup.
# All configurations run sequentially on the same single GPU and same trials.
if [[ "${RUN_LIBERO:-0}" == 1 ]]; then
    : "${CHECKPOINT:?Set CHECKPOINT for RUN_LIBERO=1}"
    for MODE in baseline fused_float step_weight group_int; do
        SELECTED="$OUT/$MODE.pt"
        [[ "$MODE" == baseline ]] && SELECTED="$PACK"
        SUITE="$SUITE" CHECKPOINT="$CHECKPOINT" GPU_LIST="$CUDA_VISIBLE_DEVICES" LLM_QUANT=gptq DIT_QUANT=gptq \
            LLM_GPTQ_PACK="$SELECTED" DIT_GPTQ_PACK="$SELECTED" DIT_ATTN=1 WBITS=4 ABITS=4 \
            GR00T_GPTQ_PATH_OVERRIDE="$SELECTED" \
            GR00T_EVAL_INIT_OFFSET="${GR00T_EVAL_INIT_OFFSET:-10}" \
            NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}" \
            OUTPUT_ROOT="${OUT}_libero/$MODE" \
            bash scripts/run_groot_benchmark.sh
    done
fi
echo "Results: $OUT/manifest.json and $OUT/benchmark.csv"
