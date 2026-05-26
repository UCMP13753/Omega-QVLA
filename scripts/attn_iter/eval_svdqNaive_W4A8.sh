#!/bin/bash
# Eval naive SVDQuant W4A8 pack — both LLM and DiT under aspq_gptq path.
set -uo pipefail
REPO=/work/mingze/aspq
SUITE="${SUITE:?must set SUITE}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-9100}"
OFF="${OFF:-10}"
TRIALS="${TRIALS:-10}"
LABEL="${LABEL:-svdqNaive_W4A8}"

INCLUDE_ALL='(.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))|.*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)).*'
EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'

case "$SUITE" in
    object)  TASK="libero_object";  DCFG="examples.Libero.custom_data_config:LiberoDataConfig";        CKPT="/work/mingze/models/VLA_ckpt/gr00t-n1.5-libero-object-posttrain"  ;;
    spatial) TASK="libero_spatial"; DCFG="examples.Libero.custom_data_config:LiberoDataConfig";        CKPT="/work/mingze/models/VLA_ckpt/gr00t-n1.5-libero-spatial-posttrain" ;;
    goal)    TASK="libero_goal";    DCFG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"; CKPT="/work/mingze/models/VLA_ckpt/gr00t-n1.5-libero-goal-posttrain"    ;;
    long)    TASK="libero_10";      DCFG="examples.Libero.custom_data_config:LiberoDataConfig";        CKPT="/work/mingze/models/VLA_ckpt/gr00t-n1.5-libero-long-posttrain"   ;;
esac

PACK="$REPO/results/multisuite_packs/${SUITE}_${LABEL}/quantized.pt"
[ -f "$PACK" ] || { echo "[err] missing $PACK"; exit 1; }

OUT="$REPO/results/attn_iter_${LABEL}/${SUITE}_off${OFF}_${TRIALS}tr"
[ -f "$OUT/merged_summary.json" ] && { echo "[skip] $OUT exists"; exit 0; }
rm -rf "$OUT/logs" "$OUT/rollouts" "$OUT/summaries" "$OUT/packdir" 2>/dev/null
mkdir -p "$OUT"

echo "[eval][$LABEL] $SUITE GPUS=$GPUS PORT=$PORT_BASE"
env \
    LIBERO_ROOT=/work/mingze/LIBERO CONDA_ROOT=/work/mingze/miniconda3 \
    QUANTVLA_CONDA_ENV=custon_asr GROOT_CONDA_ENV=custon_asr LIBERO_CONDA_ENV=custon_asr \
    QUANTVLA_CACHE_ROOT=/work/mingze/.cache/quantvla LIBERO_CONFIG_PATH=/home/mingze/.libero \
    QUANTVLA_ROOT="$REPO" CHECKPOINTS_ROOT="/work/mingze/models/VLA_ckpt" \
    CHECKPOINT="$CKPT" MODEL_PATH="$CKPT" \
    SUITE="$SUITE" VARIANT=with_dit TASK_SUITE="$TASK" DATA_CONFIG="$DCFG" \
    METHOD=aspq_gptq WBITS=4 ABITS=8 \
    GPU_LIST="$GPUS" PORT_BASE="$PORT_BASE" \
    NUM_TRIALS_PER_TASK="$TRIALS" GR00T_EVAL_INIT_OFFSET="$OFF" \
    GR00T_HYBRID_QUANT=1 \
    GR00T_ASPQ_GPTQ_PATH_OVERRIDE="$PACK" \
    GR00T_ASPQ_GPTQ_INCLUDE_OVERRIDE="$INCLUDE_ALL" \
    GR00T_ASPQ_GPTQ_EXCLUDE_OVERRIDE="$EXCLUDE" \
    GR00T_ASPQ_GPTQ_MISSING=fallback \
    GR00T_ASPQ_GPTQ_ACT_PCT=99.9 \
    GR00T_DUQUANT_INCLUDE='__NEVER_MATCH__' \
    GR00T_DUQUANT_EXCLUDE='.*' \
    OUTPUT_ROOT="$OUT" BENCHMARK_LABEL="${LABEL}_${SUITE}_off${OFF}" \
    bash "$REPO/scripts/run_custon_asr_4method_bench.sh" > "$OUT/driver.log" 2>&1
rc=$?
echo "[eval][$LABEL] $SUITE rc=$rc"
exit $rc
