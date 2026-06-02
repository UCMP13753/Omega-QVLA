# Omega-QVLA — W4A4 Quantization for VLA Action Heads

W4A4 quantization for the GR00T-N1.5 (and pi0.5) LIBERO action heads.

**The recipe** (per side, applied to both the LLM/backbone and the DiT/action head):

| side | rotation | quantizer | per-step |
|------|----------|-----------|----------|
| **LLM** (Eagle) | DuQuant `svd_hadamard` | **GPTQ** | — |
| **DiT** (action head) | DuQuant `svd_hadamard` | **RTN** (residual) | **yes** (`act_scale_table`) |

Both sides are offline **GPTQ packs** (the rotation, the RTN-vs-GPTQ residual
choice, and the per-step scale table are all baked into the pack at build time);
the runtime loads them through `GptqLinear`.

---

## 1. Requirements

- 1× A100 (40 GB) to build; 4–8× for parallel multi-suite eval.
- Python 3.10, CUDA 12.4, PyTorch 2.5.1, conda env **`omega_qvla`**
  (`pip install -e .`). pi0.5 build/eval uses a separate **openpi** venv.
- LIBERO: `git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $HOME/LIBERO && pip install -e $HOME/LIBERO`.

## 2. Setup — environment variables

All paths are env-driven (nothing is hardcoded). Set these to your machine:

```bash
export QUANTVLA_ROOT=$(pwd)                       # this repo
export QUANTVLA_CONDA_ENV=omega_qvla
export CONDA_ROOT=$HOME/miniconda3                # where conda lives (REQUIRED)
export CHECKPOINTS_ROOT=$HOME/ckpts               # gr00t / pi05 checkpoints
export LIBERO_ROOT=$HOME/LIBERO
export LIBERO_CONFIG_PATH=$HOME/.libero           # writable; LIBERO suite YAMLs
export QUANTVLA_CACHE_ROOT=$HOME/.cache/omega_qvla
export OPENPI_ROOT=$HOME/openpi                   # pi0.5 only
mkdir -p $QUANTVLA_CACHE_ROOT $LIBERO_CONFIG_PATH
```

Checkpoints (GR00T-N1.5, 4 suites):
```bash
for s in goal spatial object long; do
    huggingface-cli download youliangtan/gr00t-n1.5-libero-${s}-posttrain \
        --local-dir $CHECKPOINTS_ROOT/gr00t-n1.5-libero-${s}-posttrain
done
```

---

## 3. GR00T-N1.5 — run the recipe

Pick a suite:
```bash
SUITE=object                         # object | spatial | goal | long
CKPT=$CHECKPOINTS_ROOT/gr00t-n1.5-libero-${SUITE}-posttrain
case "$SUITE" in
    goal) TASK=libero_goal;    DCFG=examples.Libero.custom_data_config:LiberoDataConfigMeanStd ;;
    long) TASK=libero_10;      DCFG=examples.Libero.custom_data_config:LiberoDataConfig ;;
    *)    TASK=libero_${SUITE}; DCFG=examples.Libero.custom_data_config:LiberoDataConfig ;;
esac
LLM_RE='.*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'
DIT_RE='.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*'
```

### 3.1 Build the LLM pack — DuQuant svd_hadamard + GPTQ

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$QUANTVLA_ROOT python -m tools.build_gptq_weights \
    --checkpoint "$CKPT" --task-suite-name "$TASK" --data-config "$DCFG" \
    --output-path results/packs/${SUITE}_LLM/quantized.pt \
    --include-regex "$LLM_RE" --exclude-regex "$EXCLUDE" \
    --duquant-rotation --duquant-rot-mode svd_hadamard \
    --weight-bits 4 --num-samples 10 --token-cap 1024 \
    --gptq-block-size 128 --gptq-damp-percent 0.05
```

### 3.2 Build the DiT pack — DuQuant svd_hadamard + RTN residual + per-step

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$QUANTVLA_ROOT python -m tools.build_dit_a2lite_svd_gptq_perstep \
    --checkpoint "$CKPT" --task-suite-name "$TASK" --data-config "$DCFG" \
    --output-path results/packs/${SUITE}_DiT/quantized.pt \
    --num-samples 10 --token-cap 1024 --num-steps 8 \
    --svd-rank 0 --use-rtn \
    --w-bits 4 --a-bits 4 --act-percentile 99.9 \
    --duquant-block-size 64 --duquant-block-out 64 \
    --gptq-block-size 128 --gptq-damp-percent 0.05
```
(The DiT builder auto-targets `transformer_blocks.*.attn1` + `ff.net` and always
uses `svd_hadamard`; `--use-rtn` = RTN residual, `--num-steps 8` = per-step table.)

### 3.3 Merge the two packs (runtime loads one file)

```bash
python -m tools.merge_packs \
    --out results/packs/${SUITE}_MERGED/quantized.pt \
    results/packs/${SUITE}_LLM/quantized.pt \
    results/packs/${SUITE}_DiT/quantized.pt
```

### 3.4 Evaluate

```bash
env CONDA_ROOT=$CONDA_ROOT \
    SUITE=$SUITE WBITS=4 ABITS=4 \
    LLM_QUANT=gptq DIT_QUANT=gptq DIT_ATTN=1 DIT_PERSTEP=1 \
    GR00T_GPTQ_PATH_OVERRIDE=$QUANTVLA_ROOT/results/packs/${SUITE}_MERGED/quantized.pt \
    GR00T_GPTQ_INCLUDE_OVERRIDE="(${LLM_RE}|${DIT_RE})" \
    GR00T_GPTQ_MISSING=fallback \
    GPU_LIST=0,1,2,3 PORT_BASE=8000 NUM_TRIALS_PER_TASK=10 GR00T_EVAL_INIT_OFFSET=10 \
    OUTPUT_ROOT=$QUANTVLA_ROOT/results/eval/${SUITE} \
    bash $QUANTVLA_ROOT/scripts/run_groot_benchmark.sh

python -c "import json; print(round(100*json.load(open('results/eval/${SUITE}/merged_summary.json'))['total_success_rate'],1), '%')"
```
For `long`, use `GPU_LIST=0,1,2,3,4,5,6,7`.

---

## 4. What each knob is controlled by

| knob | controlled by | where |
|------|---------------|-------|
| **rotation = svd_hadamard (LLM)** | `--duquant-rotation --duquant-rot-mode svd_hadamard` | LLM build (§3.1) — baked into pack |
| **rotation = svd_hadamard (DiT)** | hardcoded in the DiT builder | DiT build (§3.2) — baked into pack |
| **GPTQ (LLM quantizer)** | `build_gptq_weights.py` runs the GPTQ solver | LLM build (§3.1) |
| **RTN residual (DiT quantizer)** | `--use-rtn` (vs GPTQ error-comp if omitted) | DiT build (§3.2) |
| **per-step scale** | `--num-steps 8` → saves `act_scale_table`; auto-dispatched at runtime by `dit_step_context` | DiT build (§3.2) |
| **weight bits = 4** | `--weight-bits 4` (LLM) / `--w-bits 4` (DiT) | build |
| **activation bits = 4** | `ABITS=4` → `GR00T_GPTQ_ABITS` (set at eval, not in pack) | eval (§3.4) |
| **which side uses which quantizer** | `LLM_QUANT` / `DIT_QUANT` ∈ `{gptq, rtn, duquant, none}` | eval (§3.4) |
| **DuQuant rotation mode (runtime path only)** | `LLM_ROT` / `DIT_ROT` → `GR00T_DUQUANT_ROT_MODE` (only when a side is `duquant`, not used in this all-pack recipe) | eval |
| **include DiT attention** | `DIT_ATTN=1` (else MLP only) | eval (§3.4) |
| **which pack file** | `GR00T_GPTQ_PATH_OVERRIDE` (single merged pack) | eval (§3.4) |
| **which layers to wrap** | `GR00T_GPTQ_INCLUDE_OVERRIDE` / `..._EXCLUDE_OVERRIDE` regex | eval (§3.4) |
| **GPUs / shards / trials** | `GPU_LIST`, `PORT_BASE`, `NUM_TRIALS_PER_TASK`, `GR00T_EVAL_INIT_OFFSET` | eval (§3.4) |

The benchmark switch `*_QUANT=gptq` ⇒ wraps that side with `GptqLinear`
(`GR00T_GPTQ_*`); `=duquant` ⇒ runtime DuQuant (rotation+RTN, `GR00T_DUQUANT_*`);
`=rtn` ⇒ pure RTN (`GR00T_RTN_*`). `run_groot_benchmark.sh` translates the
high-level switches into these `GR00T_*` env vars and composes them by scope; the
unified entry `gr00t.quantization.enable_quant_if_configured` applies them.

---

## 5. pi0.5

Same recipe on the action **Expert** (= DiT): build a GPTQ pack with
`svd_hadamard` + `--use-rtn` + per-step, then eval via `METHOD=hybrid`. The
**PaliGemma** backbone runs as runtime DuQuant `svd_hadamard` — there is no
PaliGemma-side GPTQ builder / prefix-activation recorder, so PaliGemma cannot be
a GPTQ pack.

```bash
SUITE=object
EXPERT_RE='.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
PALI_RE='.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'

# Build Expert pack (needs an obs dump at --obs-path; reuse an existing one)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$QUANTVLA_ROOT \
$OPENPI_ROOT/.venv/bin/python -m tools.build_pi05_a2lite_gptq_perstep \
    --checkpoint $CHECKPOINTS_ROOT/pi05_libero_pytorch --data-config pi05_libero \
    --obs-path duquant_act_stats/pi05_libero_${SUITE}_obs.pt \
    --output results/packs/pi05_${SUITE}_expert/quantized.pt \
    --include-regex "$EXPERT_RE" \
    --max-samples 10 --token-cap 512 --num-steps 10 --use-rtn \
    --w-bits 4 --a-bits 4 --duquant-block-size 64 --duquant-block-out 64 \
    --gptq-block-size 128 --gptq-damp-percent 0.05

# Eval: Expert GPTQ pack + PaliGemma DuQuant svd_hadamard
env CONDA_ROOT=$CONDA_ROOT METHOD=hybrid SUITE=$SUITE WBITS=4 ABITS=4 \
    GPU_LIST=0,1,2,3 PORT_BASE=8100 NUM_TRIALS_PER_TASK=10 GR00T_EVAL_INIT_OFFSET=10 \
    OPENPI_ROOT=$OPENPI_ROOT OPENPI_PY=$OPENPI_ROOT/.venv/bin/python \
    OPENPI_CONFIG=pi05_libero OPENPI_CHECKPOINT=$CHECKPOINTS_ROOT/pi05_libero_pytorch \
    OPENPI_GPTQ_PATH=$QUANTVLA_ROOT/results/packs/pi05_${SUITE}_expert/quantized.pt \
    OPENPI_GPTQ_INCLUDE="$EXPERT_RE" OPENPI_DUQUANT_INCLUDE="$PALI_RE" \
    GR00T_DUQUANT_ROT_MODE=svd_hadamard \
    OUTPUT_ROOT=$QUANTVLA_ROOT/results/eval/pi05_${SUITE} \
    bash $QUANTVLA_ROOT/scripts/run_pi05_libero_benchmark.sh
```

---

## 6. Layout

- `gr00t/quantization/` — `quant.py` (`enable_quant_if_configured`, the single
  dispatch/compose entry), `gptq_layers.py` (`GptqLinear` + GPTQ solver +
  per-step), `duquant_layers.py` (rotation + RTN runtime), `rtn_layers.py` (pure RTN).
- `tools/build_gptq_weights.py` — LLM GPTQ pack builder (`--duquant-rotation`).
- `tools/build_dit_a2lite_svd_gptq_perstep.py` — DiT pack builder (svd_hadamard,
  `--use-rtn`, per-step).
- `tools/merge_packs.py` — merge disjoint packs into one.
- `scripts/run_groot_benchmark.sh` / `run_pi05_libero_benchmark.sh` — eval.

## 7. Pitfalls

- `CONDA_ROOT` must be exported (no machine-specific fallback in the scripts).
- `LIBERO_CONFIG_PATH` must be a writable dir with the LIBERO suite YAMLs.
- pi0.5 small-dim heads (`state_proj`/`action_in_proj`/`action_out_proj`/`time_mlp`)
  collapse under A4 — keep them out of the include regex.
- Short-suite (50-episode) success rate has ±5–10pp single-seed variance —
  average over seeds/suites before calling a method-level win.
