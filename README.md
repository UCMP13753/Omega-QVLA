# Omega-QVLA — W4A4 Quantization for VLA Action Heads

Reproducing the GR00T-N1.5 and pi0.5 W4A4 LIBERO experiments. Core recipe:
**DuQuant block rotation with SVD-Hadamard composition**, optional GPTQ on
either side, optional per-DiT-step activation scale on the diffusion head.

This README is the single entry point — all paper-final code paths live in
`tools/`, `scripts/`, and `gr00t/quantization/`.

---

## 1. Requirements

- 1× NVIDIA A100 (40GB) minimum for build; 4–8× recommended for parallel
  multi-suite eval.
- Python 3.10, CUDA 12.4, PyTorch 2.5.1.
- Conda — we use two envs:
  - **`custon_asr`** — gr00t + LIBERO eval (PyTorch).
  - **`openpi`** — pi0.5 build/eval (separate venv under `openpi/.venv/`).
- Disk: ~10 GB per gr00t pack, ~5 GB per pi0.5 pack.

---

## 2. Setup

```bash
git clone <repo-url> aspq
cd aspq

# gr00t env (PyTorch)
conda create -n custon_asr python=3.10 -y
conda activate custon_asr
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -e .                 # installs the gr00t package

# pi0.5 env: follow https://github.com/Physical-Intelligence/openpi setup,
# then convert the JAX checkpoint to PyTorch (model.safetensors) — see §5.
```

### LIBERO benchmark
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $HOME/LIBERO
cd $HOME/LIBERO && pip install -e .
```

### Checkpoints
GR00T-N1.5 LIBERO-finetuned (4 suites):
```bash
for s in goal spatial object long; do
    huggingface-cli download youliangtan/gr00t-n1.5-libero-${s}-posttrain \
        --local-dir $HOME/ckpts/gr00t-n1.5-libero-${s}-posttrain
done
```

pi0.5 LIBERO: convert `gs://openpi-assets/checkpoints/pi05_libero` to a
PyTorch directory containing `model.safetensors` (use openpi's converter),
then ensure the `assets/` dir is copied in next to it (norm_stats lookup).

### Environment variables (every shell)
```bash
export QUANTVLA_ROOT=$(pwd)/aspq
export QUANTVLA_CACHE_ROOT=$HOME/.cache/quantvla
export QUANTVLA_CONDA_ENV=custon_asr
export GROOT_CONDA_ENV=custon_asr
export LIBERO_CONDA_ENV=custon_asr
export CONDA_ROOT=$HOME/miniconda3
export LIBERO_ROOT=$HOME/LIBERO
export LIBERO_CONFIG_PATH=$HOME/.libero
export CHECKPOINTS_ROOT=$HOME/ckpts
mkdir -p $QUANTVLA_CACHE_ROOT $LIBERO_CONFIG_PATH
```

---

## 3. GR00T-N1.5 — Paper-final W4A4 (E2 recipe)

**Recipe.** LLM side (Eagle): runtime DuQuant A2-lite (SVD-Hadamard
rotation + RTN, no pack). DiT side (action head): offline pack with
A2-lite rotation + RTN + per-DiT-step `act_scale_table`.

### 3.1 Build the DiT pack (per suite)

```bash
SUITE=object                     # object | spatial | goal | long
CKPT=$CHECKPOINTS_ROOT/gr00t-n1.5-libero-${SUITE}-posttrain
case "$SUITE" in
    goal) TASK=libero_goal;    DCFG=examples.Libero.custom_data_config:LiberoDataConfigMeanStd ;;
    long) TASK=libero_10;      DCFG=examples.Libero.custom_data_config:LiberoDataConfig ;;
    *)    TASK=libero_${SUITE}; DCFG=examples.Libero.custom_data_config:LiberoDataConfig ;;
esac

INCLUDE_DIT='.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*'
EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$QUANTVLA_ROOT \
python -m tools.build_dit_a2lite_svd_gptq_perstep \
    --checkpoint "$CKPT" \
    --task-suite-name "$TASK" \
    --data-config "$DCFG" \
    --output-path results/multisuite_packs/${SUITE}_DiT_a2lite_RTN_perstep_cal10_damp05/quantized.pt \
    --include-regex "$INCLUDE_DIT" --exclude-regex "$EXCLUDE" \
    --num-samples 10 --token-cap 1024 --num-steps 8 \
    --svd-rank 0 --use-rtn \
    --w-bits 4 --a-bits 4 --act-percentile 99.9 \
    --duquant-block-size 64 --duquant-block-out 64 \
    --gptq-block-size 128 --gptq-damp-percent 0.05
```

`--svd-rank 0 --use-rtn` selects the A2-lite + RTN recipe (no SVD lowrank head,
no GPTQ error compensation). To use GPTQ instead of RTN on DiT, drop `--use-rtn`.

Repeat for all four suites (~10 min/suite on A100).

### 3.2 Evaluate (per suite, fair init_offset=10)

```bash
SUITE=object
PACK=$QUANTVLA_ROOT/results/multisuite_packs/${SUITE}_DiT_a2lite_RTN_perstep_cal10_damp05/quantized.pt
CKPT=$CHECKPOINTS_ROOT/gr00t-n1.5-libero-${SUITE}-posttrain

INCLUDE_DIT='.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*'
INCLUDE_LLM='.*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'

env \
    CHECKPOINT="$CKPT" MODEL_PATH="$CKPT" \
    SUITE=$SUITE VARIANT=with_dit TASK_SUITE=libero_${SUITE} DATA_CONFIG="$DCFG" \
    METHOD=aspq_gptq WBITS=4 ABITS=4 \
    GPU_LIST=0,1,2,3 PORT_BASE=8000 NUM_TRIALS_PER_TASK=10 \
    GR00T_EVAL_INIT_OFFSET=10 \
    GR00T_HYBRID_QUANT=1 \
    GR00T_ASPQ_GPTQ_PATH_OVERRIDE="$PACK" \
    GR00T_ASPQ_GPTQ_INCLUDE_OVERRIDE="$INCLUDE_DIT" \
    GR00T_ASPQ_GPTQ_EXCLUDE_OVERRIDE="$EXCLUDE" \
    GR00T_ASPQ_GPTQ_MISSING=fallback \
    GR00T_DUQUANT_INCLUDE="$INCLUDE_LLM" GR00T_DUQUANT_EXCLUDE="$EXCLUDE" \
    GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64 \
    GR00T_DUQUANT_PERMUTE=1 GR00T_DUQUANT_ROW_ROT=restore \
    GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32 \
    GR00T_DUQUANT_LS=0.15 \
    GR00T_DUQUANT_PERM_SCORE=weight \
    GR00T_DUQUANT_ROT_MODE=svd_hadamard \
    GR00T_DUQUANT_ACT_SCALE_MODE=percentile \
    OUTPUT_ROOT=$QUANTVLA_ROOT/results/E2/${SUITE}_off10_10tr \
    BENCHMARK_LABEL=E2_${SUITE} \
    bash $QUANTVLA_ROOT/scripts/run_custon_asr_4method_bench.sh
```

Read the merged success rate:
```bash
python -c "import json; print(round(100*json.load(open('results/E2/${SUITE}_off10_10tr/merged_summary.json'))['total_success_rate'],1))"
```

For `long` use `GPU_LIST=0,1,2,3,4,5,6,7` (8 shards) — ~3 h/suite vs ~30 min
for the other three.

---

## 4. pi0.5 — Paper-final W4A4

**Recipe.** PaliGemma backbone: runtime DuQuant A2-lite (no pack). Expert
side (action head): offline pack with A2-lite rotation + GPTQ + per-step
`act_scale_table`.

### 4.1 Record LIBERO observations (one-off, per suite)

The build needs activation samples; record them with the FP16 policy:
```bash
SUITE=object                              # object | spatial | goal | 10
$OPENPI_ROOT/.venv/bin/python -m tools.record_libero_obs_for_pi05 \
    --suite libero_${SUITE} \
    --num-samples 10 \
    --output duquant_act_stats/pi05_libero_${SUITE}_obs.pt
```
(If `tools.record_libero_obs_for_pi05` is missing, run a single FP16 rollout
with `METHOD=fp16` in §4.3 and dump activations via the policy's pre-DiT
hook — see `tools/dump_pi05_xw_for_quanterr.py` for the dump pattern.)

### 4.2 Build the Expert pack (per suite)

```bash
SUITE=object
CKPT=$CHECKPOINTS_ROOT/pi05_libero_pytorch

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$QUANTVLA_ROOT \
$OPENPI_ROOT/.venv/bin/python -m tools.build_pi05_a2lite_gptq_perstep \
    --checkpoint "$CKPT" \
    --data-config pi05_libero \
    --obs-path duquant_act_stats/pi05_libero_${SUITE}_obs.pt \
    --output results/multisuite_packs/pi05_${SUITE}_a2lite_gptq_perstep_cal10/quantized.pt \
    --max-samples 10 --token-cap 512 --num-steps 10 \
    --w-bits 4 --a-bits 4 --act-percentile 99.9 \
    --duquant-block-size 64 --duquant-block-out 64 \
    --gptq-block-size 128 --gptq-damp-percent 0.05
```

### 4.3 Evaluate

```bash
SUITE=object
PACK=$QUANTVLA_ROOT/results/multisuite_packs/pi05_${SUITE}_a2lite_gptq_perstep_cal10/quantized.pt
case "$SUITE" in 10) TASK=libero_10 ;; *) TASK=libero_${SUITE} ;; esac

# include only the paligemma/expert transformer linears — NOT state_proj /
# action_in_proj / action_out_proj / time_mlp (small dims, A4 destroys them)
INCLUDE_EXPERT_PALI='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'

env \
    METHOD=aspq_gptq SUITE=$SUITE WBITS=4 ABITS=4 \
    GPU_LIST=0,1,2,3 PORT_BASE=8100 NUM_TRIALS_PER_TASK=10 \
    GR00T_EVAL_INIT_OFFSET=10 \
    OPENPI_ROOT=$HOME/openpi \
    OPENPI_PY=$HOME/openpi/.venv/bin/python \
    OPENPI_CONFIG=pi05_libero \
    OPENPI_CHECKPOINT=$CHECKPOINTS_ROOT/pi05_libero_pytorch \
    OPENPI_ASPQ_GPTQ_PATH="$PACK" \
    OPENPI_ASPQ_GPTQ_INCLUDE="$INCLUDE_EXPERT_PALI" \
    OUTPUT_ROOT=$QUANTVLA_ROOT/results/pi05_libero/E2_${SUITE} \
    bash $QUANTVLA_ROOT/scripts/run_pi05_libero_benchmark.sh
```

**Important** (override default include): `run_pi05_libero_benchmark.sh`'s
default `OPENPI_INCLUDE_DEFAULT` wraps `state_proj` / `action_in_proj` /
`action_out_proj` / `time_mlp_*` — those small-dim heads collapse under A4
and produce 0% success. The `OPENPI_ASPQ_GPTQ_INCLUDE` override above
restricts to PaliGemma + Expert transformer linears only.

---

## 5. SVDQuant-naive baseline (paper comparison)

We keep a paper-style SVDQuant-naive baseline (SmoothQuant α=0.5 + SVD-r16
lowrank head + RTN residual, per-channel static activation scale) for the
comparison table.

```bash
# Build all 3 short suites in parallel (~10 min each), then eval shard-8 each.
bash scripts/attn_iter/run_svdqNaive_W4A4_shorts.sh     # W4A4
bash scripts/attn_iter/run_svdqNaive_W4A8_shorts.sh     # W4A8
bash scripts/attn_iter/run_smoothquantNaive_W4A4_shorts.sh   # α=0.5 + RTN, no SVD head
```

Each script orchestrates Stage 1 (build packs on GPUs 0/1/2) and Stage 2
(eval shard-8 on GPUs 0–7) and prints the merged short-suite (object /
spatial / goal) success rates at the end.

---

## 6. Aggregate results

```bash
python -c "
import json, glob
for f in sorted(glob.glob('results/E2/*_off10_10tr/merged_summary.json')):
    d=json.load(open(f))
    suite=f.split('/')[-2].split('_')[0]
    print(f'{suite}: {100*d[\"total_success_rate\"]:.1f}%')
"
```

---

## 7. Layout

- `gr00t/quantization/` — runtime DuQuant + ASPQ-GPTQ layers, SVD-Hadamard
  rotation construction, DiT-step context for per-step activation scales.
- `tools/build_dit_a2lite_svd_gptq_perstep.py` — canonical gr00t DiT pack
  builder (A2-lite + optional GPTQ + per-step).
- `tools/build_pi05_a2lite_gptq_perstep.py` — pi0.5 Expert pack builder.
- `tools/build_dit_svdquant_weights.py` /
  `tools/build_pi05_svdquant_weights.py` — SVDQuant-naive baseline builders.
- `tools/build_e2_noperstep.py` — collapses an E2 pack's `act_scale_table`
  to a single bucket (per-step ablation control).
- `tools/diagnose_svd_vs_svdh_*.py`, `tools/plot_*.py` — paper figures.
- `scripts/run_custon_asr_4method_bench.sh` — gr00t LIBERO eval dispatcher.
- `scripts/run_pi05_libero_benchmark.sh` — pi0.5 LIBERO eval dispatcher.
- `scripts/attn_iter/` — SVDQ-naive baseline orchestrators.

---

## 8. Common pitfalls

- **DuQuant device sync after model load**: `enable_duquant_if_configured`
  leaves CPU buffers on a GPU-resident model; the eval scripts call
  `model.to(device)` again after wrap.
- **pi0.5 small-dim heads under A4**: the eval default include regex wraps
  `state_proj` / `action_in_proj` / `action_out_proj` / `time_mlp_*`; A4
  quantization of these tiny linears collapses success to 0%. Always
  override `OPENPI_ASPQ_GPTQ_INCLUDE` to PaliGemma + Expert layers only.
- **paligemma rotation runtime cost**: paligemma `down_proj` has
  `in_features=16384`; a dense in×in rotation costs ~1 GB/layer. Packs
  must include `duquant_rotation_blocks` + `duquant_rotation_perm` (block
  bmm fast path) — the builders save these by default.
- **LIBERO config path**: `LIBERO_CONFIG_PATH` must point to a writable
  directory containing the LIBERO suite YAMLs (the default in
  `scripts/common_paths.sh` may point elsewhere; `$HOME/.libero` is safe).
- **Single-suite ±5–10pp noise**: short-suite (50-episode) success rate
  has ±5–10pp single-seed variance. Don't conclude a method-level win
  from a single run — average over seeds or suites.
