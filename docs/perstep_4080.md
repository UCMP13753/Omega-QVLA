# Per-step scaling experiments on RTX 4080

All three backends are experimental and opt-in via **converted packs**. Existing
packs and default inference retain their original behavior. There are no measured
4080 speedups or LIBERO scores in this change; run the commands below to obtain
them on your GPU. Kernels target CUDA/SM89 and use Triton 3.1 / PyTorch 2.5.1.

| Backend | Activation | Stored residual weight | Dot operation |
|---|---|---|---|
| `fused_float` | Original per-step, per-channel A4 | Shared packed W4 | FP16 operands, FP32 accumulation; dequantization inside kernel |
| `step_weight` | Original per-step, per-channel A4 integer codes | Separate packed W4 for `W * s[t]` at each step | INT8 dot after unpacking INT4, scaled partial sums |
| `group_int` | Per-step scale shared within each K group | Shared packed W4 | INT8 dot after unpacking INT4, scaled partial sums |

These are **not native INT4 MMA kernels**. The activation codes are calculated
inside each output tile; this repeats quantization work, so speedup must be
measured. Both integer paths apply group scales to each group's INT32 dot before
FP32 accumulation. They do not incorrectly move K-channel scales to one final
epilogue. Full-model inference retains input/output rotations, optional low-rank
correction, bias, and output dtype through `GptqLinear`.

Legacy dense quantized weights do not contain their original codes/scales.
Export re-quantizes them using a per-output/per-K-group MSE clipping search. This
adds error even for `fused_float`; it is not a bitwise-equivalent packing claim.
`step_weight` has T normal banks **plus one bank for mean-scale fallback**, so
its payload is roughly T+1 times the other packed residual weights, not the
same footprint. Rotations, unchanged layers and metadata affect total pack size.

## Minimal run: all layers and all steps, no model environment required

Use a **Linux** 4080 host and an original W4A4 pack containing
`act_scale_table`. A DiT-only pack is sufficient for microbenchmarks. A merged
backbone+DiT pack is required for the full LIBERO command below.

```bash
git fetch origin feat/perstep-scaling-backends
git switch feat/perstep-scaling-backends

# If no working torch/triton environment exists, create a dedicated one first.
python3 -m venv .venv-perstep-4080
source .venv-perstep-4080/bin/activate
SETUP=1 bash scripts/run_perstep_4080.sh /absolute/path/to/original/quantized.pt
```

For an existing working environment, leave out `SETUP=1` and the venv creation.
The setup installs pinned PyTorch/Triton plus NumPy/pytest, not the full VLA
dependencies. It does not alter model checkpoints.

The script runs tests (including GPU kernel/oracle checks), exports all three
packs, and benchmarks every per-step layer, every saved step, at 32 and 128
tokens. Other dimensions are available through the Python CLI. Without a
checkpoint/capture, inputs are scale-matched random data. Those results are only
kernel validation and throughput smoke tests, **not model accuracy evidence**.

## Real activation drift and task success

Activate your existing `omega_qvla` environment with LIBERO, checkpoints and the
repository's normal environment variables configured (`CONDA_ROOT`,
`QUANTVLA_CONDA_ENV`, `LIBERO_ROOT`, etc.). Install Triton/pytest if missing. Then:

```bash
CHECKPOINT=/absolute/path/to/gr00t-n1.5-libero-object-posttrain \
SUITE=object RUN_LIBERO=1 NUM_TRIALS_PER_TASK=10 \
bash scripts/run_perstep_4080.sh /absolute/path/to/original/MERGED/quantized.pt
```

This captures raw layer inputs under the original quantized policy, from init
state offset 0 for group calibration and offset 10 for held-out residual replay.
Captures accumulate up to 256 tokens per layer/step. The group method uses a
pooled q99.9 across the calibration rows/channels within each group, not an
average of channel percentiles. Calibration and evaluation windows must not
overlap. These are modest local diagnostics; LIBERO rollouts below measure task
success separately. Capture/replay requires the **GR00T** model environment;
pi0.5's separate runtime is not integrated by this change.

After the microbenchmark, original baseline + all three converted packs run
sequentially on one visible GPU through `run_groot_benchmark.sh`, on identical
trial settings (initial-state offset 10 by default). The baseline and other backbone layers are unchanged. With
`CHECKPOINT` set and `RUN_LIBERO` unset, only real-input capture/replay is run.
The checkpoint and suite must match the original pack. Use at least 20 available
LIBERO initial states per task for the default capture split.

## Outputs and interpretation

- `manifest.json`: revision, GPU/software, source/converted file sizes and
  geometric mean **residual-only** speedup, mean/max residual output drift.
- `benchmark.csv` and incremental `benchmark.jsonl`: layer, step, actual/requested
  token counts, input type/source, re-packed weight NMSE, kernel/oracle NMSE,
  output NMSE vs original fake-quant residual, median CUDA-event latency and
  additional allocation peak. Warmup/JIT compilation/export are excluded.
- `fused_float.pt`, `step_weight.pt`, `group_int.pt`: load with the existing GPTQ
  pack mechanism. Default engine is Triton; `GR00T_PERSTEP_ENGINE=torch_reference`
  explicitly opts into the slow, unfused debugging oracle.
- Adjacent `<output>_libero/<backend>/`: existing rollout logs/results for the
  baseline and three variants when `RUN_LIBERO=1`.

Local errors compare against the **existing quantized pack**, not FP accuracy.
Kernel latency excludes rotations, low-rank correction and bias, so it is not
end-to-end policy latency. Extra allocation peak excludes already resident
weights; the CSV separately reports backend resident tensor bytes. End-to-end
success and runtime must be read from the rollout evaluation separately. Never
report an export-only CPU run or skipped GPU tests as 4080 execution.

Each run uses a fresh timestamped output directory and refuses to mix results
with an existing run. Optional environment controls: `OUT`, `PYTHON`,
`CUDA_VISIBLE_DEVICES` (one device only), `GROUP_SIZE`, `WARMUP`, `REPEATS`.

For narrower debugging or CPU pack export:

```bash
python -m tools.compare_perstep_backends --pack /path/to/quantized.pt \
  --output results/perstep_debug --layer action_head.model.transformer_blocks.0.attn1.to_q \
  --tokens 32 --warmup 5 --repeats 10

python -m tools.compare_perstep_backends --pack /path/to/quantized.pt \
  --output results/perstep_cpu_export --export-only --device cpu
```

CPU export is algebra/storage validation only. No automatic CPU fallback is
used when Triton/CUDA is unavailable.

Development validation (no GPU available): 9 CPU algebra/export/runtime tests
passed; 12 Triton interpreter numerical cases passed (CUDA nearbyint shimmed
with NumPy rint); all 36 combinations of three backends, FP16/BF16/FP32 input,
and group sizes 32/64/128/256 compiled for SM89 with the intended MMA operand
types. The 12 actual-CUDA test configurations remain unexecuted locally and
must pass on the 4080 before benchmark timings are collected.
