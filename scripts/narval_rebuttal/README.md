# AAAI rebuttal experiments on Narval

This runner adds controlled experiments to Omega-QVLA at base commit
`3727e2203568db43fc5fba06ee8686c1b47c044f`.
It does **not** claim to reproduce the paper's exact success rates or native INT4
latency. Existing `GptqLinear` uses fake quantization followed by BF16/FP32 GEMM.
The new strict reference runner records that fact in every deployment report.

## What will run

The default **core** plan uses two models, all four suites, seeds 42/43/44, and
10 trials per task (100 episodes per suite per run). The seeds govern calibration
noise/token sampling and evaluation. All variants for a given model/suite/seed
reuse exactly the same captured full-precision calibration activations.

| Variant | Backbone | Action head | Purpose |
| --- | --- | --- | --- |
| `fp16` | Original floating model | Original floating model | Reference; actual parameter dtypes are those of the loaded checkpoint, often BF16 |
| `hadamard` | H + GPTQ, W4A4 | H + RTN, W4A4 | Hadamard-only control |
| `svd_hadamard` | SVD-H + GPTQ, W4A4 | SVD-H + RTN, W4A4 | Matched composite reference |
| `backbone_only` | SVD-H + GPTQ, W4A4 | Original floating model | Isolate backbone |
| `head_only` | Original floating model | SVD-H + RTN, W4A4 | Isolate head |
| `gptq_w4a16` | GPTQ W4, original activation dtype | GPTQ W4, original activation dtype | Audit pure GPTQ, with the same target scope |
| `rtn_w4a16` | Symmetric per-output RTN W4 | Symmetric per-output RTN W4 | Simple weight-only control |

Default: **24 captures + 144 builds + 168 evaluations + 14 profiles = 350 jobs**,
**16,800 evaluation episodes**. Jobs each request one GPU; default submission
throttle is eight in flight. The profile jobs use the first seed and last selected
suite (Long in the default plan). They include vision encoding, the backbone,
full denoising and online quantization/rotation/scaling. GR00T timings exclude
external observation normalization; neither model's timings include simulator
steps or websocket transport. No simulator control-frequency claim is made.

Each build also records per-layer weight-only / activation-only / joint output
NMSE, rotation roundoff, input-channel norm ratios, and a module/byte inventory.
These diagnostics use **calibration** inputs, not held-out generalization data.
A Hadamard-only component ablation is not a full reproduction of QuaRot.

### Numerical controls

- Input block size 64 and the same weight-derived zigzag permutation for H and
  SVD-H. No hidden SVD is permitted in Hadamard-only.
- No output-side rotation in **any** of the new variants. The legacy builders add
  an output-side SVD rotation regardless of input mode; using that path would
  confound a genuinely SVD-free Hadamard-only comparison. Both H and SVD-H are
  therefore rerun under the same explicit new reference configuration.
- Backbone GPTQ uses the repository solver, solver block 128 and damping 0.01;
  head RTN uses symmetric per-output min-max scaling. Existing builders' RTN
  residual uses a different block/MSE routine. Treat this as a controlled new
  study rather than attach old headline scores to these new configurations.
- Backbone activation quantization is per token. Head activation scales are
  per input channel, calibrated at percentile 99.9 for each actual step.
- Activation fake quantization precedes floating-point GEMM. Channel scales are
  not incorrectly moved outside the reduction. No native INT4 backend is added.
- Scale tables are generated separately for each denoising schedule. Missing step
  context, out-of-range steps, empty captures or missing target layers cause an
  error; they do not silently fall back to averaged scales or FP layers.
- Calibration uses one stabilized reset observation per selected task at initial
  state index 0 (10 observations across 10 tasks in the main configuration),
  rather than ten complete trajectories. Evaluation uses initial states 20–29.
  Episodes are paired across variants by task, initial-state index and seed.
- Runtime errors and nonfinite actions fail the job; they are not silently counted
  as manipulation failures. Report code checks actual denominators.
- Serialized packs contain dequantized floating tensors. Their size is **not** a
  packed-INT4 footprint. Inventory reports distinguish tensor bytes, metadata,
  unquantized parameters, and explicitly incomplete logical weight-bit estimates.

## 1. Install / use environments on Narval

Use existing environments if possible. GPU model execution has not been verified
in this cloud workspace; a GPU preflight and smoke run are mandatory before the
full submission. No jobs have been submitted by the authoring session.

The repository's historical `environments/requirements_*.txt` are machine-specific
freezes with stale editable Git URLs and mixed torch/CUDA versions. Do not blindly
install those freezes into a working Narval environment.

You need:

- GR00T environment: Python 3.10/3.11 compatible with this repository, torch and
  torchvision matched to the available CUDA runtime, repository dependencies,
  LIBERO, robosuite, MuJoCo, EGL, and the original model dependencies (including
  flash-attn where required by the checkpoint). The README reference environment
  uses torch 2.5.1; use compatible Narval-provided wheels/modules.
- OpenPI environment: a working, version-pinned PyTorch OpenPI checkout and its
  patched transformers installation, plus LIBERO simulation dependencies. The
  runner imports OpenPI in that environment and applies denoising context at
  runtime without changing OpenPI's source.
- Both environments must see this repository on PYTHONPATH (the wrapper sets it).
  The OpenPI environment also imports the repository's GPTQ solver, which only
  requires torch/numpy from the quantization modules.
- LIBERO's BDDL files, initial-state files and assets must be installed. The code
  constructs calibration observations from LIBERO; no private observation pickle
  is required.
- Four full GR00T checkpoints, and a converted **PyTorch** pi0.5 checkpoint with
  `model.safetensors` and the normalization assets expected by OpenPI.

If setting up fresh, install the dependencies into separate environments following
this repo's README and the pinned OpenPI checkout's own instructions. Download
checkpoints and install packages from a login/data-transfer node before submitting
jobs; workers should have all required weights/tokenizer assets cached locally.
The wrapper supports venv or conda **prefix directories**, not bare conda names.
Do not assume arbitrary current OpenPI main is compatible with the checkpoint.

Copy the sample configuration:

```bash
cd /path/to/Omega-QVLA
cp scripts/narval_rebuttal/config.example.sh scripts/narval_rebuttal/config.local.sh
# Edit ACCOUNT, GROOT_ENV, PI05_ENV, CHECKPOINTS_ROOT, OPENPI_CHECKPOINT,
# LIBERO_ROOT, OPENPI_ROOT and REBUTTAL_ROOT.
```

Expected checkpoint layout:

```text
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-goal-posttrain/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-spatial-posttrain/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-long-posttrain/
$OPENPI_CHECKPOINT/model.safetensors
```

Request a GPU allocation using your account and current Narval resource policy,
then run inside the allocation:

```bash
bash scripts/narval_rebuttal/preflight.sh groot
bash scripts/narval_rebuttal/preflight.sh pi05
```

Preflight checks GPU availability, imports, data assets/initial states and checkpoint
paths. It does not replace a full policy forward smoke run. CUDA/EGL or dependency
failures must be resolved in that environment first. Set REBUTTAL_MODULES in the
local config if module loading is needed. No site-specific account or module
version is guessed.

## 2. Smoke test before the full batch

On the login node, create a small plan with one suite/seed, one episode per task,
and FP/H/SVD-H variants for both models:

```bash
source scripts/narval_rebuttal/config.local.sh
python3 -m tools.rebuttal.plan \
  --root "$REBUTTAL_ROOT/smoke" \
  --checkpoints "$CHECKPOINTS_ROOT" --pi-checkpoint "$OPENPI_CHECKPOINT" \
  --suites object --seeds 42 --smoke

# Show planned jobs first; this does not invoke sbatch.
bash scripts/narval_rebuttal/submit.sh "$REBUTTAL_ROOT/smoke/manifest.json"

# Submit the smoke test.
bash scripts/narval_rebuttal/submit.sh "$REBUTTAL_ROOT/smoke/manifest.json" --submit
```

Smoke: 2 captures, 4 builds, 6 evals, 6 profiles, 60 total evaluation episodes.
Do not proceed until captures have every expected head step, both H and SVD-H
packs build, evaluations finish without implementation errors, and latency traces
produce valid outputs. Initial-state calibration captures still use 10 tasks.

## 3. Submit the core plan

```bash
source scripts/narval_rebuttal/config.local.sh
python3 -m tools.rebuttal.plan \
  --root "$REBUTTAL_ROOT/core" \
  --checkpoints "$CHECKPOINTS_ROOT" --pi-checkpoint "$OPENPI_CHECKPOINT"

bash scripts/narval_rebuttal/submit.sh "$REBUTTAL_ROOT/core/manifest.json"
bash scripts/narval_rebuttal/submit.sh "$REBUTTAL_ROOT/core/manifest.json" --submit
```

Every job has afterok dependencies on its inputs. Submission records each sbatch
ID atomically in `submitted.json`. Repeating the submission command does not
resubmit recorded or completed jobs. Throttling uses afterany lanes so unrelated
job failures do not stop the entire graph. Failed prerequisites still correctly
block their dependent jobs.

The source digest is pinned in the manifest. Editing the runner/runtime or relevant
model code after generating a plan causes workers to refuse the run. Use a new
run directory for changed scientific settings/code. A successful done marker also
checks the manifest digest before reuse.

To retry failed jobs, first inspect their logs, cancel the remaining old jobs
(including DependencyNeverSatisfied dependents), and ensure none remain active.
Then use the **same unchanged code/manifest**:

```bash
bash scripts/narval_rebuttal/submit.sh "$REBUTTAL_ROOT/core/manifest.json" --submit --retry-failed
```

Completed jobs are retained. If code fixes are needed, use a fresh plan/run
instead; do not combine results generated by different code revisions.

## 4. Optional sensitivity experiments

`--extras` includes the core variants plus Long-suite-only controls:

- SVD-only and permutation-only;
- pooled channel scaling versus per-step scaling;
- W4A8;
- 2 calibration observations versus 10;
- input block sizes 32 and 128 versus 64;
- independently calibrated 4-step and 10-step models, each with a matching
  floating-point baseline. This tests recalibration at each schedule, not
  transfer of one scale table across schedules or solvers.

Use `--suites long --extras` to keep this batch small and avoid duplicating other
core suites. A separate run will still recompute Long core controls; do not pool
duplicated controls across core/extras as additional independent repeats.

```bash
python3 -m tools.rebuttal.plan \
  --root "$REBUTTAL_ROOT/sensitivity-long" \
  --checkpoints "$CHECKPOINTS_ROOT" --pi-checkpoint "$OPENPI_CHECKPOINT" \
  --suites long --extras
bash scripts/narval_rebuttal/submit.sh "$REBUTTAL_ROOT/sensitivity-long/manifest.json" --submit
```

## 5. Results and interpretation

```bash
python3 -m tools.rebuttal.report --manifest "$REBUTTAL_ROOT/core/manifest.json"
```

Outputs:

- `report.md`, `report.json`: per-suite seed mean/std, exact numerators and
  denominators, complete four-suite macro averages, and paired saved/spoiled
  outcome counts for composite versus H-only and floating reference.
- `eval/`: individual task/initial-state/seed/success records, including errors.
- `packs/*.diagnostics.json`: per-layer separated reconstruction errors and
  norm ratios; all layers reported, including regressions.
- `packs/*.inventory.json`: actual module scope, original parameter bytes,
  dequantized pack bytes, metadata bytes and explicitly labeled logical estimates.
- `profile/`: synchronized latency, peak allocated/reserved GPU memory, GPU/library
  metadata, and a separate CPU/CUDA operator trace. Profiler overhead is excluded
  from the reported 30 timed iterations after five warmup calls.
- `status/`: completion markers; `slurm/`: stdout/stderr; `submitted.json`: job IDs.

The reporter leaves missing runs missing, instead of treating them as zeros or
silently averaging incomplete suites. Std is across calibration/evaluation runs.
Pooled Wilson intervals are labeled descriptive; task/checkpoint clustering means
episodes are not independent statistical replicates. Paired counts use exactly
matched episode identities.

### What still needs external evidence

This batch does not invent unavailable baselines or hardware support:

- Native W4A4 / W4A8 / W4A16 deployment comparisons require implemented real
  kernels and a validated operator graph. This repository does not supply that
  native backend; floating fake-quant timings cannot resolve that review concern.
- AWQ, OmniQuant, SpinQuant, a full QuaRot reproduction and PTQ4DiT adaptations
  require their actual implementations. Their names are not attached to proxies.
- Published QuantVLA's exact 108/180-layer accounting and the old 0.5% result
  denominators require original checkpoints/logs. Fresh experiments do not explain
  historical table values.
- New robot rollouts and energy/embedded-device tests are outside a Narval GPU job.

## Local validation performed

CPU tests cover orthogonal equivalence (including bias), proof that H-only does
not call SVD, strict per-step dispatch and rejection of schedule mismatch,
OpenPI's cached callable/context reset, exact denominators, paired result
identities, missing-run handling, manifest DAG and matching controls. Run:

```bash
python -m unittest discover -s tests/rebuttal -v
python -m compileall -q tools/rebuttal
bash -n scripts/narval_rebuttal/*.sh scripts/narval_rebuttal/job.sbatch
```

Full GR00T/OpenPI model execution, GPU timing and LIBERO success rates require the
Narval smoke run; they have not been measured in the authoring environment.
