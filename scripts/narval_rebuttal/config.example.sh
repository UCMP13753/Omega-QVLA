# Copy to config.local.sh (gitignored) and edit paths/account on Narval.
# Use two existing validated environments. They may be conda env directories or venvs.
export ACCOUNT="YOUR_COMPUTE_CANADA_ACCOUNT"
export GROOT_ENV="$HOME/venvs/omega_qvla"
export PI05_ENV="$HOME/openpi/.venv"
export CHECKPOINTS_ROOT="$SCRATCH/checkpoints"
export OPENPI_CHECKPOINT="$CHECKPOINTS_ROOT/pi05_libero_pytorch"
export LIBERO_ROOT="$HOME/LIBERO"
export OPENPI_ROOT="$HOME/openpi"
export REBUTTAL_ROOT="$SCRATCH/omega-aaai-20261026"
# Existing site environment may need different module versions. Check `module avail`.
# Leave empty when your environments already supply CUDA/runtime libraries.
export REBUTTAL_MODULES=""
export REBUTTAL_MAX_IN_FLIGHT=8
export REBUTTAL_TIME="12:00:00"
export REBUTTAL_MEM="128G"
export REBUTTAL_CPUS=8
