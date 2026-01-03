#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1000}"

source /home/yhzhu/miniconda3/etc/profile.d/conda.sh
conda activate mjwarp_env

python3 "$REPO_ROOT/scripts/train_irb2400_ppo.py" \
  --seed 0 \
  --device "$DEVICE" \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITERS" \
  --action-mode residual \
  --track-q-std 0.05 \
  --track-qd-std 1.5 \
  --residual-scale 10 \
  --residual-clip 10 \
  --residual-ramp-steps 0 \
  --residual-filter-tau 0.03 \
  --action-l2-weight -1e-3 \
  --action-rate-weight -5e-2 \
  --num-steps-per-env 32 \
  --run-name "repro_friction_residual_model999"
