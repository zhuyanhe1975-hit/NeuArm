#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-500}"

source /home/yhzhu/miniconda3/etc/profile.d/conda.sh
conda activate mjwarp_env

python3 "$REPO_ROOT/scripts/train_irb2400_ppo.py" \
  --seed 0 \
  --device "$DEVICE" \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITERS" \
  --action-mode gain_sched \
  --no-friction \
  --track-q-std 0.25 \
  --kp-delta-max 0.15 \
  --kd-delta-max 0.15 \
  --gain-ramp-steps 500000 \
  --gain-filter-tau 0.03 \
  --num-steps-per-env 32 \
  --action-l2-weight -1e-2 \
  --action-rate-weight -5e-3 \
  --run-name "repro_damping0_model100"
