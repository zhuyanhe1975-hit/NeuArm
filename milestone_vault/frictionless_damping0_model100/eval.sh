#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-2000}"

source /home/yhzhu/miniconda3/etc/profile.d/conda.sh
conda activate mjwarp_env

python3 "$REPO_ROOT/scripts/eval_irb2400_tracking.py" \
  --seed 0 \
  --device "$DEVICE" \
  --checkpoint "$HERE/checkpoints/model_100.pt" \
  --action-mode gain_sched \
  --no-friction \
  --steps "$STEPS" \
  --plots
