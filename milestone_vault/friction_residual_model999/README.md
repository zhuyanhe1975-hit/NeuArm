# Milestone: friction + residual — model_999

## Snapshot

- Checkpoint: `checkpoints/model_999.pt`
- Source run (original logs): `logs/rsl_rl/neuarm_irb2400_tracking/2026-01-02_22-02-51`
- Eval: TCP site, 2000 steps

## Metrics (eval)

- Joint RMSE (rad): mean `0.002163`, p95 `0.003687`, max `0.003796`
- TCP error (mm): mean `6.245`, p95 `10.417`, max `10.863`
- Residual |tau| (N·m): mean `8.331`, p95 `8.569`, max `8.885`

## Plots

![joint_error](plots/joint_error.png)
![joint_vel_error](plots/joint_vel_error.png)
![torque](plots/torque.png)
![action](plots/action.png)

## One‑click

- Train: `bash train.sh`
- Eval: `bash eval.sh`
- Replay (viewer): `bash play.sh`

