from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to rsl_rl model_*.pt")
  parser.add_argument("--device", type=str, default="auto", help="cpu | cuda:0 | auto")
  parser.add_argument("--site", type=str, default="tcp", help="Site name to evaluate: tcp | ee")
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--no-record", action="store_true", help="Disable writing eval_history.jsonl")
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root / "src"))

  from irb2400_rl.mjlab_bootstrap import ensure_mjlab_on_path
  from run_record import record_eval

  ensure_mjlab_on_path()
  os.environ.setdefault("MUJOCO_GL", "egl")

  import numpy as np
  import torch
  from rsl_rl.runners import OnPolicyRunner

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

  from irb2400_rl.task.env_cfg import (
    Irb2400TrackingTaskParams,
    make_irb2400_tracking_env_cfg,
  )

  if args.device == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  env_cfg = make_irb2400_tracking_env_cfg(
    params=Irb2400TrackingTaskParams(num_envs=1),
    play=True,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=1.0)

  policy = None
  if args.checkpoint is not None:
    agent_cfg = RslRlOnPolicyRunnerCfg(
      experiment_name="neuarm_irb2400_tracking",
      logger="tensorboard",
    )
    eval_log_dir = repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking" / "_eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(vec_env, asdict(agent_cfg), str(eval_log_dir), device=device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  site_pat = rf".*{args.site}$"
  site_ids, site_names = robot.find_sites((site_pat,), preserve_order=True)
  if len(site_ids) != 1:
    # Backward-compatible fallback.
    site_ids, site_names = robot.find_sites((r".*ee$",), preserve_order=True)
    if len(site_ids) != 1:
      raise RuntimeError(f"Expected exactly 1 site for {args.site} (or ee fallback), got {site_names}")
  ee_site_local = site_ids[0]
  ee_site_global = int(robot.data.indexing.site_ids[ee_site_local].item())

  num_joints = robot.data.joint_pos.shape[1]
  qpos_adr = robot.data.indexing.joint_q_adr.to(dtype=torch.long)

  # Use a separate MuJoCo CPU forward pass for desired FK to avoid mutating the
  # MJWarp simulation state (and to avoid CUDA graph pitfalls).
  import mujoco

  mj_model = env.sim.mj_model
  mj_data_des = mujoco.MjData(mj_model)

  obs, _ = env.reset()

  ee_errs_mm = []
  joint_rmse_rad = []
  joint_abs_err = []
  with torch.no_grad():
    for _ in range(args.steps):
      if policy is None:
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
      else:
        # rsl_rl policy expects policy obs only.
        action = policy(obs["policy"])

      obs, _, _, _, _ = env.step(action)

      cmd = env.command_manager.get_command("traj")
      q_des = cmd[:, 0:num_joints]
      q = robot.data.joint_pos[:, 0:num_joints]
      joint_rmse_rad.append(float(torch.sqrt(torch.mean((q_des - q) ** 2)).item()))
      joint_abs_err.append((q_des - q).abs().squeeze(0).detach().cpu().numpy())

      # Current EE pos.
      ee_pos = robot.data.site_pos_w[:, ee_site_local]

      # Desired EE pos via CPU MuJoCo forward at q_des.
      mj_data_des.qpos[:] = 0.0
      mj_data_des.qvel[:] = 0.0
      mj_data_des.qpos[qpos_adr.cpu().numpy()] = q_des[0].detach().cpu().numpy()
      mujoco.mj_forward(mj_model, mj_data_des)
      ee_des_np = mj_data_des.site_xpos[ee_site_global].copy()
      ee_des = torch.tensor(ee_des_np, device=env.device, dtype=torch.float32).unsqueeze(0)

      err_mm = torch.linalg.vector_norm(ee_des - ee_pos, dim=-1) * 1000.0
      ee_errs_mm.append(float(err_mm.item()))

  env.close()

  ee_errs_mm_np = np.asarray(ee_errs_mm, dtype=np.float32)
  joint_rmse_rad_np = np.asarray(joint_rmse_rad, dtype=np.float32)
  joint_abs_err_np = np.asarray(joint_abs_err, dtype=np.float32)  # (T, 6)
  print(f"Joint RMSE (rad): mean={joint_rmse_rad_np.mean():.4f}  p95={np.percentile(joint_rmse_rad_np,95):.4f}  max={joint_rmse_rad_np.max():.4f}")
  per_joint_rmse = np.sqrt(np.mean(joint_abs_err_np**2, axis=0))
  print("Per-joint RMSE (rad):", " ".join([f"j{i+1}={per_joint_rmse[i]:.4f}" for i in range(num_joints)]))
  if num_joints >= 3:
    arm_rmse = float(np.sqrt(np.mean(joint_abs_err_np[:, :3] ** 2)))
    wrist_rmse = float(np.sqrt(np.mean(joint_abs_err_np[:, 3:6] ** 2))) if num_joints >= 6 else float("nan")
    print(f"Arm(1-3) RMSE (rad): {arm_rmse:.4f}  Wrist(4-6) RMSE (rad): {wrist_rmse:.4f}")
  print(f"TCP error (mm): mean={ee_errs_mm_np.mean():.3f}  p95={np.percentile(ee_errs_mm_np,95):.3f}  max={ee_errs_mm_np.max():.3f}")

  if not args.no_record:
    metrics = {
      "joint_rmse_rad": {
        "mean": float(joint_rmse_rad_np.mean()),
        "p95": float(np.percentile(joint_rmse_rad_np, 95)),
        "max": float(joint_rmse_rad_np.max()),
      },
      "per_joint_rmse_rad": {f"j{i+1}": float(per_joint_rmse[i]) for i in range(num_joints)},
      "tcp_err_mm": {
        "mean": float(ee_errs_mm_np.mean()),
        "p95": float(np.percentile(ee_errs_mm_np, 95)),
        "max": float(ee_errs_mm_np.max()),
      },
      "steps": int(args.steps),
      "device": str(device),
    }
    record_eval(
      eval_log_dir=repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking" / "_eval",
      checkpoint=args.checkpoint,
      site=args.site,
      metrics=metrics,
    )


if __name__ == "__main__":
  main()
