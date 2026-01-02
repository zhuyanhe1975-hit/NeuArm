from __future__ import annotations

import argparse
import os
import sys
import warnings

# Silence a known PyTorch 2.8+ warning triggered inside rsl_rl (harmless).
# Set NEUARM_SHOW_PYTORCH_WARNINGS=1 to show it.
if os.getenv("NEUARM_SHOW_PYTORCH_WARNINGS", "0") != "1":
  warnings.filterwarnings(
    "ignore",
    message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
    category=UserWarning,
  )

from dataclasses import asdict
from pathlib import Path


import re


def _find_latest_checkpoint(experiment_root: Path) -> str | None:
  latest_ptr = experiment_root / "_latest.txt"
  if not latest_ptr.exists():
    return None
  run_dir = Path(latest_ptr.read_text(encoding="utf-8").strip())
  if not run_dir.exists():
    return None
  best_it = -1
  best_path: Path | None = None
  for p in run_dir.glob("model_*.pt"):
    m = re.match(r"model_(\d+)\.pt$", p.name)
    if not m:
      continue
    it = int(m.group(1))
    if it > best_it:
      best_it = it
      best_path = p
  return str(best_path) if best_path is not None else None




def _safe_stem(path_str: str | None) -> str:
  if not path_str:
    return "no_ckpt"
  p = Path(path_str)
  return p.stem.replace(".", "_")


def _save_eval_plots(
  *,
  out_dir: Path,
  dt_s: float,
  joint_err_rad: "np.ndarray",
  action: "np.ndarray" | None,
  dpi: int,
  title_prefix: str,
) -> dict[str, str]:
  """Save diagnostic plots (joint error, optionally action) and return artifact paths."""
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
  except Exception as e:
    print(f"[WARN] matplotlib not available; skipping plots: {e}")
    return {}

  out_dir.mkdir(parents=True, exist_ok=True)

  T = int(joint_err_rad.shape[0])
  t = (np.arange(T, dtype=np.float32) * float(dt_s))

  # Joint error figure.
  fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
  axes = axes.reshape(-1)
  for i, ax in enumerate(axes[: joint_err_rad.shape[1]]):
    ax.plot(t, joint_err_rad[:, i], linewidth=1.0)
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_title(f"j{i+1} error (rad)")
    ax.grid(True, alpha=0.3)
  for ax in axes:
    ax.set_xlabel("time (s)")
  fig.suptitle(f"{title_prefix} joint tracking error")
  fig.tight_layout(rect=(0, 0, 1, 0.96))

  joint_png = out_dir / "joint_error.png"
  fig.savefig(joint_png, dpi=int(dpi))
  plt.close(fig)

  artifacts: dict[str, str] = {"joint_error_png": str(joint_png)}

  # Action and action-rate figure (optional but useful for spotting jitter).
  if action is not None and action.size > 0:
    da = np.diff(action, axis=0, prepend=action[:1])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, action, linewidth=0.8)
    axes[0].set_title("policy action (residual) per joint")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, da / max(float(dt_s), 1e-6), linewidth=0.8)
    axes[1].set_title("action rate (approx, per joint)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{title_prefix} policy action")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    action_png = out_dir / "action.png"
    fig.savefig(action_png, dpi=int(dpi))
    plt.close(fig)
    artifacts["action_png"] = str(action_png)

  return artifacts
def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to rsl_rl model_*.pt")
  parser.add_argument("--device", type=str, default="auto", help="cpu | cuda:0 | auto")
  parser.add_argument("--site", type=str, default="tcp", help="Site name to evaluate: tcp | ee")
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--plots", action="store_true", help="Save eval time-series plots to _eval/plots")
  parser.add_argument("--no-plots", dest="plots", action="store_false", help="Disable saving plots")
  parser.set_defaults(plots=True)
  parser.add_argument("--plot-dpi", type=int, default=160)
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
  from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
  )

  from irb2400_rl.task.env_cfg import (
    Irb2400TrackingTaskParams,
    make_irb2400_tracking_env_cfg,
  )

  if args.device == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  experiment_root = repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking"

  env_cfg = make_irb2400_tracking_env_cfg(
    params=Irb2400TrackingTaskParams(num_envs=1),
    play=True,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=1.0)

  dt_s = float(getattr(env_cfg.sim.mujoco, "timestep", 0.001)) * float(getattr(env_cfg, "decimation", 1))

  policy = None
  resolved_checkpoint = args.checkpoint
  if resolved_checkpoint is None:
    resolved_checkpoint = _find_latest_checkpoint(experiment_root)
  if resolved_checkpoint is not None:
    agent_cfg = RslRlOnPolicyRunnerCfg(
      policy=RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(256, 256, 128),
        critic_hidden_dims=(256, 256, 128),
        activation="elu",
      ),
      algorithm=RslRlPpoAlgorithmCfg(
        learning_rate=3.0e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        entropy_coef=0.005,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        schedule="adaptive",
      ),
      experiment_name="neuarm_irb2400_tracking",
      logger="tensorboard",
    )
    eval_log_dir = repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking" / "_eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(vec_env, asdict(agent_cfg), str(eval_log_dir), device=device)
    try:
      runner.load(resolved_checkpoint, load_optimizer=False)
      policy = runner.get_inference_policy(device=device)
    except Exception as e:
      print(f"[WARN] failed to load checkpoint {resolved_checkpoint}: {e}")
      policy = None

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
  actions = []
  with torch.no_grad():
    for _ in range(args.steps):
      if policy is None:
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
      else:
        # rsl_rl policy expects policy obs only.
        try:
          action = policy(obs["policy"])
        except Exception:
          # Some rsl_rl builds expect the full observation dict.
          action = policy(obs)

      obs, _, _, _, _ = env.step(action)

      actions.append(action.squeeze(0).detach().cpu().numpy())

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

  artifacts: dict[str, str] = {}
  if args.plots:
    plot_dir = (repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking" / "_eval" / "plots" / f"{_safe_stem(resolved_checkpoint)}_{args.site}_{args.steps}")
    joint_err_np = np.asarray(joint_abs_err, dtype=np.float32)
    action_np = np.asarray(actions, dtype=np.float32) if len(actions) else None
    artifacts = _save_eval_plots(out_dir=plot_dir, dt_s=dt_s, joint_err_rad=joint_err_np, action=action_np, dpi=args.plot_dpi, title_prefix=_safe_stem(resolved_checkpoint))

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
      checkpoint=resolved_checkpoint,
      site=args.site,
      metrics=metrics,
      extra={"artifacts": artifacts} if artifacts else None,
    )


if __name__ == "__main__":
  main()
