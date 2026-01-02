from __future__ import annotations

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
from datetime import datetime
from pathlib import Path

import argparse


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=1024)
  # Policy/control update = sim_dt(1ms) * decimation(5) = 5ms.
  parser.add_argument("--decimation", type=int, default=5)
  parser.add_argument("--episode-length-s", type=float, default=6.0)
  parser.add_argument("--effort-limit", type=float, default=300.0)
  parser.add_argument("--track-q-std", type=float, default=0.25, help="Std (rad) for track_q reward shaping")
  parser.add_argument("--track-qd-std", type=float, default=1.5, help="Std (rad/s) for track_qd reward shaping")
  parser.add_argument("--preset", type=str, default="", help="Optional preset: fine")
  # Command distribution overrides (stress-test training).
  parser.add_argument("--cmd-sine-freq-lo", type=float, default=None, help="Override sine freq lower bound (Hz)")
  parser.add_argument("--cmd-sine-freq-hi", type=float, default=None, help="Override sine freq upper bound (Hz)")
  parser.add_argument("--cmd-sine-cycles-lo", type=int, default=None, help="Override sine cycles lower bound (int)")
  parser.add_argument("--cmd-sine-cycles-hi", type=int, default=None, help="Override sine cycles upper bound (int)")
  parser.add_argument("--cmd-joint-delta-scale", type=float, default=None, help="Override joint_delta_scale (fraction of joint range)")
  parser.add_argument("--cmd-j6-scale", type=float, default=None, help="Override joint 6 scale inside joint_delta_scale_by_joint (others unchanged)")
  parser.add_argument(
    "--action-mode",
    type=str,
    default="gain_sched",
    choices=("gain_sched", "residual"),
    help="Policy/controller mode: gain_sched (NN outputs Kp/Kd multipliers) or residual (NN outputs residual torques)",
  )
  # Gain-scheduling authority (start small; ramp in slowly).
  parser.add_argument("--kp-delta-max", type=float, default=0.15, help="Max +/- gain multiplier delta for Kp scheduling (action->multiplier)")
  parser.add_argument("--kd-delta-max", type=float, default=0.15, help="Max +/- gain multiplier delta for Kd scheduling (action->multiplier)")
  parser.add_argument("--gain-ramp-steps", type=int, default=500000, help="Ramp-in steps for gain scheduling authority (0 disables)")
  parser.add_argument("--gain-filter-tau", type=float, default=0.03, help="Low-pass filter time constant for gain multipliers (0 disables)")
  # Residual torque authority (used when --action-mode residual; ignored otherwise).
  parser.add_argument("--residual-scale", type=float, default=2.0, help="Residual torque scale (N*m) for action=1")
  parser.add_argument("--residual-clip", type=float, default=5.0, help="Residual torque clip (N*m)")
  parser.add_argument("--residual-ramp-steps", type=int, default=500000, help="Ramp-in steps for residual authority (0 disables)")
  parser.add_argument("--residual-filter-tau", type=float, default=0.03, help="Low-pass filter time constant for residual torque (s, 0 disables)")
  parser.add_argument("--action-l2-weight", type=float, default=-1e-2)
  parser.add_argument("--action-rate-weight", type=float, default=-5e-3)
  parser.add_argument("--device", type=str, default=None, help="cpu | cuda:0 | auto")
  parser.add_argument("--max-iterations", type=int, default=1000)
  parser.add_argument("--num-steps-per-env", type=int, default=32)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--run-name", type=str, default="")
  parser.add_argument("--resume", type=str, default="", help="Resume training from a checkpoint path (model_*.pt)")
  parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest checkpoint under logs/rsl_rl/neuarm_irb2400_tracking")
  parser.add_argument("--resume-no-optimizer", action="store_true", help="When resuming, do not load optimizer state")
  parser.add_argument("--init-noise-std", type=float, default=0.2, help="Initial policy action noise std")
  parser.add_argument("--no-record", action="store_true", help="Disable simple JSONL/stdout run recording")
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root / "src"))

  from irb2400_rl.mjlab_bootstrap import ensure_mjlab_on_path

  from run_record import RslRlStdoutParser, tee_stdout_to_file, update_latest_pointer, write_run_metadata

  ensure_mjlab_on_path()

  os.environ.setdefault("MUJOCO_GL", "egl")

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
    ACTION_TERM_NAME,
    Irb2400TrackingTaskParams,
    make_irb2400_tracking_env_cfg,
  )

  preset = (args.preset or "").strip().lower()
  if preset == "fine":
    args.track_q_std = 0.05
    args.init_noise_std = min(float(args.init_noise_std), 0.10)
    args.action_rate_weight = -2e-3
    args.num_steps_per_env = max(int(args.num_steps_per_env), 64)

  if args.device in (None, "auto"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  # Actions are clipped to [-1, 1] by the vec-env wrapper.
  clip_actions = 1.0

  params = Irb2400TrackingTaskParams(
    num_envs=args.num_envs,
    decimation=args.decimation,
    episode_length_s=args.episode_length_s,
    effort_limit=args.effort_limit,
    action_mode=str(args.action_mode),
  )
  env_cfg = make_irb2400_tracking_env_cfg(params=params)

  # Optional: override command distribution (for stress-tests).
  cmd_cfg = env_cfg.commands.get("traj")
  if cmd_cfg is not None:
    if args.cmd_sine_freq_lo is not None or args.cmd_sine_freq_hi is not None:
      lo, hi = getattr(cmd_cfg, "sine_freq_hz_range", (0.2, 1.0))
      if args.cmd_sine_freq_lo is not None:
        lo = float(args.cmd_sine_freq_lo)
      if args.cmd_sine_freq_hi is not None:
        hi = float(args.cmd_sine_freq_hi)
      cmd_cfg.sine_freq_hz_range = (lo, hi)
    if args.cmd_sine_cycles_lo is not None or args.cmd_sine_cycles_hi is not None:
      lo, hi = getattr(cmd_cfg, "sine_cycles_range", (1, 3))
      if args.cmd_sine_cycles_lo is not None:
        lo = int(args.cmd_sine_cycles_lo)
      if args.cmd_sine_cycles_hi is not None:
        hi = int(args.cmd_sine_cycles_hi)
      cmd_cfg.sine_cycles_range = (lo, hi)
    if args.cmd_joint_delta_scale is not None:
      cmd_cfg.joint_delta_scale = float(args.cmd_joint_delta_scale)
    if args.cmd_j6_scale is not None:
      scales = getattr(cmd_cfg, "joint_delta_scale_by_joint", None)
      if scales is not None and len(scales) == 6:
        cmd_cfg.joint_delta_scale_by_joint = (
          float(scales[0]),
          float(scales[1]),
          float(scales[2]),
          float(scales[3]),
          float(scales[4]),
          float(args.cmd_j6_scale),
        )

  act = env_cfg.actions[ACTION_TERM_NAME]
  if str(args.action_mode).strip().lower() == "gain_sched":
    # Configure gain scheduling action term safely (do not disturb baseline at a=0).
    if hasattr(act, "kp_delta_max"):
      act.kp_delta_max = float(args.kp_delta_max)
      act.kd_delta_max = float(args.kd_delta_max)
      act.gain_ramp_steps = int(args.gain_ramp_steps)
      act.gain_filter_tau = float(args.gain_filter_tau)
  else:
    # Configure residual authority safely (ramp + filter).
    if hasattr(act, "residual_scale"):
      act.residual_scale = float(args.residual_scale)
    if hasattr(act, "residual_clip"):
      act.residual_clip = float(args.residual_clip)
    if hasattr(act, "residual_ramp_steps"):
      act.residual_ramp_steps = int(args.residual_ramp_steps)
    if hasattr(act, "residual_filter_tau"):
      act.residual_filter_tau = float(args.residual_filter_tau)

  # Configure reward weights (action regularization).
  env_cfg.rewards["action_l2"].weight = float(args.action_l2_weight)
  if "action_rate_l2" in env_cfg.rewards:
    env_cfg.rewards["action_rate_l2"].weight = float(args.action_rate_weight)

  # Configure tracking reward scale.
  env_cfg.rewards["track_q"].params["std"] = float(args.track_q_std)
  env_cfg.rewards["track_qd"].params["std"] = float(args.track_qd_std)

  agent_cfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=float(args.init_noise_std),
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
    run_name=args.run_name,
    logger="tensorboard",
    num_steps_per_env=args.num_steps_per_env,
    max_iterations=args.max_iterations,
    save_interval=args.save_interval,
  )

  log_root = repo_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_dir_name += f"_{agent_cfg.run_name}"
  log_dir = log_root / log_dir_name
  log_dir.mkdir(parents=True, exist_ok=True)

  record = not args.no_record
  if record:
    # Persist key experiment settings so eval/play can reconstruct the correct env/action mode.
    step_dt_s = float(getattr(env_cfg.sim.mujoco, "timestep", 0.001)) * float(getattr(env_cfg, "decimation", 1))
    write_run_metadata(
      log_dir,
      argv=sys.argv,
      extra={
        "args": vars(args),
        "device": device,
        "action_mode": str(args.action_mode),
        "env_params": asdict(params),
        "env": {
          "sim_dt_s": float(getattr(env_cfg.sim.mujoco, "timestep", 0.001)),
          "decimation": int(getattr(env_cfg, "decimation", 1)),
          "step_dt_s": step_dt_s,
        },
        "command_override": {
          "trajectory_type": getattr(cmd_cfg, "trajectory_type", None) if cmd_cfg is not None else None,
          "sine_freq_hz_range": getattr(cmd_cfg, "sine_freq_hz_range", None) if cmd_cfg is not None else None,
          "sine_cycles_range": getattr(cmd_cfg, "sine_cycles_range", None) if cmd_cfg is not None else None,
          "joint_delta_scale": getattr(cmd_cfg, "joint_delta_scale", None) if cmd_cfg is not None else None,
          "joint_delta_scale_by_joint": getattr(cmd_cfg, "joint_delta_scale_by_joint", None) if cmd_cfg is not None else None,
        },
      },
    )
    update_latest_pointer(log_root, log_dir)
    train_stdout_log = log_dir / "train_stdout.log"
    train_jsonl = log_dir / "train_metrics.jsonl"
    parser_rec = RslRlStdoutParser(train_jsonl)
  else:
    train_stdout_log = None
    parser_rec = None

  print(f"[INFO] device={device} num_envs={args.num_envs} log_dir={log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

  runner = OnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device=device)

  resume_ckpt = (args.resume or "").strip()
  if args.resume_latest:
    latest_ptr = log_root / "_latest.txt"
    if latest_ptr.exists():
      latest_run = Path(latest_ptr.read_text(encoding="utf-8").strip())
      if latest_run.exists():
        best_it = -1
        best_path: Path | None = None
        for p in latest_run.glob("model_*.pt"):
          try:
            it = int(p.stem.split("_", 1)[1])
          except Exception:
            continue
          if it > best_it:
            best_it = it
            best_path = p
        if best_path is not None:
          resume_ckpt = str(best_path)
  if resume_ckpt:
    map_location = "cpu" if str(device).startswith("cpu") else None
    runner.load(
      resume_ckpt,
      load_optimizer=(not args.resume_no_optimizer),
      map_location=map_location,
    )
    print(f"[INFO] resumed from checkpoint: {resume_ckpt}")

  if record:
    with tee_stdout_to_file(train_stdout_log):
      class _LineTap:
        def __init__(self, wrapped_stream, parser):
          self._wrapped = wrapped_stream
          self._parser = parser
          self._buf = ""
        def write(self, s: str) -> int:
          self._buf += s
          while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._parser.process_line(line + "\n")
          return self._wrapped.write(s)
        def flush(self) -> None:
          return self._wrapped.flush()
      old_out = sys.stdout
      sys.stdout = _LineTap(sys.stdout, parser_rec)
      try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
      finally:
        sys.stdout = old_out
        parser_rec.close()
  else:
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
