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
  # Residual torque safety knobs (start small; ramp in slowly).
  parser.add_argument("--residual-scale", type=float, default=2.0)
  parser.add_argument("--residual-clip", type=float, default=5.0)
  parser.add_argument("--residual-ramp-steps", type=int, default=500000)
  parser.add_argument("--residual-filter-tau", type=float, default=0.03)
  parser.add_argument("--action-l2-weight", type=float, default=-1e-2)
  parser.add_argument("--action-rate-weight", type=float, default=-5e-3)
  parser.add_argument("--device", type=str, default=None, help="cpu | cuda:0 | auto")
  parser.add_argument("--max-iterations", type=int, default=1000)
  parser.add_argument("--num-steps-per-env", type=int, default=32)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--run-name", type=str, default="")
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
    Irb2400TrackingTaskParams,
    make_irb2400_tracking_env_cfg,
  )

  if args.device in (None, "auto"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  # Residual action is clipped to [-1, 1] before scaling inside the action term.
  clip_actions = 1.0

  env_cfg = make_irb2400_tracking_env_cfg(
    params=Irb2400TrackingTaskParams(
      num_envs=args.num_envs,
      decimation=args.decimation,
      episode_length_s=args.episode_length_s,
      effort_limit=args.effort_limit,
    )
  )

  # Configure residual torque action term safely (do not disturb baseline).
  act = env_cfg.actions["residual_tau"]
  act.residual_scale = float(args.residual_scale)
  act.residual_clip = float(args.residual_clip)
  act.residual_ramp_steps = int(args.residual_ramp_steps)
  act.residual_filter_tau = float(args.residual_filter_tau)

  # Configure reward weights (action regularization).
  env_cfg.rewards["action_l2"].weight = float(args.action_l2_weight)
  if "action_rate_l2" in env_cfg.rewards:
    env_cfg.rewards["action_rate_l2"].weight = float(args.action_rate_weight)

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
    write_run_metadata(log_dir, argv=sys.argv, extra={"args": vars(args), "device": device})
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
