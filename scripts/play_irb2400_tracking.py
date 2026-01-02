from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path


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


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to rsl_rl model_*.pt (defaults to latest)")
  parser.add_argument("--device", type=str, default="auto", help="cpu | cuda:0 | auto")
  parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (use ,/. to switch when >1)")
  parser.add_argument("--env-idx", type=int, default=0, help="Initial env index to view")
  parser.add_argument("--steps", type=int, default=0, help="Stop after N env steps (0 = run until window closes)")
  parser.add_argument("--viewer", type=str, default="auto", help="native | viser | auto")
  parser.add_argument("--fps", type=float, default=60.0, help="Viewer frame rate")
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root / "src"))

  from irb2400_rl.mjlab_bootstrap import ensure_mjlab_on_path

  ensure_mjlab_on_path()

  import torch

  if args.device == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  # Prefer native GUI when a display is available.
  if args.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer_backend = "native" if has_display else "viser"
  else:
    viewer_backend = args.viewer

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
  )
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  from rsl_rl.runners import OnPolicyRunner

  from irb2400_rl.task.env_cfg import Irb2400TrackingTaskParams, make_irb2400_tracking_env_cfg

  os.environ.setdefault("MUJOCO_GL", "glfw")

  env_cfg = make_irb2400_tracking_env_cfg(
    params=Irb2400TrackingTaskParams(num_envs=int(args.num_envs)),
    play=True,
  )
  env_cfg.viewer.env_idx = int(args.env_idx)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=1.0)

  experiment_root = repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking"
  resolved_checkpoint = args.checkpoint or _find_latest_checkpoint(experiment_root)
  if resolved_checkpoint is None:
    print("[INFO] resolved checkpoint: <none> (running zero-action policy)")
  else:
    print(f"[INFO] resolved checkpoint: {resolved_checkpoint}")

  if resolved_checkpoint is None:
    action_shape = env.unwrapped.action_space.shape  # type: ignore[attr-defined]

    class _ZeroPolicy:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    policy = _ZeroPolicy()
  else:
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
    play_log_dir = repo_root / "logs" / "rsl_rl" / "neuarm_irb2400_tracking" / "_play"
    play_log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, asdict(agent_cfg), str(play_log_dir), device=device)
    runner.load(str(resolved_checkpoint), load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] loaded policy from checkpoint: {resolved_checkpoint}")

  print("[INFO] native viewer keys: ENTER=reset SPACE=pause/resume -=slower +=faster ,/.=prev/next env (when num_envs>1)")

  num_steps = None if int(args.steps) <= 0 else int(args.steps)
  if viewer_backend == "native":
    NativeMujocoViewer(env, policy, frame_rate=float(args.fps)).run(num_steps=num_steps)
  elif viewer_backend == "viser":
    ViserPlayViewer(env, policy, frame_rate=float(args.fps)).run(num_steps=num_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")

  env.close()


if __name__ == "__main__":
  main()

