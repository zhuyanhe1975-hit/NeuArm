from __future__ import annotations

import argparse
import os
import re
import sys
import time
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


def _compute_env_origins_grid(num_envs: int, env_spacing: float) -> "list[list[float]]":
  # Match mjlab TerrainImporter._compute_env_origins_grid.
  import math

  if num_envs <= 0:
    return []
  num_rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
  num_cols = math.ceil(num_envs / num_rows)
  origins: list[list[float]] = []
  for i in range(num_envs):
    ii = i // num_cols
    jj = i % num_cols
    x = -((ii) - (num_rows - 1) / 2.0) * float(env_spacing)
    y = ((jj) - (num_cols - 1) / 2.0) * float(env_spacing)
    origins.append([x, y, 0.0])
  return origins


class _RealTimeStepMixin:
  """Drive simulation steps by wall-clock time instead of 'one step per frame'.

  This keeps the animation real-time even when the native viewer is capped by
  monitor refresh (VSync). Rendering may be 60Hz, but simulation can still run
  at e.g. 200Hz by taking multiple env steps per rendered frame.
  """

  _rt_step_dt: float
  _rt_accum_s: float

  def _rt_init(self, step_dt: float) -> None:
    self._rt_step_dt = float(step_dt)
    self._rt_accum_s = 0.0

  def tick(self) -> bool:  # type: ignore[override]
    # Based on BaseViewer.tick, but step multiple times per frame as needed.
    self._process_actions()

    # Render-side sync (perturbations).
    with self._render_timer.measure_time():
      self.sync_viewer_to_env()

      if not self._is_paused:
        dt_wall = self._timer.tick()
        self._rt_accum_s += float(dt_wall) * float(self._time_multiplier)

        # Avoid spiraling after stalls (e.g., window dragged).
        max_catchup_steps = 200
        steps = min(int(self._rt_accum_s / max(self._rt_step_dt, 1e-9)), max_catchup_steps)
        if steps > 0:
          self._rt_accum_s -= steps * self._rt_step_dt
          for _ in range(steps):
            self.step_simulation()
        else:
          # Yield a bit if we are ahead of schedule.
          remain = max(self._rt_step_dt - self._rt_accum_s, 0.0)
          time.sleep(min(0.001, remain))
      else:
        # Keep updating the displayed pose while paused.
        self._timer.tick()

      self.sync_env_to_viewer()

    self._accumulated_render_time += self._render_timer.measured_time
    self._frame_count += 1
    self._update_fps()
    return True


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to rsl_rl model_*.pt (defaults to latest)")
  parser.add_argument("--device", type=str, default="auto", help="cpu | cuda:0 | auto")
  parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (use ,/. to switch when >1)")
  parser.add_argument("--env-idx", type=int, default=0, help="Initial env index to view")
  parser.add_argument("--show-other-envs", action="store_true", help="Render all envs at once (otherwise only the selected env is shown)")
  parser.add_argument(
    "--camera",
    type=str,
    default="root",
    help="Camera reference: world | root | link6 (default: root). link6 tracks the EE so the base may appear to move.",
  )
  parser.add_argument("--steps", type=int, default=0, help="Stop after N env steps (0 = run until window closes)")
  parser.add_argument("--viewer", type=str, default="auto", help="native | viser | auto")
  parser.add_argument("--fps", type=float, default=60.0, help="Viewer frame rate (ignored when --realtime is set)")
  parser.add_argument(
    "--realtime",
    action="store_true",
    help="Run at real-time (wall clock). This stays real-time even if native viewer is VSync-capped by stepping multiple env steps per rendered frame.",
  )
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
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer, ViewerConfig

  from rsl_rl.runners import OnPolicyRunner

  from irb2400_rl.task.env_cfg import Irb2400TrackingTaskParams, make_irb2400_tracking_env_cfg

  os.environ.setdefault("MUJOCO_GL", "glfw")

  env_cfg = make_irb2400_tracking_env_cfg(
    params=Irb2400TrackingTaskParams(num_envs=int(args.num_envs)),
    play=True,
  )
  env_cfg.viewer.env_idx = int(args.env_idx)
  cam = (args.camera or "").strip().lower()
  if cam == "world":
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    env_cfg.viewer.entity_name = None
    env_cfg.viewer.body_name = None
  elif cam in ("root", "base"):
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.body_name = None
  elif cam in ("link6", "ee", "tcp"):
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.body_name = "link_6"
  else:
    raise ValueError(f"Unsupported --camera '{args.camera}', expected world|root|link6")

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

  fps = float(args.fps)
  sim_hz = 1.0 / max(float(env.unwrapped.step_dt), 1e-9)
  if args.realtime:
    print(f"[INFO] realtime=on render_fps={fps:.1f}Hz sim_hz={sim_hz:.1f}Hz (step_dt={env.unwrapped.step_dt:.4f}s)")
  else:
    print(f"[INFO] realtime=off fps={fps:.1f}Hz (step_dt={env.unwrapped.step_dt:.4f}s)")

  num_steps = None if int(args.steps) <= 0 else int(args.steps)
  try:
    if viewer_backend == "native":
      print("[INFO] native viewer keys: ENTER=reset SPACE=pause/resume -=slower +=faster ,/.=prev/next env (when num_envs>1)")
      if args.show_other_envs:
        import mujoco

        class _GridEnvNativeViewer(NativeMujocoViewer):
          def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            spacing = float(getattr(self.env.unwrapped.scene, "env_spacing", 2.0))
            self._env_origins = _compute_env_origins_grid(self.env.unwrapped.num_envs, spacing)

          def sync_env_to_viewer(self) -> None:
            v = self.viewer
            assert v is not None
            assert self.mjm is not None and self.mjd is not None and self.vopt is not None

            with self._mj_lock:
              sim_data = self.env.unwrapped.sim.data

              # Primary env (selected) into mjd.
              self.mjd.qpos[:] = sim_data.qpos[self.env_idx].cpu().numpy()
              self.mjd.qvel[:] = sim_data.qvel[self.env_idx].cpu().numpy()
              if self.mjm.nmocap > 0:
                self.mjd.mocap_pos[:] = sim_data.mocap_pos[self.env_idx].cpu().numpy()
                self.mjd.mocap_quat[:] = sim_data.mocap_quat[self.env_idx].cpu().numpy()
              mujoco.mj_forward(self.mjm, self.mjd)

              v.user_scn.ngeom = 0
              if self._show_debug_vis and hasattr(self.env.unwrapped, "update_visualizers"):
                from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

                visualizer = MujocoNativeDebugVisualizer(v.user_scn, self.mjm, self.env_idx)
                self.env.unwrapped.update_visualizers(visualizer)

              # Render other envs (offset into a grid so they don't overlap).
              if self.vd is None:
                self.vd = mujoco.MjData(self.mjm)
              assert self.pert is not None

              for i in range(self.env.unwrapped.num_envs):
                if i == self.env_idx:
                  continue
                self.vd.qpos[:] = sim_data.qpos[i].cpu().numpy()
                self.vd.qvel[:] = sim_data.qvel[i].cpu().numpy()
                if self.mjm.nmocap > 0:
                  self.vd.mocap_pos[:] = sim_data.mocap_pos[i].cpu().numpy()
                  self.vd.mocap_quat[:] = sim_data.mocap_quat[i].cpu().numpy()
                mujoco.mj_forward(self.mjm, self.vd)

                start = int(v.user_scn.ngeom)
                mujoco.mjv_addGeoms(self.mjm, self.vd, self.vopt, self.pert, self.catmask, v.user_scn)
                origin = self._env_origins[i]
                for gi in range(start, int(v.user_scn.ngeom)):
                  v.user_scn.geoms[gi].pos[0] += float(origin[0])
                  v.user_scn.geoms[gi].pos[1] += float(origin[1])
                  v.user_scn.geoms[gi].pos[2] += float(origin[2])

              v.sync(state_only=True)

        if args.realtime:
          class _GridEnvNativeViewerRT(_RealTimeStepMixin, _GridEnvNativeViewer):
            pass

          viewer = _GridEnvNativeViewerRT(env, policy, frame_rate=fps)
          viewer._rt_init(env.unwrapped.step_dt)
        else:
          viewer = _GridEnvNativeViewer(env, policy, frame_rate=fps)
        viewer.run(num_steps=num_steps)
      else:
        # By default, render only the selected env. This avoids overlapping robots
        # when the scene spacing is not reflected in the native viewer.
        class _SingleEnvNativeViewer(NativeMujocoViewer):
          def setup(self) -> None:
            super().setup()
            self.vd = None

        if args.realtime:
          class _SingleEnvNativeViewerRT(_RealTimeStepMixin, _SingleEnvNativeViewer):
            pass

          viewer = _SingleEnvNativeViewerRT(env, policy, frame_rate=fps)
          viewer._rt_init(env.unwrapped.step_dt)
        else:
          viewer = _SingleEnvNativeViewer(env, policy, frame_rate=fps)
        viewer.run(num_steps=num_steps)
    elif viewer_backend == "viser":
      print("[INFO] viser: open the printed URL in your browser to view the scene; Ctrl-C to stop.")
      ViserPlayViewer(env, policy, frame_rate=fps).run(num_steps=num_steps)
    else:
      raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")
  except KeyboardInterrupt:
    print("\n[INFO] interrupted by user, closing viewer...")

  env.close()


if __name__ == "__main__":
  main()
