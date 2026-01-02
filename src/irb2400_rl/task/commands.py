from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class JointTrajectoryCommand(CommandTerm):
  cfg: JointTrajectoryCommandCfg

  def __init__(self, cfg: JointTrajectoryCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: "Entity" = env.scene[cfg.entity_name]
    joint_ids, _ = self.robot.find_joints(cfg.joint_names_expr, preserve_order=True)
    self.joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    self._segment_duration = torch.zeros(self.num_envs, device=self.device)
    self._q0 = torch.zeros(self.num_envs, len(joint_ids), device=self.device)
    self._q1 = torch.zeros_like(self._q0)

    self._q_des = torch.zeros_like(self._q0)
    self._qd_des = torch.zeros_like(self._q0)
    self._qdd_des = torch.zeros_like(self._q0)

    # Sine trajectory parameters (per env)
    self._sine_amp = torch.zeros_like(self._q0)
    self._sine_cycles = torch.ones(self.num_envs, 1, device=self.device)
    self._sine_freq_hz = torch.ones(self.num_envs, 1, device=self.device)

    # TCP desired trajectory (for reward shaping only).
    self._has_tcp = cfg.tcp_site_name is not None
    if self._has_tcp:
      site_pat = rf".*{cfg.tcp_site_name}$"
      site_ids, site_names = self.robot.find_sites((site_pat,), preserve_order=True)
      if len(site_ids) != 1:
        raise ValueError(
          f"Expected exactly 1 tcp site matching '{site_pat}', got: {site_names}"
        )
      self._tcp_site_local = int(site_ids[0])
      self._tcp_site_global = int(
        self.robot.data.indexing.site_ids[self._tcp_site_local].item()
      )

      self._tcp0 = torch.zeros(self.num_envs, 3, device=self.device)
      self._tcp_des = torch.zeros_like(self._tcp0)
      self._tcpd_des = torch.zeros_like(self._tcp0)
      self._tcpa_des = torch.zeros_like(self._tcp0)

      # Precompute a coarse FK table for the commanded joint trajectory:
      #   tcp_des(t) ~= FK(q_des(t)).
      # This keeps the TCP reward consistent with the joint-space command
      # (i.e., perfect joint tracking => ~0 TCP error), without doing per-step FK.
      if len(cfg.tcp_fk_knots_t) < 2:
        raise ValueError("tcp_fk_knots_t must have >= 2 knots")
      if cfg.tcp_fk_knots_t[0] != 0.0 or cfg.tcp_fk_knots_t[-1] != 1.0:
        raise ValueError("tcp_fk_knots_t must start at 0.0 and end at 1.0")
      if any(b <= a for a, b in zip(cfg.tcp_fk_knots_t, cfg.tcp_fk_knots_t[1:])):
        raise ValueError("tcp_fk_knots_t must be strictly increasing")

      self._tcp_fk_knots_t = torch.tensor(
        cfg.tcp_fk_knots_t, device=self.device, dtype=torch.float32
      )
      self._tcp_fk_knots_t_np = np.asarray(cfg.tcp_fk_knots_t, dtype=np.float64)
      self._tcp_fk_knots_pos = torch.zeros(
        self.num_envs, len(cfg.tcp_fk_knots_t), 3, device=self.device
      )

      # Reuse a MuJoCo CPU forward pass to compute tcp FK at knot configurations.
      self._mj_model = env.sim.mj_model
      self._mj_data_fk = mujoco.MjData(self._mj_model)
      joint_q_adr = self.robot.data.indexing.joint_q_adr[self.joint_ids].to(
        dtype=torch.long
      )
      self._qpos_adr_np = joint_q_adr.cpu().numpy()

    self.metrics["joint_pos_rmse"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self._q_des, self._qd_des, self._qdd_des], dim=-1)

  @property
  def phase(self) -> torch.Tensor:
    denom = torch.clamp(self._segment_duration, min=1e-6)
    return 1.0 - torch.clamp(self.time_left / denom, 0.0, 1.0)

  @property
  def has_tcp(self) -> bool:
    return self._has_tcp

  @property
  def tcp_site_local(self) -> int:
    if not self._has_tcp:
      raise AttributeError("TCP is disabled (tcp_site_name=None)")
    return self._tcp_site_local

  @property
  def tcp_pos_des(self) -> torch.Tensor:
    if not self._has_tcp:
      raise AttributeError("TCP is disabled (tcp_site_name=None)")
    return self._tcp_des

  @property
  def tcp_vel_des(self) -> torch.Tensor:
    if not self._has_tcp:
      raise AttributeError("TCP is disabled (tcp_site_name=None)")
    return self._tcpd_des

  @property
  def tcp_acc_des(self) -> torch.Tensor:
    if not self._has_tcp:
      raise AttributeError("TCP is disabled (tcp_site_name=None)")
    return self._tcpa_des

  def _update_metrics(self) -> None:
    q = self.robot.data.joint_pos[:, self.joint_ids]
    err = self._q_des - q
    self.metrics["joint_pos_rmse"] = torch.sqrt(torch.mean(err * err, dim=-1) + 1e-8)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self._segment_duration[env_ids] = self.time_left[env_ids].clone()

    q = self.robot.data.joint_pos[env_ids][:, self.joint_ids]
    lim = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]

    self._q0[env_ids] = q

    if self.cfg.joint_delta_scale_by_joint is None:
      span = (lim[..., 1] - lim[..., 0]) * self.cfg.joint_delta_scale
    else:
      if len(self.cfg.joint_delta_scale_by_joint) != self._q0.shape[1]:
        raise ValueError(
          "joint_delta_scale_by_joint must match num_joints: "
          f"{len(self.cfg.joint_delta_scale_by_joint)} != {self._q0.shape[1]}"
        )
      scale = torch.tensor(
        self.cfg.joint_delta_scale_by_joint, device=self.device, dtype=q.dtype
      ).view(1, -1)
      span = (lim[..., 1] - lim[..., 0]) * scale
    delta = (2.0 * torch.rand_like(q) - 1.0) * span
    q1 = torch.clamp(q + delta, lim[..., 0], lim[..., 1])
    self._q1[env_ids] = q1

    if self.cfg.trajectory_type.lower() == "sine":
      # Periodic sine around current q0 (phase=0).
      # Use symmetric amplitude within joint limits.
      q0 = q
      # available symmetric margin to joint limits
      margin = torch.minimum(q0 - lim[..., 0], lim[..., 1] - q0)
      margin = torch.clamp(margin * 0.95, min=0.0)
      # scale from joint range (and optional per-joint scale)
      if self.cfg.joint_delta_scale_by_joint is None:
        scale = float(self.cfg.joint_delta_scale)
        amp = (lim[..., 1] - lim[..., 0]) * (scale * 0.5)
      else:
        scale = torch.tensor(
          self.cfg.joint_delta_scale_by_joint, device=self.device, dtype=q0.dtype
        ).view(1, -1)
        amp = (lim[..., 1] - lim[..., 0]) * (scale * 0.5)
      amp = torch.minimum(amp, margin)
      self._sine_amp[env_ids] = amp
      # sample a single base frequency per env to keep a common segment duration
      f_lo, f_hi = self.cfg.sine_freq_hz_range
      f = torch.empty(env_ids.shape[0], 1, device=self.device).uniform_(float(f_lo), float(f_hi))
      c_lo, c_hi = self.cfg.sine_cycles_range
      cycles = torch.randint(int(c_lo), int(c_hi) + 1, (env_ids.shape[0], 1), device=self.device)
      self._sine_freq_hz[env_ids] = f
      self._sine_cycles[env_ids] = cycles.to(dtype=f.dtype)
      T = (cycles.to(dtype=f.dtype) / torch.clamp(f, min=1e-3)).squeeze(-1)
      # Override the manager timer so resampling happens exactly at segment end.
      self.time_left[env_ids] = T
      self._segment_duration[env_ids] = T

    if self._has_tcp:
      # tcp0 from current sim state (GPU). tcp FK knots from MuJoCo CPU FK.
      self._tcp0[env_ids] = self.robot.data.site_pos_w[env_ids, self._tcp_site_local]

      q0_cpu = q.detach().cpu().numpy()
      dq_cpu = (q1 - q).detach().cpu().numpy()
      env_ids_cpu = env_ids.detach().cpu().numpy()
      K = len(self._tcp_fk_knots_t_np)
      tcp_knots_cpu = np.zeros((env_ids_cpu.shape[0], K, 3), dtype=np.float64)

      for i in range(env_ids_cpu.shape[0]):
        for k, tk in enumerate(self._tcp_fk_knots_t_np):
          sk = 10.0 * tk**3 - 15.0 * tk**4 + 6.0 * tk**5
          qk = q0_cpu[i] + sk * dq_cpu[i]
          self._mj_data_fk.qpos[:] = 0.0
          self._mj_data_fk.qvel[:] = 0.0
          self._mj_data_fk.qpos[self._qpos_adr_np] = qk
          mujoco.mj_forward(self._mj_model, self._mj_data_fk)
          tcp_knots_cpu[i, k] = self._mj_data_fk.site_xpos[self._tcp_site_global].copy()

      self._tcp_fk_knots_pos[env_ids] = torch.as_tensor(
        tcp_knots_cpu, device=self.device, dtype=self._tcp_fk_knots_pos.dtype
      )

  def _update_command(self) -> None:
    # Quintic time scaling: s(t) = 10 t^3 - 15 t^4 + 6 t^5, t in [0, 1]
    T = torch.clamp(self._segment_duration, min=1e-6).unsqueeze(-1)
    t = self.phase.unsqueeze(-1)  # normalized [0,1]

    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    ds_dt = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
    d2s_dt2 = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T * T)

    dq = self._q1 - self._q0
    q_des = self._q0 + s * dq
    qd_des = ds_dt * dq
    qdd_des = d2s_dt2 * dq

    qd_des = torch.clamp(qd_des, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
    qdd_des = torch.clamp(qdd_des, -self.cfg.max_joint_acc, self.cfg.max_joint_acc)

    self._q_des = q_des
    self._qd_des = qd_des
    self._qdd_des = qdd_des

    if self._has_tcp:
      # Piecewise linear interpolation of FK waypoints in normalized time t.
      tk = self._tcp_fk_knots_t  # (K,)
      t0 = torch.clamp(t.squeeze(-1), 0.0, 1.0)  # (B,)
      # Segment index i s.t. tk[i] <= t < tk[i+1]
      idx = torch.bucketize(t0, tk[1:], right=False)
      idx = torch.clamp(idx, 0, tk.numel() - 2)  # (B,)

      idx_g = idx.view(-1, 1, 1).expand(-1, 1, 3)
      idx_hi_g = (idx + 1).view(-1, 1, 1).expand(-1, 1, 3)
      p_lo = self._tcp_fk_knots_pos.gather(1, idx_g).squeeze(1)
      p_hi = self._tcp_fk_knots_pos.gather(1, idx_hi_g).squeeze(1)

      t_lo = tk[idx]
      t_hi = tk[idx + 1]
      denom = torch.clamp(t_hi - t_lo, min=1e-6)
      alpha = ((t0 - t_lo) / denom).unsqueeze(-1)

      self._tcp_des = p_lo + alpha * (p_hi - p_lo)
      # Approximate derivative under piecewise-linear interpolation.
      T_scalar = torch.clamp(self._segment_duration, min=1e-6).unsqueeze(-1)
      self._tcpd_des = (p_hi - p_lo) / (denom.unsqueeze(-1) * T_scalar)
      self._tcpa_des.zero_()


@dataclass(kw_only=True)
class JointTrajectoryCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  joint_names_expr: tuple[str, ...] = (r".*joint_[1-6]$",)
  joint_delta_scale: float = 0.25
  joint_delta_scale_by_joint: tuple[float, ...] | None = None
  # Trajectory generator type:
  # - "quintic": random point-to-point quintic in joint space
  # - "sine": periodic sine around current q (phase=0)
  trajectory_type: str = "quintic"
  # Sine settings (used when trajectory_type=="sine").
  sine_freq_hz_range: tuple[float, float] = (0.2, 1.0)
  sine_cycles_range: tuple[int, int] = (1, 3)
  max_joint_vel: float = 3.0
  max_joint_acc: float = 20.0
  # If set, compute a TCP desired trajectory (position-only) for reward shaping.
  tcp_site_name: str | None = "tcp"
  # Normalized time knots (t in [0,1]) used to precompute FK(tcp) along the
  # commanded joint trajectory; reduces TCP reward saturation at mm scales.
  tcp_fk_knots_t: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
  debug_vis: bool = False

  def build(self, env: "ManagerBasedRlEnv") -> JointTrajectoryCommand:
    return JointTrajectoryCommand(self, env)
