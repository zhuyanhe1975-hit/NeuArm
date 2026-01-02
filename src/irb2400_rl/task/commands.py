from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    self.metrics["joint_pos_rmse"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self._q_des, self._qd_des, self._qdd_des], dim=-1)

  @property
  def phase(self) -> torch.Tensor:
    denom = torch.clamp(self._segment_duration, min=1e-6)
    return 1.0 - torch.clamp(self.time_left / denom, 0.0, 1.0)

  def _update_metrics(self) -> None:
    q = self.robot.data.joint_pos[:, self.joint_ids]
    err = self._q_des - q
    self.metrics["joint_pos_rmse"] = torch.sqrt(
      torch.mean(err * err, dim=-1) + 1e-8
    )

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

@dataclass(kw_only=True)
class JointTrajectoryCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  joint_names_expr: tuple[str, ...] = (r".*joint_[1-6]$",)
  joint_delta_scale: float = 0.25
  joint_delta_scale_by_joint: tuple[float, ...] | None = None
  max_joint_vel: float = 3.0
  max_joint_acc: float = 20.0
  debug_vis: bool = False

  def build(self, env: "ManagerBasedRlEnv") -> JointTrajectoryCommand:
    return JointTrajectoryCommand(self, env)
