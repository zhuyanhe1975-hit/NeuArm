from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import math

from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class ResidualComputedTorqueAction(ActionTerm):
  cfg: "ResidualComputedTorqueActionCfg"

  def __init__(self, cfg: "ResidualComputedTorqueActionCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    self.robot: "Entity" = self._entity

    # IMPORTANT: Keep joint ordering consistent with the command generator
    # (which uses find_joints(..., preserve_order=True)). Using
    # find_joints_by_actuator_names() can reorder matches and silently scramble
    # joint-to-command alignment, causing large tracking errors.
    joint_ids, _ = self.robot.find_joints(cfg.actuator_names, preserve_order=True)
    if len(joint_ids) == 0:
      raise ValueError(f"No actuated joints matched: {cfg.actuator_names}")

    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._action_dim = int(self._joint_ids.numel())

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._i_err = torch.zeros_like(self._raw_actions)
    self._tau_resid_filt = torch.zeros_like(self._raw_actions)
    self._tau_resid_applied = torch.zeros_like(self._raw_actions)
    self._tau_cmd = torch.zeros_like(self._raw_actions)

    if cfg.ctff_joint_mask is not None and len(cfg.ctff_joint_mask) != self._action_dim:
      raise ValueError(
        "ctff_joint_mask must match action_dim: "
        f"{len(cfg.ctff_joint_mask)} != {self._action_dim}"
      )

    if cfg.err_scale_by_joint is not None and len(cfg.err_scale_by_joint) != self._action_dim:
      raise ValueError(
        "err_scale_by_joint must match action_dim: "
        f"{len(cfg.err_scale_by_joint)} != {self._action_dim}"
      )

    # Global DoF addresses (into sim.qvel/qfrc/qM).
    joint_v_adr = self.robot.data.indexing.joint_v_adr
    self._dof_ids = joint_v_adr[self._joint_ids].to(dtype=torch.long)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def tau_resid_applied(self) -> torch.Tensor:
    """Residual torque actually applied (post scale/clip/filter), in N*m."""
    return self._tau_resid_applied

  @property
  def tau_cmd(self) -> torch.Tensor:
    """Total joint torque command applied (post clamp), in N*m."""
    return self._tau_cmd

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._i_err[env_ids] = 0.0
    self._tau_resid_filt[env_ids] = 0.0
    self._tau_resid_applied[env_ids] = 0.0
    self._tau_cmd[env_ids] = 0.0

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

  def apply_actions(self) -> None:
    dt = float(self._env.step_dt)

    cmd = self._env.command_manager.get_command(self.cfg.command_name)
    q_des = cmd[:, 0 : self._action_dim]
    qd_des = cmd[:, 6 : 6 + self._action_dim]
    qdd_des = cmd[:, 12 : 12 + self._action_dim]

    q = self.robot.data.joint_pos[:, self._joint_ids]
    qd = self.robot.data.joint_vel[:, self._joint_ids]

    err = q_des - q
    err_d = qd_des - qd

    if self.cfg.ki != 0.0:
      self._i_err.add_(err * dt)
      self._i_err.clamp_(-self.cfg.integral_limit, self.cfg.integral_limit)

    abs_err = torch.abs(err)
    if self.cfg.err_scale_by_joint is None:
      err_scale = float(self.cfg.err_scale)
      gain_alpha = torch.tanh(abs_err / max(err_scale, 1e-6))
    else:
      err_scale = torch.tensor(
        self.cfg.err_scale_by_joint, device=self.device, dtype=abs_err.dtype
      ).view(1, -1)
      gain_alpha = torch.tanh(abs_err / torch.clamp(err_scale, min=1e-6))

    kp = self.cfg.kp_min + (self.cfg.kp_max - self.cfg.kp_min) * gain_alpha
    kd = self.cfg.kd_min + (self.cfg.kd_max - self.cfg.kd_min) * gain_alpha

    if self.cfg.kp_joint_scale is not None:
      kp_scale = torch.tensor(
        self.cfg.kp_joint_scale, device=self.device, dtype=kp.dtype
      ).view(1, -1)
      kp = kp * kp_scale
    if self.cfg.kd_joint_scale is not None:
      kd_scale = torch.tensor(
        self.cfg.kd_joint_scale, device=self.device, dtype=kd.dtype
      ).view(1, -1)
      kd = kd * kd_scale

    tau_pid = kp * err + kd * err_d + self.cfg.ki * self._i_err

    tau_ff = 0.0
    ff_mode = self.cfg.ff_mode.lower()
    if ff_mode not in ("none", "gravcomp", "bias", "ctff"):
      raise ValueError(
        f"Unsupported ff_mode='{self.cfg.ff_mode}', expected none|gravcomp|bias|ctff"
      )

    if ff_mode != "none":
      nv = int(self._env.sim.mj_model.nv)
      dof = self._dof_ids

      if ff_mode == "gravcomp":
        # Gravity compensation only.
        qfrc = self._env.sim.data.qfrc_gravcomp[:, :nv].contiguous()
        tau_ff = qfrc.index_select(1, dof)
      else:
        # Bias compensation (gravity + coriolis + constraint terms).
        qfrc_bias = self._env.sim.data.qfrc_bias[:, :nv].contiguous()
        tau_ff = qfrc_bias.index_select(1, dof)

        if ff_mode == "ctff":
          # Add inertia term M(q) * qdd_des (computed-torque feedforward).
          qM = self._env.sim.data.qM[:, :nv, :nv].contiguous()
          qM_sub = qM.index_select(1, dof).index_select(2, dof).contiguous()
          qdd = qdd_des.contiguous()
          tau_mass = torch.sum(qM_sub * qdd.unsqueeze(1), dim=-1)
          if self.cfg.ctff_joint_mask is not None:
            mask = torch.tensor(
              self.cfg.ctff_joint_mask, device=self.device, dtype=tau_mass.dtype
            ).view(1, -1)
            tau_mass = tau_mass * mask
          tau_ff = tau_ff + tau_mass

    residual_scale = float(self.cfg.residual_scale)
    if self.cfg.residual_ramp_steps and self.cfg.residual_ramp_steps > 0:
      ramp = min(1.0, float(self._env.common_step_counter) / float(self.cfg.residual_ramp_steps))
      residual_scale = residual_scale * ramp

    tau_resid = residual_scale * self._raw_actions
    if self.cfg.residual_clip is not None:
      tau_resid = torch.clamp(tau_resid, -self.cfg.residual_clip, self.cfg.residual_clip)

    if self.cfg.residual_filter_tau and self.cfg.residual_filter_tau > 0.0:
      # First-order low-pass: y <- a*y + (1-a)*x, a = exp(-dt/tau)
      a = math.exp(-dt / float(self.cfg.residual_filter_tau))
      self._tau_resid_filt.mul_(a).add_(tau_resid, alpha=(1.0 - a))
      tau_resid = self._tau_resid_filt

    self._tau_resid_applied[:] = tau_resid

    tau = tau_ff + tau_pid + tau_resid
    tau = torch.clamp(tau, -self.cfg.effort_limit, self.cfg.effort_limit)

    self._tau_cmd[:] = tau

    self.robot.set_joint_effort_target(tau, joint_ids=self._joint_ids)

@dataclass(kw_only=True)
class ResidualComputedTorqueActionCfg(ActionTermCfg):
  entity_name: str = "robot"
  clip: dict[str, tuple] | None = None
  command_name: str = "traj"
  actuator_names: tuple[str, ...] = (r".*joint_[1-6]$",)
  effort_limit: float = 300.0

  # Residual (NN) scaling.
  residual_scale: float = 30.0
  residual_clip: float | None = 60.0
  # Safety: slowly ramp in residual authority over global env steps
  # (control steps, i.e. decimated steps). 0 disables ramping.
  residual_ramp_steps: int = 0
  # Safety: low-pass filter residual torque (seconds). 0 disables filtering.
  residual_filter_tau: float = 0.0

  # Variable-gain PID (joint-space).
  kp_min: float = 50.0
  kp_max: float = 300.0
  kd_min: float = 2.0
  kd_max: float = 30.0
  ki: float = 0.0
  err_scale: float = 0.15  # rad, controls gain ramp-up
  err_scale_by_joint: tuple[float, ...] | None = None  # rad, optional per-joint
  integral_limit: float = 0.2  # rad*s (per joint)
  # Optional per-joint gain scaling (length = action_dim).
  kp_joint_scale: tuple[float, ...] | None = None
  kd_joint_scale: tuple[float, ...] | None = None

  # Feedforward mode:
  # - "none": no feedforward
  # - "gravcomp": gravity compensation only (qfrc_gravcomp)
  # - "bias": bias compensation only (qfrc_bias)
  # - "ctff": computed-torque feedforward (qfrc_bias + M(q) * qdd_des)
  #
  # Optional: apply the inertia term only on a subset of joints by setting
  # ctff_joint_mask (length = action_dim). Joints with mask=False will
  # receive only the bias term (plus PD/residual).
  ff_mode: str = "gravcomp"
  ctff_joint_mask: tuple[bool, ...] | None = None

  def build(self, env: "ManagerBasedRlEnv") -> ResidualComputedTorqueAction:
    return ResidualComputedTorqueAction(self, env)
