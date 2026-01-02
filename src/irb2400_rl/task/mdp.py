from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def _robot(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> "Entity":
  return cast("Entity", env.scene[asset_cfg.name])


def joint_pos(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG) -> torch.Tensor:
  return _robot(env, asset_cfg).data.joint_pos


def joint_vel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG) -> torch.Tensor:
  return _robot(env, asset_cfg).data.joint_vel


def last_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return env.action_manager.action


def generated_commands(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  return env.command_manager.get_command(command_name)


def joint_pos_error(env: "ManagerBasedRlEnv", command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  q_des = cmd[:, 0:6]
  q = _robot(env, asset_cfg).data.joint_pos
  return q_des - q


def joint_vel_error(env: "ManagerBasedRlEnv", command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  qd_des = cmd[:, 6:12]
  qd = _robot(env, asset_cfg).data.joint_vel
  return qd_des - qd


def traj_phase(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  phase = getattr(term, "phase", None)
  if phase is None:
    raise AttributeError(f"Command term '{command_name}' does not expose .phase")
  return cast(torch.Tensor, phase).unsqueeze(-1)


def track_joint_pos_exp(env: "ManagerBasedRlEnv", command_name: str, std: float) -> torch.Tensor:
  err = joint_pos_error(env, command_name)
  mse = torch.mean(err * err, dim=-1)
  return torch.exp(-mse / (std * std))


def track_joint_vel_exp(env: "ManagerBasedRlEnv", command_name: str, std: float) -> torch.Tensor:
  err = joint_vel_error(env, command_name)
  mse = torch.mean(err * err, dim=-1)
  return torch.exp(-mse / (std * std))


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
  a = env.action_manager.action
  return torch.sum(a * a, dim=-1)


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
  a = env.action_manager.action
  prev = env.action_manager.prev_action
  da = a - prev
  return torch.sum(da * da, dim=-1)


def joint_pos_outside_soft_limits(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  margin: float = 0.01,
) -> torch.Tensor:
  robot = _robot(env, asset_cfg)
  q = robot.data.joint_pos
  lim = robot.data.soft_joint_pos_limits
  below = q < (lim[..., 0] - margin)
  above = q > (lim[..., 1] + margin)
  return torch.any(below | above, dim=-1)

