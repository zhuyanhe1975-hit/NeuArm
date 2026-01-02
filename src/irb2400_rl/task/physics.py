from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def set_joint_friction_and_damping(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  *,
  asset_cfg: SceneEntityCfg,
  dof_damping: tuple[float, ...] | None = None,
  dof_frictionloss: tuple[float, ...] | None = None,
) -> None:
  """Set per-DoF viscous damping and Coulomb/static frictionloss on the plant.

  Notes:
  - MuJoCo models "dry" joint friction via `dof_frictionloss` (N*m). This acts as
    Coulomb friction and can also create stick-slip (static friction) behavior.
  - Viscous friction is modeled via `dof_damping` (N*m*s/rad).
  - This writes to the MuJoCo/Warp model fields so it affects the dynamics
    (i.e., it's part of the object/plant model, not the controller).
  """
  asset = env.scene[asset_cfg.name]

  joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
  if len(joint_ids) == 0:
    raise ValueError(f"No joints matched for friction/damping update: {asset_cfg.joint_names}")
  joint_ids_t = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
  dof_ids = asset.data.indexing.joint_v_adr[joint_ids_t].to(dtype=torch.long)

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)

  if dof_damping is not None:
    if len(dof_damping) != len(joint_ids):
      raise ValueError(f"dof_damping length {len(dof_damping)} != num_joints {len(joint_ids)}")
    vals = torch.tensor(dof_damping, device=env.device, dtype=env.sim.model.dof_damping.dtype).view(1, -1)
    env.sim.model.dof_damping[env_ids.unsqueeze(-1), dof_ids.view(1, -1)] = vals

  if dof_frictionloss is not None:
    if len(dof_frictionloss) != len(joint_ids):
      raise ValueError(f"dof_frictionloss length {len(dof_frictionloss)} != num_joints {len(joint_ids)}")
    vals = torch.tensor(dof_frictionloss, device=env.device, dtype=env.sim.model.dof_frictionloss.dtype).view(1, -1)
    env.sim.model.dof_frictionloss[env_ids.unsqueeze(-1), dof_ids.view(1, -1)] = vals

