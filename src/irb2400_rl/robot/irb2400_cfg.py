from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets


REPO_ROOT = Path(__file__).resolve().parents[3]


IRB2400_MJCF: Path = REPO_ROOT / "assets" / "abb_irb2400" / "mjcf" / "irb2400_mjlab.xml"
assert IRB2400_MJCF.exists(), f"Missing MJCF: {IRB2400_MJCF}"

IRB2400_MESH_DIR: Path = REPO_ROOT / "assets" / "abb_irb2400" / "meshes" / "collision"
assert IRB2400_MESH_DIR.exists(), f"Missing meshes: {IRB2400_MESH_DIR}"


def get_irb2400_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(IRB2400_MJCF))
  assets: dict[str, bytes] = {}
  update_assets(assets, IRB2400_MESH_DIR, meshdir=spec.meshdir)
  spec.assets = assets
  return spec


@dataclass(frozen=True)
class Irb2400ActuatorLimits:
  effort_limit: float = 1000.0  # N*m (tune as needed)
  gear: float = 1.0


def get_irb2400_robot_cfg(
  *,
  actuator_limits: Irb2400ActuatorLimits = Irb2400ActuatorLimits(),
  soft_joint_pos_limit_factor: float = 0.98,
) -> EntityCfg:
  motor_cfg = BuiltinMotorActuatorCfg(
    target_names_expr=(r".*joint_[1-6]$",),
    effort_limit=actuator_limits.effort_limit,
    gear=actuator_limits.gear,
    armature=0.0,
    frictionloss=0.0,
  )

  articulation = EntityArticulationInfoCfg(
    actuators=(motor_cfg,),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
  )

  init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={r".*joint_[1-6]$": 0.0},
    joint_vel={r".*joint_[1-6]$": 0.0},
  )

  return EntityCfg(
    init_state=init_state,
    spec_fn=get_irb2400_spec,
    articulation=articulation,
  )
