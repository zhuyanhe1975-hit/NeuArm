from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_scene_to_default
from mjlab.envs.mdp.terminations import nan_detection, time_out
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from irb2400_rl.robot.irb2400_cfg import Irb2400ActuatorLimits, get_irb2400_robot_cfg
from irb2400_rl.task.actions import ResidualComputedTorqueActionCfg
from irb2400_rl.task.commands import JointTrajectoryCommandCfg
from irb2400_rl.task import mdp


@dataclass(frozen=True)
class Irb2400TrackingTaskParams:
  num_envs: int = 1024
  # Control / policy update period = sim_dt * decimation = 0.001 * 5 = 0.005s (5ms).
  decimation: int = 5
  episode_length_s: float = 6.0
  effort_limit: float = 1000.0
  traj_segment_time_range: tuple[float, float] = (0.6, 1.8)
  traj_joint_delta_scale: float = 0.05


def make_irb2400_tracking_env_cfg(
  *,
  params: Irb2400TrackingTaskParams = Irb2400TrackingTaskParams(),
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  scene = SceneCfg(
    num_envs=params.num_envs,
    env_spacing=2.5,
    entities={
      "robot": get_irb2400_robot_cfg(
        actuator_limits=Irb2400ActuatorLimits(effort_limit=params.effort_limit)
      )
    },
  )

  commands: dict[str, CommandTermCfg] = {
    "traj": JointTrajectoryCommandCfg(
      resampling_time_range=params.traj_segment_time_range,
      entity_name="robot",
      joint_names_expr=(r".*joint_[1-6]$",),
      joint_delta_scale=params.traj_joint_delta_scale,
      # Keep wrist-roll (j6) motion smaller to avoid large joint errors while
      # still tracking wrist-center position.
      joint_delta_scale_by_joint=(0.05, 0.05, 0.05, 0.05, 0.05, 0.005),
      trajectory_type="sine",
      sine_freq_hz_range=(0.2, 0.8),
      sine_cycles_range=(2, 4),
      tcp_site_name=None,
      debug_vis=False,
    )
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms={
        "q": ObservationTermCfg(func=mdp.joint_pos),
        "qd": ObservationTermCfg(func=mdp.joint_vel),
        "q_err": ObservationTermCfg(
          func=mdp.joint_pos_error, params={"command_name": "traj"}
        ),
        "qd_err": ObservationTermCfg(
          func=mdp.joint_vel_error, params={"command_name": "traj"}
        ),
        "phase": ObservationTermCfg(
          func=mdp.traj_phase, params={"command_name": "traj"}
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "q": ObservationTermCfg(func=mdp.joint_pos),
        "qd": ObservationTermCfg(func=mdp.joint_vel),
        "cmd": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "traj"}),
        "phase": ObservationTermCfg(func=mdp.traj_phase, params={"command_name": "traj"}),
        "last_action": ObservationTermCfg(func=mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "residual_tau": ResidualComputedTorqueActionCfg(
      entity_name="robot",
      command_name="traj",
      actuator_names=(r".*joint_[1-6]$",),
      effort_limit=params.effort_limit,
      # Stage-0 stable controller: PD + gravity compensation, no residual.
      residual_scale=0.0,
      residual_clip=0.0,
      # Stronger PD for tight joint tracking while staying stable.
      kp_min=50.0,
      kp_max=350.0,
      kd_min=2.0,
      kd_max=35.0,
      ki=0.0,
      err_scale=0.15,
      # Make j6 gains ramp up sooner (it tends to drift since wrist-center FK
      # is insensitive to wrist-roll).
      err_scale_by_joint=(0.15, 0.15, 0.15, 0.15, 0.15, 0.05),
      integral_limit=0.2,
      # Keep arm joints tight and gently boost only j6 (wrist roll).
      # j6: keep stiffness modest but add damping to avoid oscillation.
      kp_joint_scale=(2.0, 2.0, 2.0, 0.7, 0.7, 4.0),
      kd_joint_scale=(2.0, 2.0, 2.0, 0.7, 0.7, 12.0),
      ff_mode="ctff",
      ctff_joint_mask=(True, True, True, True, True, False),
    ),
  }

  events = {
    "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05) if not play else (0.0, 0.0),
        "velocity_range": (-0.05, 0.05) if not play else (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*joint_[1-6]$",)),
      },
    ),
  }

  rewards = {
    "track_q": RewardTermCfg(
      func=mdp.track_joint_pos_exp,
      weight=2.0,
      # Too-small std makes rewards ~0 for modest errors => weak learning signal.
      params={"command_name": "traj", "std": 0.25},
    ),
    "track_qd": RewardTermCfg(
      func=mdp.track_joint_vel_exp,
      weight=0.5,
      params={"command_name": "traj", "std": 1.5},
    ),
    # Strongly prefer "do nothing" initially; PPO quickly discovers baseline tracking.
    "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-1e-2),
    # Also penalize action changes to avoid residual-induced jitter.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-5e-3),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "nan": TerminationTermCfg(func=nan_detection),
    # Be less trigger-happy; frequent resets kill learning.
    "joint_limits": TerminationTermCfg(
      func=mdp.joint_pos_outside_soft_limits, params={"margin": 0.15}
    ),
  }

  if play:
    episode_length_s = float(1e9)
  else:
    episode_length_s = params.episode_length_s

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    decimation=params.decimation,
    episode_length_s=episode_length_s,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.001, integrator="implicitfast", iterations=50),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="link_6",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
  )
