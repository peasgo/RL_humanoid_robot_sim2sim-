# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
import math
import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG



def v4_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b
    return torch.stack([vel[:, 2], vel[:, 0], vel[:, 1]], dim=-1)


def v4_base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b
    return torch.stack([ang[:, 0], ang[:, 2], ang[:, 1]], dim=-1)


def v4_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    return torch.stack([grav[:, 2], grav[:, 0], grav[:, 1]], dim=-1)


def v4_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel = asset.data.root_lin_vel_b
    actual_forward = vel[:, 2]
    actual_lateral = vel[:, 0]
    error = torch.square(cmd[:, 0] - actual_forward) + torch.square(cmd[:, 1] - actual_lateral)
    return torch.exp(-error / std**2)


def v4_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang = asset.data.root_ang_vel_b
    actual_yaw_rate = ang[:, 1]
    error = torch.square(cmd[:, 2] - actual_yaw_rate)
    return torch.exp(-error / std**2)


def v4_lin_vel_y_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def v4_ang_vel_xz_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b
    return torch.square(ang[:, 0]) + torch.square(ang[:, 2])


def v4_flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    return torch.square(grav[:, 0]) + torch.square(grav[:, 2])


def action_smoothness_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:

    act = env.action_manager.action
    prev = env.action_manager.prev_action
    return torch.sum(torch.abs(act - prev), dim=1)


def waist_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    waist_idx = asset.find_joints("Waist_2")[0]
    waist_pos = asset.data.joint_pos[:, waist_idx]
    waist_default = asset.data.default_joint_pos[:, waist_idx]
    deviation = torch.sum(torch.square(waist_pos - waist_default), dim=-1)
    return deviation


def quadruped_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.22,
    tolerance: float = 0.05,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    current_height = robot.data.root_pos_w[:, 2]
    error = torch.abs(current_height - target_height)
    outside = torch.clamp(error - tolerance, min=0.0)
    reward = torch.exp(-torch.square(outside) / 0.005)
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


def feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:

    contact_sensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_forces[:, 0, :], dim=-1)
    in_contact = (contact_force > threshold).float()
    return torch.mean(in_contact, dim=-1) if in_contact.dim() > 1 else in_contact


def reset_waist_joint_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    asset = env.scene[asset_cfg.name]
    waist_joint_ids = asset.find_joints("Waist_2")[0]
    default_waist_pos = asset.data.default_joint_pos[env_ids][:, waist_joint_ids]
    asset.set_joint_position_target(default_waist_pos, joint_ids=waist_joint_ids, env_ids=env_ids)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
  
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)

    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)



@configclass
class V4QuadrupedSceneCfg(InteractiveSceneCfg):


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = V4_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "RSDp", "RSDy", "RARMp", "RARMAP",
            "LSDp", "LSDy", "LARMp", "LARMAp",
            "RHIPp", "RHIPy", "RKNEEP", "RANKLEp",
            "LHIPp", "LHIPy", "LKNEEp", "LANKLEp",
        ],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:


    @configclass
    class PolicyCfg(ObsGroup):
       

        base_ang_vel = ObsTerm(
            func=v4_base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.2,
        )
        projected_gravity = ObsTerm(
            func=v4_projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2), 
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 3.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_waist_target = EventTerm(
        func=reset_waist_joint_target,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:

    track_lin_vel_xy_exp = RewTerm(
        func=v4_track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=v4_track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    lin_vel_z_l2 = RewTerm(func=v4_lin_vel_y_l2, weight=-2.0)

    ang_vel_xy_l2 = RewTerm(func=v4_ang_vel_xz_l2, weight=-0.05)

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    joint_pos = RewTerm(
        func=joint_position_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    energy = RewTerm(func=energy, weight=-2e-5)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*|.*ARMA.*"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link|Waist.*"),
            "threshold": 1.0,
        },
    )

    flat_orientation_l2 = RewTerm(func=v4_flat_orientation_l2, weight=-2.5)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    waist_lock = RewTerm(func=waist_deviation_penalty, weight=-5.0)



@configclass
class TerminationsCfg:
    """终止条件配置"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:

    pass

@configclass
class V4QuadrupedFlatEnvCfg(ManagerBasedRLEnvCfg):
    """V4四足行走平地环境配置"""

    scene: V4QuadrupedSceneCfg = V4QuadrupedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化"""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class V4QuadrupedFlatEnvCfg_PLAY(V4QuadrupedFlatEnvCfg):


    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
