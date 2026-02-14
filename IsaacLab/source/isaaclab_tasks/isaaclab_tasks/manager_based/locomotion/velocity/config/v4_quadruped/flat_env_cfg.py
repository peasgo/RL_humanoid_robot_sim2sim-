# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# V4 四足机器人 flat env 配置
# 基于 unitree_rl_lab Go2 配置改写，适配 V4 坐标系
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
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG


# ============================================================
# V4 坐标系适配函数
# V4 机器人 body frame: X=左右, Y=上下(重力方向), Z=前后
# Isaac Sim world frame: X=前后, Y=左右, Z=上下
# rot=(0.7071, 0.7071, 0, 0) 即绕X轴旋转90度
# body_x -> world_x, body_y -> world_z, body_z -> world_-y
# ============================================================

def v4_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b
    # remap: forward=body_z, lateral=body_x, vertical=body_y
    return torch.stack([vel[:, 2], vel[:, 0], vel[:, 1]], dim=-1)


def v4_base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b
    return torch.stack([ang[:, 0], ang[:, 2], ang[:, 1]], dim=-1)


def v4_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    return torch.stack([grav[:, 2], grav[:, 0], grav[:, 1]], dim=-1)


# ============================================================
# 速度跟踪奖励（V4坐标系适配）
# ============================================================

def v4_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel = asset.data.root_lin_vel_b
    # V4: forward=body_z, lateral=body_x
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
    # V4: yaw rate = body_y angular velocity
    actual_yaw_rate = ang[:, 1]
    error = torch.square(cmd[:, 2] - actual_yaw_rate)
    return torch.exp(-error / std**2)


# ============================================================
# V4 惩罚函数
# ============================================================

def v4_lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚垂直方向速度（V4: body Y轴）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def v4_ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚非yaw方向角速度（V4: body X和Z轴）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b
    return torch.square(ang[:, 0]) + torch.square(ang[:, 2])


def v4_flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚非水平姿态（V4: gravity应在body Y轴）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    return torch.square(grav[:, 0]) + torch.square(grav[:, 2])


def v4_lateral_vel_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚侧向速度（V4: body X轴），防止走歪"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


# ============================================================
# 步态奖励（来自宇树 unitree_rl_lab）
# ============================================================

def v4_feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str = None,
) -> torch.Tensor:
    """对角步态奖励：约束脚的接触/摆动相位
    
    对于四足对角步态(trot):
    - 前右+后左 同相 (offset=0.0)
    - 前左+后右 同相 (offset=0.5)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def v4_feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """站立时四脚着地奖励"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


# ============================================================
# 其他辅助奖励
# ============================================================

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


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """宇树风格：站立时加大默认姿态惩罚"""
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


def v4_feet_slide(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚脚在地面滑动"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    vel_norm = torch.norm(body_vel, dim=-1)
    return torch.sum(contacts.float() * vel_norm, dim=-1)


def v4_air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚四脚空中时间方差，鼓励对称步态"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def v4_bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """V4: gravity应对齐body Y轴"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 1]).abs() > limit_angle


def reset_waist_joint_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    waist_joint_ids = asset.find_joints("Waist_2")[0]
    default_waist_pos = asset.data.default_joint_pos[env_ids][:, waist_joint_ids]
    asset.set_joint_position_target(default_waist_pos, joint_ids=waist_joint_ids, env_ids=env_ids)


# ============================================================
# 场景配置
# ============================================================

FEET_BODIES = ["R_ARM_feet", "L_arm_feet", "R_Feet", "L_Feet"]


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


# ============================================================
# 命令配置 - 只走直线，不转弯
# ============================================================

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
            lin_vel_x=(0.3, 0.8),     # 只前进
            lin_vel_y=(0.0, 0.0),     # 不侧移
            ang_vel_z=(0.0, 0.0),     # 不转弯
        ),
    )


# ============================================================
# 动作配置
# ============================================================

@configclass
class ActionsCfg:

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "RSDp", "RSDy", "RARMp",
            "LSDp", "LSDy", "LARMp",
            "RHIPp", "RHIPy", "RKNEEP",
            "LHIPp", "LHIPy", "LKNEEp",
        ],
        scale=0.25,
        use_default_offset=True,
    )


# ============================================================
# 观测配置
# ============================================================

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        base_ang_vel = ObsTerm(
            func=v4_base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
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
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================
# 事件配置（域随机化）
# ============================================================

@configclass
class EventCfg:

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
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


# ============================================================
# 奖励配置 - 基于宇树Go2，适配V4
# ============================================================

@configclass
class RewardsCfg:

    # -- 任务：速度跟踪（核心奖励）
    track_lin_vel_xy = RewTerm(
        func=v4_track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=v4_track_ang_vel_z_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- 基座惩罚
    lin_vel_z_l2 = RewTerm(func=v4_lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=v4_ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=v4_flat_orientation_l2, weight=-2.5)
    lateral_vel = RewTerm(func=v4_lateral_vel_penalty, weight=-1.5)

    # -- 关节惩罚
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=energy, weight=-2e-5)

    joint_pos = RewTerm(
        func=joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- 步态奖励（宇树核心：对角步态 trot）
    # 脚顺序: R_ARM_feet(前右), L_arm_feet(前左), R_Feet(后右), L_Feet(后左)
    # trot步态: 前右+后左同相(0.0), 前左+后右同相(0.5)
    feet_gait = RewTerm(
        func=v4_feet_gait,
        weight=1.0,
        params={
            "period": 0.5,
            "offset": [0.0, 0.5, 0.5, 0.0],  # FR, FL, RR, RL
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "threshold": 0.5,
            "command_name": "base_velocity",
        },
    )

    # -- 脚部奖励/惩罚
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=v4_air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES)},
    )
    feet_slide = RewTerm(
        func=v4_feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
        },
    )

    # -- 站立时四脚着地
    feet_contact_still = RewTerm(
        func=v4_feet_contact_without_cmd,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "command_name": "base_velocity",
        },
    )

    # -- 接触惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link|Waist.*"),
            "threshold": 1.0,
        },
    )

    # -- V4 特有：锁定腰部
    waist_lock = RewTerm(func=waist_deviation_penalty, weight=-5.0)


# ============================================================
# 终止条件
# ============================================================

@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )

    bad_orientation = DoneTerm(func=v4_bad_orientation, params={"limit_angle": 0.8})


# ============================================================
# Curriculum（暂不使用，保持简单）
# ============================================================

@configclass
class CurriculumCfg:
    pass


# ============================================================
# 主环境配置
# ============================================================

@configclass
class V4QuadrupedFlatEnvCfg(ManagerBasedRLEnvCfg):
    """V4四足直线行走环境配置"""

    scene: V4QuadrupedSceneCfg = V4QuadrupedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
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
