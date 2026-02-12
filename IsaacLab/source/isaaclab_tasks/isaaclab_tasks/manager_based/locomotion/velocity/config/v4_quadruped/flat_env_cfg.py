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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 导入 locomotion velocity 的 mdp 模块（包含 feet_air_time 等函数）
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG



def v4_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V4适配：返回重映射后的线速度 [前进(+Z), 左右(X), 上下(Y)]"""
    asset: RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b
    # 前进=局部+Z(世界-Y), 左右=局部X(世界X), 上下=局部Y(世界+Z)
    return torch.stack([vel[:, 2], vel[:, 0], vel[:, 1]], dim=-1)


def v4_base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V4适配：返回重映射后的角速度 [roll(X), pitch(+Z), yaw(Y)]"""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b
    # roll=局部X, pitch=局部+Z(绕前进轴的俯仰), yaw=局部Y(绕上方轴的转弯)
    return torch.stack([ang[:, 0], ang[:, 2], ang[:, 1]], dim=-1)


def v4_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V4适配：返回重映射后的重力投影 [前进(+Z), 左右(X), 上下(Y)]"""
    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    # 前进=局部+Z, 左右=局部X, 上下=局部Y
    return torch.stack([grav[:, 2], grav[:, 0], grav[:, 1]], dim=-1)


def v4_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V4适配：跟踪线速度命令（前进+左右）。
    命令 command[:, 0] = 前进速度 → 对应 +root_lin_vel_b[:, 2]（局部+Z=世界-Y=前进）
    命令 command[:, 1] = 左右速度 → 对应 root_lin_vel_b[:, 0]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel = asset.data.root_lin_vel_b
    actual_forward = vel[:, 2]   # 局部+Z = 世界-Y = 前进方向
    actual_lateral = vel[:, 0]   # 局部X = 世界X = 左右方向
    error = torch.square(cmd[:, 0] - actual_forward) + torch.square(cmd[:, 1] - actual_lateral)
    return torch.exp(-error / std**2)


def v4_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V4适配：跟踪角速度命令（转弯=绕世界Z轴=绕局部Y轴）。
    命令 command[:, 2] = 转弯角速度 → 对应 root_ang_vel_b[:, 1]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang = asset.data.root_ang_vel_b
    # 转弯角速度 = 绕局部Y轴
    actual_yaw_rate = ang[:, 1]
    error = torch.square(cmd[:, 2] - actual_yaw_rate)
    return torch.exp(-error / std**2)


def v4_lin_vel_y_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V4适配：惩罚上下方向线速度（局部Y=世界Z，不希望上下跳动）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def v4_ang_vel_xz_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V4适配：惩罚非转弯方向的角速度（局部X和Z，不希望翻滚/俯仰）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b
    # 惩罚绕X轴（roll）和绕Z轴（pitch在V4中）的角速度
    return torch.square(ang[:, 0]) + torch.square(ang[:, 2])


def v4_flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V4适配：惩罚非水平姿态。
    正确四足姿态时 projected_gravity_b = (0, -1, 0)（重力沿局部-Y=世界-Z）
    惩罚 X 和 Z 分量（不是标准的 X 和 Y）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    # 惩罚 X 和 Z 分量（正确时应为0）
    return torch.square(grav[:, 0]) + torch.square(grav[:, 2])


def action_smoothness_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """惩罚动作的二阶差分（jerk），抑制高频振荡。
    jerk = action[t] - 2*action[t-1] + action[t-2]
    当连续三步动作一致时jerk=0，振荡时jerk很大。
    """
    act = env.action_manager.action
    prev = env.action_manager.prev_action
    # prev_prev 不直接可用，用 action_rate 的变化来近似
    # 简化版：惩罚 (action - prev_action) 的绝对值，L1更能抑制尖峰
    return torch.sum(torch.abs(act - prev), dim=1)


def waist_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚腰部关节（Waist_2）偏离初始位置（π rad = 180°）。
    腰部不受策略控制，但外力/碰撞可能导致偏移，此惩罚鼓励策略避免产生导致腰部转动的动作。
    """
    asset = env.scene[asset_cfg.name]
    # 获取 Waist_2 关节的索引
    waist_idx = asset.find_joints("Waist_2")[0]
    # 当前关节位置
    waist_pos = asset.data.joint_pos[:, waist_idx]
    # 默认位置（π rad）
    waist_default = asset.data.default_joint_pos[:, waist_idx]
    # 偏差的平方
    deviation = torch.sum(torch.square(waist_pos - waist_default), dim=-1)
    return deviation


def quadruped_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.22,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """四足高度保持奖励：鼓励机器人保持在目标高度附近。"""
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
    """脚掌接地奖励：鼓励四只脚掌保持与地面接触。
    检测末端效应器（手部和脚部）的接触力，有接触则给予奖励。
    """
    contact_sensor = env.scene[sensor_cfg.name]
    # 获取接触力的范数
    net_forces = contact_sensor.data.net_forces_w_history
    # 取最近一帧的接触力
    contact_force = torch.norm(net_forces[:, 0, :], dim=-1)
    # 有接触（力>阈值）的脚数量，归一化到 [0, 1]
    in_contact = (contact_force > threshold).float()
    # 返回接触脚的比例（4只脚全接地=1.0）
    return torch.mean(in_contact, dim=-1) if in_contact.dim() > 1 else in_contact


def reset_waist_joint_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset时设置Waist_2的PD目标为默认值(180°)，防止被拉回0°。"""
    asset = env.scene[asset_cfg.name]
    waist_joint_ids = asset.find_joints("Waist_2")[0]
    default_waist_pos = asset.data.default_joint_pos[env_ids][:, waist_joint_ids]
    asset.set_joint_position_target(default_waist_pos, joint_ids=waist_joint_ids, env_ids=env_ids)


@configclass
class V4QuadrupedSceneCfg(InteractiveSceneCfg):
    """V4四足行走场景配置"""

    # 地面
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

    # 机器人
    robot: ArticulationCfg = V4_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触力传感器 - 所有body
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 灯光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """速度命令配置"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,  # V4前进方向是局部+Z，不是局部X，heading_w会算错
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),   # 前进/后退（映射到局部+Z）
            lin_vel_y=(-0.5, 0.5),   # 左右（映射到局部X）
            ang_vel_z=(-1.0, 1.0),   # 转弯（映射到绕局部Y轴）
        ),
    )


@configclass
class ActionsCfg:
    """动作配置 - 关节位置控制
    Waist_2 被排除在策略控制之外，锁定在默认位置（180°）。
    腰部执行器的高刚度(400)会将其保持在初始状态。
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # 排除 Waist_2，只控制四肢的16个关节
        joint_names=[
            "RSDp", "RSDy", "RARMp", "RARMAP",   # 前右腿（右臂）
            "LSDp", "LSDy", "LARMp", "LARMAp",   # 前左腿（左臂）
            "RHIPp", "RHIPy", "RKNEEP", "RANKLEp",  # 后右腿
            "LHIPp", "LHIPy", "LKNEEp", "LANKLEp",  # 后左腿
        ],
        scale=0.25,  # [参考 Go2] 降低动作幅度 (原0.5)，使控制更精细
        use_default_offset=True,  # 使用默认关节位置作为偏移（四足姿态）
    )


@configclass
class ObservationsCfg:
    """观测配置 - 使用V4坐标系适配的观测函数"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组"""

        # V4适配：重映射后的线速度 [前进, 左右, 上下]
        base_lin_vel = ObsTerm(
            func=v4_base_lin_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 从±0.1降到±0.05，减少sim2real gap
        )
        # V4适配：重映射后的角速度 [roll, pitch, yaw]
        base_ang_vel = ObsTerm(
            func=v4_base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 从±0.2降到±0.1
        )
        # V4适配：重映射后的重力投影
        projected_gravity = ObsTerm(
            func=v4_projected_gravity,
            noise=Unoise(n_min=-0.025, n_max=0.025),  # 从±0.05降到±0.025
        )
        # 速度命令 [前进, 左右, 转弯]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # 关节位置（相对于默认位置）
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 关节速度
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),  # 从±1.5降到±0.5，大幅减少速度噪声
        )
        # 上一步动作
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置"""

    # === 启动时 ===
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0), # [参考 Go2] 扩大质量随机化范围 (原0.0, 0.5)
            "operation": "add",
        },
    )

    # === 重置时 ===
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
            # 注意：quat_mul(default_quat, delta_quat) 中 delta 在局部坐标系应用
            # V4的基础旋转绕X轴+90°后，"yaw"会变成绕世界-Y轴旋转（导致翻转）
            # 暂时禁用朝向随机化
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
            "position_range": (0.0, 0.0),  # 重置到默认位置（四足姿态），无偏移
            "velocity_range": (0.0, 0.0),
        },
    )

    # 显式设置 Waist_2 的 PD 位置目标为默认值（180°）
    # 这解决了 Waist_2 被排除在策略控制之外后，PD 目标默认为 0° 的问题
    reset_waist_target = EventTerm(
        func=reset_waist_joint_target,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === 间隔事件 ===
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """奖励配置 - 使用V4坐标系适配的奖励函数"""

    # === 任务奖励（正）— 权重要显著高于惩罚项总和，否则策略倾向站着不动 ===
    # V4适配：跟踪线速度命令（前进+左右）
    track_lin_vel_xy_exp = RewTerm(
        func=v4_track_lin_vel_xy_exp,
        weight=3.0,  # 从1.5提高到3.0，让前进跟踪成为主导奖励
        params={"command_name": "base_velocity", "std": 0.35},  # 从sqrt(0.25)=0.5收紧到0.35，要求更精确跟踪
    )
    # V4适配：跟踪角速度命令（转弯=绕局部Y轴）
    track_ang_vel_z_exp = RewTerm(
        func=v4_track_ang_vel_z_exp,
        weight=1.5,  # 从0.75提高到1.5
        params={"command_name": "base_velocity", "std": 0.35},
    )

    # === 姿态奖励 ===
    # 四足高度保持
    quadruped_height = RewTerm(
        func=quadruped_height_reward,
        weight=0.5,  # 从1.0降到0.5，减少对前进的抑制
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.22,
            "tolerance": 0.05,
        },
    )

    # 脚掌接地奖励：鼓励四只脚掌保持接触地面
    feet_contact = RewTerm(
        func=feet_contact_reward,
        weight=0.3,  # 从0.5降到0.3
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLEy|.*ARMAy"),
            "threshold": 1.0,
        },
    )

    # === 惩罚项（负）— 适度惩罚，不能压过跟踪奖励 ===
    # V4适配：惩罚上下方向线速度（局部Y=世界Z，不希望上下跳动）
    lin_vel_z_l2 = RewTerm(func=v4_lin_vel_y_l2, weight=-1.0)  # 从-2.0降到-1.0

    # V4适配：惩罚非转弯方向角速度（局部X和Z，不希望翻滚/俯仰）
    ang_vel_xy_l2 = RewTerm(func=v4_ang_vel_xz_l2, weight=-0.05)

    # 惩罚大力矩（节省能量）
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)  # 恢复到-0.0002，不要过度限制力矩

    # 惩罚大加速度（平滑动作）
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 恢复到-2.5e-7

    # 惩罚动作变化过快（平滑动作）— 抗抖动核心惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # 从-0.01提高到-0.05（适度，不要太大影响前进）

    # 惩罚动作变化的L1范数（抑制尖峰抖动）
    action_smoothness = RewTerm(func=action_smoothness_l2, weight=-0.02)

    # 惩罚关节速度过大（减少高频运动）
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0005,  # 从-0.001降到-0.0005
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 四足步态空中时间奖励
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,  # [参考 Go2] 从0.5降到0.25，避免过度抬腿
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*|.*ARMA.*"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # 不期望的接触（躯干触地）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link|Waist.*"),
            "threshold": 1.0,
        },
    )

    # V4适配：姿态保持（惩罚重力在局部X和Z分量，正确时应为0）
    flat_orientation_l2 = RewTerm(func=v4_flat_orientation_l2, weight=-2.5)  # [参考 Go2] 从-1.5增加到-2.5，加强平稳性要求

    # 关节位置限制惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # 腰部锁定惩罚：惩罚 Waist_2 偏离初始位置（180°），保持腰部不转
    waist_lock = RewTerm(func=waist_deviation_penalty, weight=-5.0)



@configclass
class TerminationsCfg:
    """终止条件配置"""

    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 躯干接触地面（摔倒）
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """课程学习配置（平地不需要地形课程）"""
    pass


# ============================================================
# Environment Configuration
# ============================================================

@configclass
class V4QuadrupedFlatEnvCfg(ManagerBasedRLEnvCfg):
    """V4四足行走平地环境配置"""

    # 场景
    scene: V4QuadrupedSceneCfg = V4QuadrupedSceneCfg(num_envs=4096, env_spacing=2.5)
    # 观测
    observations: ObservationsCfg = ObservationsCfg()
    # 动作
    actions: ActionsCfg = ActionsCfg()
    # 命令
    commands: CommandsCfg = CommandsCfg()
    # 奖励
    rewards: RewardsCfg = RewardsCfg()
    # 终止
    terminations: TerminationsCfg = TerminationsCfg()
    # 事件
    events: EventCfg = EventCfg()
    # 课程
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化"""
        # 通用设置
        self.decimation = 4  # 仿真4步，策略输出1次动作 → 50Hz动作频率
        self.episode_length_s = 20.0

        # 仿真设置
        self.sim.dt = 0.005  # 200Hz仿真频率
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器更新周期
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class V4QuadrupedFlatEnvCfg_PLAY(V4QuadrupedFlatEnvCfg):
    """V4四足行走平地环境配置（播放/评估模式）"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # 减少环境数量
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # 关闭观测噪声
        self.observations.policy.enable_corruption = False

        # 关闭随机推力
        self.events.base_external_force_torque = None
        self.events.push_robot = None
