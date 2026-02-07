# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Rough Terrain Environment Configuration
====================================================

V3 人形机器人在复杂地形上的行走任务配置。
包含奖励函数、观察空间、终止条件等。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs - 导入 V3 机器人配置
##
from isaaclab_assets.robots.v3_humanoid import V3_HUMANOID_CFG, V3_HUMANOID_LEGS_ONLY_CFG


# ============================================================
# 自定义奖励函数
# ============================================================

def feet_separation_reward(
    env: ManagerBasedRLEnv, 
    target_distance: float, 
    std: float, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    奖励两脚之间的距离接近目标距离。
    防止双脚过于靠近或过于分开。
    """
    foot_pos = env.scene[asset_cfg.name].data.body_pos_w
    # 假设选中了左右脚 (num_envs, 2, 3)
    diff = foot_pos[:, 0, :] - foot_pos[:, 1, :]
    # 只看水平距离 (x, y)
    separation = torch.norm(diff[:, :2], dim=1)
    # 高斯核奖励
    error = separation - target_distance
    reward = torch.exp(-torch.square(error) / (std**2))
    return reward


def stand_height_reward(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    std: float, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    奖励躯干（Base）的高度接近目标高度。
    防止机器人蹲下或跳起。
    """
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    current_height = base_pos[:, 2]
    error = current_height - target_height
    reward = torch.exp(-torch.square(error) / (std**2))
    return reward


def upper_body_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: list,
) -> torch.Tensor:
    """
    奖励上半身关节保持在默认位置附近。
    用于行走任务中保持上半身稳定。
    """
    robot = env.scene[asset_cfg.name]
    # 获取指定关节的位置
    joint_pos = robot.data.joint_pos
    # 计算与默认位置的偏差
    deviation = torch.abs(joint_pos).sum(dim=1)
    # 使用指数衰减奖励
    reward = torch.exp(-deviation * 0.1)
    return reward


def arm_swing_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    奖励手臂与腿部的协调摆动（自然行走姿态）。
    左臂与右腿同步，右臂与左腿同步。
    """
    robot = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    
    # 这里需要根据实际关节索引调整
    # 简化版：奖励手臂有适度的运动
    arm_vel = torch.abs(joint_vel[:, :12]).mean(dim=1)  # 假设前12个是手臂关节
    reward = torch.clamp(arm_vel * 0.1, max=1.0)
    return reward


# ============================================================
# 奖励配置类
# ============================================================

@configclass
class V3HumanoidRewards(RewardsCfg):
    """V3 人形机器人行走任务的奖励配置"""

    # === 基础奖励/惩罚 ===
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    alive = RewTerm(func=mdp.is_alive, weight=5.0)

    # === 速度跟踪奖励 ===
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # === 步态奖励 ===
    gait_alternating = RewTerm(
        func=mdp.alternating_gait_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LANKLEy", "RANKLEy"]),
            "std": 0.2,
            "max_err": 0.3,
        },
    )

    # === 单腿支撑时间惩罚 ===
    single_stance_air_time_penalty = RewTerm(
        func=mdp.single_stance_air_time_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LANKLEy", "RANKLEy"]),
            "threshold": 0.35,
        },
    )

    # === 滑动惩罚 ===
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LANKLEy", "RANKLEy"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["LANKLEy", "RANKLEy"]),
        },
    )

    # === 关节限位惩罚 ===
    # 踝关节限位（更严格）
    dof_pos_limits_ankle = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ANKLE.*"])},
    )

    # 全关节限位
    dof_pos_limits_all = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # === 姿态维持 ===
    # 髋部关节偏差惩罚
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIPy", ".*HIPr"])},
    )

    # 全关节偏差惩罚
    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # === 站立高度奖励 ===
    stand_height = RewTerm(
        func=stand_height_reward,
        weight=1.0,
        params={
            "target_height": 0.35,  # 根据机器人实际高度调整
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # === 双脚间距奖励 ===
    feet_separation = RewTerm(
        func=feet_separation_reward,
        weight=0.8,
        params={
            "target_distance": 0.15,  # 根据机器人宽度调整
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=["LANKLEy", "RANKLEy"]),
        },
    )

    # === 上半身稳定性（可选） ===
    # 如果需要上半身保持稳定，取消注释
    # upper_body_stability = RewTerm(
    #     func=upper_body_stability_reward,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "joint_names": ["Waist", "Waist_2", ".*SD.*", ".*ARM.*"],
    #     },
    # )

    # === 禁用不需要的奖励 ===
    joint_deviation_arms = None
    joint_deviation_torso = None


# ============================================================
# 环境配置类
# ============================================================

@configclass
class V3HumanoidRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """V3 人形机器人复杂地形环境配置"""
    
    rewards: V3HumanoidRewards = V3HumanoidRewards()

    def __post_init__(self):
        super().__post_init__()
        
        # === 场景配置 - 使用 V3 机器人 ===
        self.scene.robot = V3_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # === 随机化配置 ===
        self.events.push_robot = None  # 禁用推力扰动（初期训练）
        self.events.base_com = None
        self.events.add_base_mass = None
        
        # 物理材质随机化
        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        
        # 关节初始化随机化
        self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        
        # 基座重置配置
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # === 奖励权重调整 ===
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.feet_air_time = None

        # === 命令范围配置 ===
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # === 终止条件 ===
        # 当这些部位接触地面时终止
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link", "Waist", "Waist_2",
            "RHIPp", "RHIPy", "RHIPr", "RKNEEP",
            "LHIPp", "LHIPy", "LHIPr", "LKNEEp",
        ]


@configclass
class V3HumanoidRoughEnvCfg_PLAY(V3HumanoidRoughEnvCfg):
    """用于测试/演示的配置"""
    
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # 固定命令用于测试
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
