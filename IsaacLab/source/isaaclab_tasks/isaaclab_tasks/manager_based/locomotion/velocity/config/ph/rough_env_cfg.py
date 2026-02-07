# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.parallelhuman import PARALLELHUMAN_CFG

def feet_separation_reward(env: ManagerBasedRLEnv, target_distance: float, std: float, asset_cfg: SceneEntityCfg):
    """
    奖励两脚之间的距离接近目标距离。
    """
    # 1. 获取脚的位置
    # env.scene.sensors[asset_cfg.name] 会获取我们在配置里指定的刚体（脚）的数据
    # data.body_pos_w 是世界坐标系下的位置 (num_envs, num_bodies, 3)
    foot_pos = env.scene[asset_cfg.name].data.body_pos_w
    # 假设我们通过正则选到了左脚和右脚 (num_envs, 2, 3)
    # 计算两脚之间的向量差
    diff = foot_pos[:, 0, :] - foot_pos[:, 1, :]
    
    # 计算距离 (只看 XY 平面，忽略高度差，或者看整体距离)
    # 这里我们只关心水平距离 (x, y)，索引 0 和 1
    separation = torch.norm(diff[:, :2], dim=1)
    
    # 2. 计算奖励 (高斯核：距离越接近 target，奖励越接近 1)
    error = separation - target_distance
    reward = torch.exp(-torch.square(error) / (std**2))
    
    return reward

def stand_height_reward(env: ManagerBasedRLEnv, target_height: float, std: float, asset_cfg: SceneEntityCfg):
    """
    奖励躯干（Base）的高度接近目标高度。
    """
    # 获取机器人基座（Base）的位置
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    
    # 获取 Z 轴高度 (索引 2)
    current_height = base_pos[:, 2]
    
    # 计算奖励
    error = current_height - target_height
    reward = torch.exp(-torch.square(error) / (std**2))
    
    return reward

def com_height_reward(env: ManagerBasedRLEnv, target_height: float, std: float, asset_cfg: SceneEntityCfg):
    """
    奖励质心高度接近目标高度。
    """
    # 获取机器人质心位置（root COM）
    com_pos = env.scene[asset_cfg.name].data.root_com_pose_w[:, :3]
    current_height = com_pos[:, 2]

    error = current_height - target_height
    reward = torch.exp(-torch.square(error) / (std**2))

    return reward



@configclass
class ParallelhumanRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    alive = RewTerm(
        func=mdp.is_alive, 
        weight=5.0  # 权重可以根据你的需求调整，通常在 1.0 ~ 5.0 之间
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # --- 交替步态奖励（对称/交替） ---
    gait_alternating = RewTerm(
        func=mdp.alternating_gait_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot.*"),
            "std": 0.2,
            "max_err": 0.3,
        },
    )

    single_stance_air_time_penalty = RewTerm(
        func=mdp.single_stance_air_time_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot.*"),
            "threshold": 0.35,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            # 修改点：连杆名匹配 LFootR, RFootR 等
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot.*"),
        },
    )

    # --- 关节限位惩罚（踝关节更严格） ---
    # 你的踝关节名为 RAankleP, RAnkleR, LAnkleP, LAnkleR
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Ankle.*")},
    )

    # 全关节限位惩罚（避免其他关节靠近极限）
    dof_pos_limits_all = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # --- 髋部偏航/横滚关节姿态维持 ---
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipY", ".*HipR"])},
    )


    # 约束全关节偏离默认姿态（避免关节大幅度偏离）
    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


    # 1. 站立高度奖励 (Anti-Crouch)
    # 逻辑：目标高度设为 0.3 (对应训练时的配置)
    stand_height = RewTerm(
        func=stand_height_reward, # 调用上面写的函数
        weight=1.0,               # 正权重表示奖励
        params={
            "target_height": 0.31, # 修改为 0.27，保持微屈 (原0.30)
            "std": 0.05,            # 容忍度，越小要求越严
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"), # 指定获取机器人的数据
        },
    )

    # 1.1 质心高度奖励 (移除以避免重复)
    com_height = None


    # 2. 双脚间距奖励 (Feet Separation)
    # 逻辑：希望两脚分开大约 0.15 米 (根据你的机器人宽度调整)
    feet_separation = RewTerm(
        func=feet_separation_reward,
        weight=1.0,
        params={
            "target_distance": 0.13, # 比如希望双脚间距 15cm
            "std": 0.05,            # 这是一个比较严格的限制
            # 重要：这里要通过名字找到左脚和右脚
            # 你的 parallelhuman.py 里踝关节叫 .*Ankle.*，通常刚体也是类似的
            # 这里我们要选中脚部刚体。如果不确定刚体名，可以用 ".*Foot.*" 试试
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot.*"), 
        },
    )

    # 你的模型暂时没有手臂，建议将权重设为 None 或删除以防报错
    joint_deviation_arms = None 

    # 你的模型暂时没有独立的 torso 关节，设为 None
    joint_deviation_torso = None


@configclass
class ParallelhumanRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ParallelhumanRewards = ParallelhumanRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = PARALLELHUMAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # if self.scene.height_scanner:
        #     self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.base_com = None
        # 1. 开启质量随机化 (Add Base Mass)
        # ----------------------------------------------------------------
        # 原代码：self.events.add_base_mass = None  <-- 注释掉或删除这行
        
        # 新代码：重新配置参数
        # 注意：一定要确认你的机器人根节点叫什么，通常是 "base_link"
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        # 设置质量变化的范围 (kg)，例如在原质量基础上 +/- 5kg
        self.events.add_base_mass = None


        # 2. 开启物理材质/摩擦力随机化 (Physics Material)
        # ----------------------------------------------------------------
        # 你的原代码里其实没有把 physics_material 设为 None，所以它默认其实是开启的。
        # 但为了保险起见，或者如果你想修改摩擦力范围，可以显式地配置它：
        
        # 指定作用于所有刚体 (".*")
        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        # 设置静摩擦力范围 (默认是 0.8)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        # 设置动摩擦力范围 (默认是 0.6)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.feet_air_time = None



        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        # Terminations
        # 找到你的 flat_env_cfg.py 中的 base_contact 部分
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link","LThighP", "LThighY", "LThighR", "LShankP", 
            "RThighP", "RThighY", "RThighR", "RShankP"
        ]


@configclass
class ParallelhumanRoughEnvCfg_PLAY(ParallelhumanRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
