# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm , SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, RewardsCfg,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# 用你自己的软指模型
from isaaclab_assets.robots.softfinger import SOFTFINGER_CFG


# @configclass
# class SoftfingerActionsCfg:
#     joint_pos = mdp.JointPositionActionCfg(
#         asset_name="robot",
#         joint_names=[".*"],
#         # joint_names=[
#         #     "F2B_F3B",
#         #     "F3B_F4B",
#         #     "F2B_F1B",
#         #     "F1B_F1P",
#         #     "FTB_FTM",
#         #     "FT2T_FTB",
#         #     "F1B_FT2T",
#         #     "F2B_F2P",
#         #     "F3B_F3P",
#         #     "F4B_F4P",
#         # ],
#         scale=0.20,
#         use_default_offset=True,
#     )


@configclass
class SoftfingerRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=10.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    # )
    height_F2B = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="F2B"),
            "sensor_cfg": None,
        },
    )
    height_F3B = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="F3B"),
            "sensor_cfg": None,
        },
    )
    height_F1B = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="F1B"),
            "sensor_cfg": None,
        },
    )
    height_F4B = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="F4B"),
            "sensor_cfg": None,
        },
    )

    # height_FTB = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-50.0,
    #     params={
    #         "target_height": 0.03,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="FTB"),
    #         "sensor_cfg": None,
    #     },
    # )

    height_F1P = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.02,
            "asset_cfg": SceneEntityCfg("robot", body_names="F1P"),
            "sensor_cfg": None,
        },
    )
    height_F2P = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.02,
            "asset_cfg": SceneEntityCfg("robot", body_names="F2P"),
            "sensor_cfg": None,
        },
    )
    height_F3P = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.02,
            "asset_cfg": SceneEntityCfg("robot", body_names="F3P"),
            "sensor_cfg": None,
        },
    )
    height_F4P = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": 0.02,
            "asset_cfg": SceneEntityCfg("robot", body_names="F4P"),
            "sensor_cfg": None,
        },
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.25, # 权重给的大一点，有助于把腿台的高一些
    #     params={ 
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.4,
    #     },
    # )
    # feet_slide = RewTerm( # 不希望腿在地面上滑动的
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # # Penalize ankle joint limits
    # dof_pos_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    # )
    # # Penalize deviation from default of the joints that are not essential for locomotion
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1, # 惩罚髋关节偏离默认位置 deviation:偏差
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    # )
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #                 ".*_shoulder_roll_joint",
    #                 ".*_shoulder_yaw_joint",
    #                 ".*_elbow_pitch_joint",
    #                 ".*_elbow_roll_joint",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1, # 旨在训练为：这些关节是不需要动的
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_five_joint",
    #                 ".*_three_joint",
    #                 ".*_six_joint",
    #                 ".*_four_joint",
    #                 ".*_zero_joint",
    #                 ".*_one_joint",
    #                 ".*_two_joint",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_torso = RewTerm( # 腰&躯干不需要动
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    # )


@configclass
class SoftfingerRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: SoftfingerRewards = SoftfingerRewards()
    # actions: SoftfingerActionsCfg = SoftfingerActionsCfg()  # 👈 绑定新动作表

    def __post_init__(self):
        super().__post_init__()

        # ===(1) 用软指USD，而不是GO2===========================================
        self.scene.robot = SOFTFINGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ===(2) 地形/扫描器：软指不是腿足机器人，默认关掉高度扫描避免找不到base报错
        self.scene.height_scanner.prim_path = None

        # # ===(3) 地形稍微柔和一些，防止初期弹飞
        # if self.scene.terrain.terrain_generator is not None:
        #     tg = self.scene.terrain.terrain_generator
        #     if "boxes" in tg.sub_terrains:
        #         tg.sub_terrains["boxes"].grid_height_range = (0.01, 0.05)
        #     if "random_rough" in tg.sub_terrains:
        #         tg.sub_terrains["random_rough"].noise_range = (0.005, 0.03)
        #         tg.sub_terrains["random_rough"].noise_step = 0.005

        # ===(4) 动作幅度：把关节位置动作缩小一点，便于稳定探索
        self.actions.joint_pos.scale = 0.20

        # ===(5) 事件：移除对“base/脚”的依赖事件，保留通用的复位
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None

        # 复位范围更温和
        self.events.reset_robot_joints.params["position_range"] = (-0.5, 0.5)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }


        # ===(6) 奖励：去掉“脚滞空”等腿足特有项，保留基础能量正则与动作平滑
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0) # 给了前进速度
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0) # 不要横向速度
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0) # 给了转向速度

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["FTB","F1B","F2B","F3B","F4B"]
        self.terminations.base_contact.params["threshold"] =15.0  #种植条件：底座发生碰撞

        # ===(8) 观测：默认观测即可；若你的USD没有惯性/IMU，保持最小观测
        # self.observations.policy 的 height_scan 若存在也不使用（已在上方关掉）


@configclass
class SoftfingerRoughEnvCfg_PLAY(SoftfingerRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Playground：更少环境、更紧凑、无扰动
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0 # evaluation时间拉长一些

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None  # 设置为 None 以在评估时随机生成机器人位置，而不是根据地形等级分配
        
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            tg = self.scene.terrain.terrain_generator
            tg.num_rows = 5
            tg.num_cols = 5
            tg.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0) #朝向

        # disable randomization for play
        self.observations.policy.enable_corruption = False # 评估时不需要观测扰动
        # remove random pushing
        self.events.base_external_force_torque = None # 可加可不加
        self.events.push_robot = None

