# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Flat Terrain Environment Configuration
===================================================

V3 人形机器人在平地上的行走任务配置。
继承自复杂地形配置，简化地形和观察空间。
"""

from isaaclab.utils import configclass
from .rough_env_cfg import V3HumanoidRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.v3_humanoid import V3_HUMANOID_CFG, V3_HUMANOID_LEGS_ONLY_CFG


@configclass
class V3HumanoidFlatEnvCfg(V3HumanoidRoughEnvCfg):
    """V3 人形机器人平地环境配置"""
    
    def __post_init__(self):
        super().__post_init__()

        # === 改为平地 ===
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # === 禁用高度扫描 ===
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        
        # === 禁用地形课程 ===
        self.curriculum.terrain_levels = None
        
        # === 调整奖励（平地可以更激进） ===
        if self.rewards.feet_air_time is not None:
            self.rewards.feet_air_time.weight = 1.0
            self.rewards.feet_air_time.params["threshold"] = 0.6


@configclass
class V3HumanoidFlatEnvCfg_PLAY(V3HumanoidFlatEnvCfg):
    """用于测试的平地配置"""
    
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        
        # 固定命令
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        # 禁用随机化
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


