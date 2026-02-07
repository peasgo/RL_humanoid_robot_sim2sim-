# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .rough_env_cfg import SoftfingerRoughEnvCfg


@configclass
class SoftfingerFlatEnvCfg(SoftfingerRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None #只需要平地
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None #删了高度扫描观测
        # no terrain curriculum
        self.curriculum.terrain_levels = None  #不需要地形变化

        # === 奖励：去掉“脚滞空”等腿足特有项，保留基础能量正则与动作平滑
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

@configclass
class SoftfingerFlatEnvCfg_PLAY(SoftfingerFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
