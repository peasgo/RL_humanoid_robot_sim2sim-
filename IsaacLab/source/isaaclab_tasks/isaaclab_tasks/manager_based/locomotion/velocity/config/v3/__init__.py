# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Locomotion Task Registration
========================================

注册 V3 人形机器人的行走任务环境。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 平地环境 - 全身控制
gym.register(
    id="Isaac-Velocity-Flat-V3Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:V3HumanoidFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:V3HumanoidFlatPPORunnerCfg",
    },
)

# 复杂地形环境 - 全身控制
gym.register(
    id="Isaac-Velocity-Rough-V3Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:V3HumanoidRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:V3HumanoidRoughPPORunnerCfg",
    },
)


