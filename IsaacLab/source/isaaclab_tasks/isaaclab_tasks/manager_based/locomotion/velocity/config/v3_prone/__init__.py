# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Stand-to-Prone Task Registration
=============================================

注册 V3 人形机器人的站立到倒地（Stand-to-Prone）任务环境。
这是一个多阶段动作任务：站立 -> 深蹲 -> 跪地 -> 俯卧/四足
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 站立到俯卧任务 - 完整过程
gym.register(
    id="Isaac-V3Humanoid-StandToProne-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.prone_env_cfg:V3StandToProneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:V3StandToPronePPORunnerCfg",
    },
)

# 四足爬行任务 - 从俯卧姿态开始
gym.register(
    id="Isaac-V3Humanoid-Crawling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.prone_env_cfg:V3CrawlingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:V3CrawlingPPORunnerCfg",
    },
)

# 课程学习版本 - 分阶段训练
gym.register(
    id="Isaac-V3Humanoid-StandToProne-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.prone_env_cfg:V3StandToProneCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:V3StandToPronePPORunnerCfg",
    },
)
