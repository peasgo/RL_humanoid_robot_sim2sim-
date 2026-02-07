# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Stand-to-Prone Task Agent Configurations
=====================================================

导出站立到倒地任务的训练配置。
"""

from .rsl_rl_ppo_cfg import (
    V3StandToPronePPORunnerCfg,
    V3CrawlingPPORunnerCfg,
    V3StandToProneCurriculumPPORunnerCfg,
)
