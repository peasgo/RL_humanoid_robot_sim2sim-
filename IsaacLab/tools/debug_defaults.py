# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal script to check robot default joint positions directly from Articulation View.
"""

import argparse
from isaaclab.app import AppLauncher

# 添加参数解析
parser = argparse.ArgumentParser(description="Check robot default joint positions.")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the robot USD file.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动仿真应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext, SimulationCfg

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # 简单的 Articulation 配置，不加 actuators 也不加别的，只读数据
    # 注意：如果不加 actuators，Articulation 类可能会在 post_init 报错
    # 所以我们手动绕过 Articulation 类，直接用 USD API 或者 Physics API 检查，
    # 但为了模拟 Environment 的行为，我还是得用 Articulation。
    # 这次我加上一个绝对没问题的 Dummy Actuator 配置。
    
    from isaaclab.actuators import ImplicitActuatorCfg

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=args.usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
        }
    )

    robot = Articulation(robot_cfg)
    sim.reset()
    
    # 必须 step 几次让物理引擎初始化
    for _ in range(5):
        sim.step()

    # ---------------------------------------------------------
    # 核心检查逻辑
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(" INSPECTING JOINT DEFAULTS (What use_default_offset=True sees)")
    print("="*80)
    
    # 1. 直接打印 robot.data.default_joint_pos
    # 这是 IsaacLab 从 USD 解析出来的“默认位置”
    default_pos = robot.data.default_joint_pos[0]
    joint_names = robot.joint_names
    
    for i, name in enumerate(joint_names):
        val = default_pos[i].item()
        print(f"Joint: {name:<20} | Default Pos: {val:.4f} rad")
    
    print("="*80 + "\n")
    simulation_app.close()

if __name__ == "__main__":
    main()
