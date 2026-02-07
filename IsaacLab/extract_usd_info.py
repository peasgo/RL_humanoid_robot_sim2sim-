#!/usr/bin/env python3
"""提取USD文件的初始信息"""

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="提取USD文件的初始信息")
parser.add_argument("usd_path", type=str, help="USD文件路径")
parser.add_argument("--prim_path", type=str, default="/World/Robot", help="Prim路径")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.sim.spawners.from_files import UsdFileCfg
import omni.isaac.lab.sim as sim_utils

def extract_usd_info(usd_path: str, prim_path: str):
    """提取USD文件信息"""

    # 创建仿真环境
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 配置机器人
    robot_cfg = ArticulationCfg(
        spawn=UsdFileCfg(usd_path=usd_path),
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(),
    )

    # 创建机器人
    robot = Articulation(cfg=robot_cfg)

    # 重置场景
    sim.reset()
    robot.reset()

    print("\n" + "="*60)
    print("USD文件信息提取")
    print("="*60)
    print(f"文件路径: {usd_path}")
    print(f"Prim路径: {prim_path}")

    # 基本信息
    print(f"\n【基本信息】")
    print(f"机器人数量: {robot.num_instances}")
    print(f"刚体数量: {robot.num_bodies}")
    print(f"关节数量: {robot.num_joints}")

    # 刚体名称
    print(f"\n【刚体名称】")
    for i, name in enumerate(robot.body_names):
        print(f"  {i}: {name}")

    # 关节名称
    print(f"\n【关节名称】")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i}: {name}")

    # 关节属性
    print(f"\n【关节默认属性】")
    print(f"默认关节位置(tensor):\n{robot.data.default_joint_pos}")
    print(f"关节位置限制(tensor):\n{robot.data.default_joint_pos_limits}")
    print(f"关节刚度:\n{robot.data.default_joint_stiffness}")
    print(f"关节阻尼:\n{robot.data.default_joint_damping}")

    # 按关节名逐项打印 + 检查是否越界（用于验证 USD 默认偏移/默认姿态是否异常）
    print(f"\n【逐关节默认位置检查】")
    default_pos = robot.data.default_joint_pos[0].detach().cpu()
    limits = robot.data.default_joint_pos_limits[0].detach().cpu()
    oob_count = 0
    for i, name in enumerate(robot.joint_names):
        val = float(default_pos[i])
        jmin = float(limits[i, 0])
        jmax = float(limits[i, 1])
        out_of_range = (val < jmin) or (val > jmax)
        if out_of_range:
            oob_count += 1
        status = "OUT_OF_RANGE" if out_of_range else "OK"
        print(f"  {i:02d} {name:<16s} default={val: .6f} range=[{jmin: .6f}, {jmax: .6f}] {status}")
    if oob_count > 0:
        print(f"\n[WARN] 有 {oob_count}/{robot.num_joints} 个关节默认位置越界。若训练里 action use_default_offset=True，偏移会直接注入动作导致不稳定。")
    else:
        print(f"\n[OK] 所有关节默认位置都在范围内。")

    # 刚体属性
    print(f"\n【刚体默认属性】")
    print(f"质量:\n{robot.data.default_mass}")
    print(f"惯性:\n{robot.data.default_inertia}")

    # 根状态
    print(f"\n【根状态】")
    print(f"根位置: {robot.data.default_root_state[:, :3]}")
    print(f"根方向(四元数): {robot.data.default_root_state[:, 3:7]}")

    print("\n" + "="*60)

if __name__ == "__main__":
    extract_usd_info(args_cli.usd_path, args_cli.prim_path)
    simulation_app.close()
