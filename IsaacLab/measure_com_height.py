#!/usr/bin/env python3
"""测量机器人实际质心高度"""

import torch
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="ph", choices=["ph", "softfinger"])
parser.add_argument("--headless", action="store_true", default=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv

if args_cli.robot == "ph":
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ph.rough_env_cfg import ParallelhumanRoughEnvCfg
    env_cfg = ParallelhumanRoughEnvCfg()
else:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.softfinger.rough_env_cfg import SoftfingerRoughEnvCfg
    env_cfg = SoftfingerRoughEnvCfg()

env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)

# 重置环境
env.reset()

# 获取数据
robot = env.scene["robot"]
root_link_pos = robot.data.root_pos_w[0]
root_com_pos = robot.data.root_com_pos_w[0]
body_com_pos_b = robot.data.body_com_pos_b[0, 0]  # 第一个body的质心偏移

print("=" * 70)
print(f"机器人: {args_cli.robot.upper()}")
print("=" * 70)
print(f"\nRoot Link位置 (世界坐标):")
print(f"  X: {root_link_pos[0]:.4f} m")
print(f"  Y: {root_link_pos[1]:.4f} m")
print(f"  Z: {root_link_pos[2]:.4f} m")

print(f"\n质心位置 (世界坐标):")
print(f"  X: {root_com_pos[0]:.4f} m")
print(f"  Y: {root_com_pos[1]:.4f} m")
print(f"  Z: {root_com_pos[2]:.4f} m")

print(f"\n质心相对Root Link的偏移 (body frame):")
print(f"  X: {body_com_pos_b[0]:.4f} m")
print(f"  Y: {body_com_pos_b[1]:.4f} m")
print(f"  Z: {body_com_pos_b[2]:.4f} m")

print(f"\n关键结论:")
print(f"  Root Link高度: {root_link_pos[2]:.4f} m")
print(f"  质心高度: {root_com_pos[2]:.4f} m")
print(f"  高度差: {root_com_pos[2] - root_link_pos[2]:.4f} m")

env.close()
simulation_app.close()
