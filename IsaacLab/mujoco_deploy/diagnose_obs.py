import sys
import numpy as np
sys.path.insert(0, '/home/rl/RL-human_robot/IsaacLab/mujoco_deploy')
from deploy_mujoco import MujocoDeployer

XML_PATH = "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy/WholeAssembleV2_mujoco.xml"
POLICY_PATH = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-28_10-29-50/exported/policy.pt"

deployer = MujocoDeployer(XML_PATH, POLICY_PATH)
deployer.reset()

obs = deployer.get_obs()
obs_np = obs.cpu().numpy().squeeze()

print("\n=== 观测空间诊断 ===")
print(f"观测维度: {obs_np.shape}")
print(f"\n观测分量:")
idx = 0
print(f"[{idx:2d}-{idx+2:2d}] base_lin_vel: {obs_np[idx:idx+3]}")
idx += 3
print(f"[{idx:2d}-{idx+2:2d}] base_ang_vel: {obs_np[idx:idx+3]}")
idx += 3
print(f"[{idx:2d}-{idx+2:2d}] gravity:      {obs_np[idx:idx+3]}")
idx += 3
print(f"[{idx:2d}-{idx+2:2d}] commands:     {obs_np[idx:idx+3]}")
idx += 3
print(f"[{idx:2d}-{idx+11:2d}] joint_pos:    {obs_np[idx:idx+12]}")
idx += 12
print(f"[{idx:2d}-{idx+11:2d}] joint_vel:    {obs_np[idx:idx+12]}")
idx += 12
print(f"[{idx:2d}-{idx+11:2d}] last_action:  {obs_np[idx:idx+12]}")

print(f"\n观测统计:")
print(f"  min: {obs_np.min():.3f}")
print(f"  max: {obs_np.max():.3f}")
print(f"  mean: {obs_np.mean():.3f}")
print(f"  std: {obs_np.std():.3f}")

import torch
with torch.no_grad():
    action = deployer.policy(obs).cpu().numpy().squeeze()

print(f"\n策略输出:")
print(f"  action: {action}")
print(f"  min: {action.min():.3f}, max: {action.max():.3f}")
