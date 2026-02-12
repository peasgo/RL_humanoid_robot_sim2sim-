#!/usr/bin/env python3
"""在IsaacLab中测试策略的方向响应。
使用最小化的环境配置，减少启动时间。
"""
import sys
import os
import time

# 设置环境
os.environ["ISAACSIM_PATH"] = "/home/rl/.local/share/ov/pkg/isaac-sim-4.5.0"
os.environ["ISAACSIM_PYTHON_EXE"] = "/home/rl/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh"

# 添加IsaacLab路径
sys.path.insert(0, "/home/rl/RL-human_robot/IsaacLab/source/isaaclab")
sys.path.insert(0, "/home/rl/RL-human_robot/IsaacLab/source/isaaclab_tasks")
sys.path.insert(0, "/home/rl/RL-human_robot/IsaacLab/source/isaaclab_assets")

import torch
import numpy as np

# 不启动IsaacLab，直接用策略+手动构建obs来测试
# 这样可以验证策略在"理想obs"下的行为

policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/exported/policy.pt"
policy = torch.jit.load(policy_path, map_location="cpu")
policy.eval()

print("策略加载成功")

# 模拟IsaacLab中的obs：
# 在IsaacLab中，初始状态下：
# - lin_vel_b = [0, 0, 0] (静止)
# - ang_vel_b = [0, 0, 0] (静止)
# - gravity_b = 对于V4 (quat=[0.7071, 0.7071, 0, 0])
#   projected_gravity = quat_apply_inverse(quat, [0,0,-1])
#   V4 base_link绕X轴旋转90°，所以gravity在body frame中是 [0, -1, 0]
#   V4 remap: [gz, gx, gy] = [0, 0, -1] → 不对
#   让我重新计算...

# V4 quat = [w,x,y,z] = [0.7071, 0.7071, 0, 0]
# 这是绕X轴旋转90°
# R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
# gravity_world = [0, 0, -1]
# gravity_body = R^T @ [0, 0, -1] = [[1,0,0],[0,0,1],[0,-1,0]] @ [0,0,-1] = [0, -1, 0]
# V4 remap gravity: [gz, gx, gy] = [0, 0, -1]
# 等等，gravity_body = [0, -1, 0]
# V4 remap: [v[2], v[0], v[1]] = [0, 0, -1]

# 用get_gravity_orientation验证
def get_gravity(q):
    w, x, y, z = q
    return np.array([-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))])

quat = [0.70710678, 0.70710678, 0.0, 0.0]
grav = get_gravity(quat)
print(f"gravity_body = {grav}")
# V4 remap
grav_obs = np.array([grav[2], grav[0], grav[1]])
print(f"gravity_obs (V4 remap) = {grav_obs}")

# 构建理想的初始obs (IsaacLab中的状态)
def make_ideal_obs(cmd, joint_pos_rel=None, joint_vel=None, last_action=None):
    obs = np.zeros(62, dtype=np.float32)
    # lin_vel = [0, 0, 0] (静止)
    obs[0:3] = [0, 0, 0]
    # ang_vel = [0, 0, 0] (静止)
    obs[3:6] = [0, 0, 0]
    # gravity (V4 remap)
    obs[6:9] = grav_obs
    # cmd
    obs[9:12] = cmd
    # joint_pos_rel = 0 (在default位置)
    if joint_pos_rel is not None:
        obs[12:29] = joint_pos_rel
    # joint_vel = 0
    if joint_vel is not None:
        obs[29:46] = joint_vel
    # last_action = 0
    if last_action is not None:
        obs[46:62] = last_action
    return obs

# 测试不同cmd
cmds = [
    ("FWD 0.5",  [0.5, 0, 0]),
    ("FWD 0.3",  [0.3, 0, 0]),
    ("STAND",    [0, 0, 0]),
    ("BWD 0.3",  [-0.3, 0, 0]),
    ("BWD 0.5",  [-0.5, 0, 0]),
    ("LEFT 0.3", [0, 0.3, 0]),
    ("TURN_L",   [0, 0, 0.5]),
]

print("\n" + "="*80)
print("  理想obs下的策略输出 (零状态 + 不同cmd)")
print("="*80)

isaac16 = ['LHIPp','RHIPp','LHIPy','RHIPy',
           'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
           'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']

for name, cmd in cmds:
    obs = make_ideal_obs(cmd)
    with torch.no_grad():
        action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
    print(f"\n  {name}: cmd={cmd}")
    print(f"    action L2={np.linalg.norm(action):.4f}")
    for i in range(16):
        print(f"    {isaac16[i]:10s}: {action[i]:+.4f}", end="")
        if (i+1) % 4 == 0:
            print()

# 多步模拟（不用物理引擎，只看策略的闭环行为）
print("\n" + "="*80)
print("  多步策略推理（无物理，只看action演化）")
print("="*80)

for name, cmd in [("FWD 0.5", [0.5,0,0]), ("BWD 0.5", [-0.5,0,0]), ("STAND", [0,0,0])]:
    obs = make_ideal_obs(cmd)
    last_action = np.zeros(16, dtype=np.float32)
    
    print(f"\n  {name}:")
    for step in range(10):
        obs[46:62] = last_action
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
        last_action = action.copy()
        
        # 简单模拟：假设关节完美跟踪，更新joint_pos_rel
        action_scale = 0.25
        obs[12:28] = action[:16] * action_scale  # 近似：joint_pos_rel ≈ action * scale
        
        print(f"    Step {step}: action L2={np.linalg.norm(action):.4f}, "
              f"HIP_L={action[0]:+.3f}, HIP_R={action[1]:+.3f}, "
              f"KNEE_L={action[6]:+.3f}, KNEE_R={action[7]:+.3f}, "
              f"ANKLE_L={action[10]:+.3f}, ANKLE_R={action[11]:+.3f}")

# 关键测试：用MuJoCo的实际obs（从trace_obs_diff.py的结果）
# 看策略在MuJoCo obs下的行为是否与理想obs下不同
print("\n" + "="*80)
print("  对比：理想obs vs MuJoCo初始obs")
print("="*80)

# MuJoCo step 0的obs（从trace_obs_diff.py）
mj_obs_step0 = np.zeros(62, dtype=np.float32)
mj_obs_step0[0:3] = [-0.14228, 0.29072, 0.37997]  # lin_vel (非零！)
mj_obs_step0[3:6] = [0, 0, 0]  # ang_vel (近似)
mj_obs_step0[6:9] = grav_obs  # gravity
# cmd和其他分量在下面设置

for name, cmd in [("FWD 0.5", [0.5,0,0]), ("BWD 0.5", [-0.5,0,0])]:
    ideal_obs = make_ideal_obs(cmd)
    mj_obs = mj_obs_step0.copy()
    mj_obs[9:12] = cmd
    
    with torch.no_grad():
        ideal_action = policy(torch.from_numpy(ideal_obs).unsqueeze(0)).numpy().squeeze()
        mj_action = policy(torch.from_numpy(mj_obs).unsqueeze(0)).numpy().squeeze()
    
    diff = mj_action - ideal_action
    print(f"\n  {name}:")
    print(f"    理想obs action L2={np.linalg.norm(ideal_action):.4f}")
    print(f"    MuJoCo obs action L2={np.linalg.norm(mj_action):.4f}")
    print(f"    差异 L2={np.linalg.norm(diff):.4f}")
    print(f"    理想: HIP_L={ideal_action[0]:+.3f}, HIP_R={ideal_action[1]:+.3f}")
    print(f"    MuJoCo: HIP_L={mj_action[0]:+.3f}, HIP_R={mj_action[1]:+.3f}")
