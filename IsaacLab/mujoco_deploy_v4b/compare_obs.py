#!/usr/bin/env python3
"""对比IsaacLab记录的obs和MuJoCo的obs，找出差异。"""
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=150)

# 加载数据
isaac_data = np.load("IsaacLab/mujoco_deploy_v4/isaaclab_reference_obs.npz")
mujoco_data = np.load("IsaacLab/mujoco_deploy_v4/mujoco_obs.npz")

print("IsaacLab数据 keys:", list(isaac_data.keys()))
print("MuJoCo数据 keys:", list(mujoco_data.keys()))

for k in isaac_data.keys():
    v = isaac_data[k]
    print(f"  isaac[{k}]: shape={v.shape}, dtype={v.dtype}")

for k in mujoco_data.keys():
    v = mujoco_data[k]
    print(f"  mujoco[{k}]: shape={v.shape}, dtype={v.dtype}")

# 对比obs
if 'obs' in isaac_data and 'obs' in mujoco_data:
    isaac_obs = isaac_data['obs']
    mujoco_obs = mujoco_data['obs']
    
    n = min(len(isaac_obs), len(mujoco_obs))
    print(f"\n对比前{n}步的obs:")
    
    # obs结构: [lin_vel(3), ang_vel(3), gravity(3), cmd(3), joint_pos_rel(17), joint_vel(17), last_action(16)]
    segments = [
        (0, 3, "lin_vel"),
        (3, 6, "ang_vel"),
        (6, 9, "gravity"),
        (9, 12, "cmd"),
        (12, 29, "joint_pos_rel"),
        (29, 46, "joint_vel"),
        (46, 62, "last_action"),
    ]
    
    for step in range(min(5, n)):
        print(f"\n--- Step {step} ---")
        io = isaac_obs[step]
        mo = mujoco_obs[step]
        
        for start, end, name in segments:
            diff = io[start:end] - mo[start:end]
            max_diff = np.max(np.abs(diff))
            print(f"  {name:16s}: isaac={io[start:end]}")
            print(f"  {' ':16s}: mujoco={mo[start:end]}")
            if max_diff > 0.01:
                print(f"  {' ':16s}: DIFF={diff}  max={max_diff:.4f} !!!")
            else:
                print(f"  {' ':16s}: max_diff={max_diff:.6f} OK")

# 对比actions
if 'actions' in isaac_data and 'actions' in mujoco_data:
    isaac_act = isaac_data['actions']
    mujoco_act = mujoco_data['actions']
    n = min(len(isaac_act), len(mujoco_act))
    print(f"\n对比前5步的actions:")
    for step in range(min(5, n)):
        ia = isaac_act[step]
        ma = mujoco_act[step]
        diff = ia - ma
        max_diff = np.max(np.abs(diff))
        print(f"  Step {step}: max_diff={max_diff:.6f}")
        if max_diff > 0.01:
            print(f"    isaac:  {ia}")
            print(f"    mujoco: {ma}")
            print(f"    diff:   {diff}")
