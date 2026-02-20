"""Quick test: load exported policy and run with zero obs to verify output.
Run WITHOUT IsaacLab: python test_exported_policy.py
"""
import torch
import numpy as np
import sys

policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-19_20-05-08/exported/policy.pt"

print(f"Loading: {policy_path}")
policy = torch.jit.load(policy_path, map_location="cpu")
policy.eval()

num_obs = 48
num_actions = 13

# Test 1: zero observation
obs_zero = torch.zeros(1, num_obs)
with torch.no_grad():
    action_zero = policy(obs_zero).numpy().squeeze()
print(f"\n=== Test 1: Zero observation ===")
print(f"obs shape: {obs_zero.shape}")
print(f"action shape: {action_zero.shape}")
print(f"action: {action_zero}")
print(f"action range: [{action_zero.min():.4f}, {action_zero.max():.4f}]")
print(f"action abs mean: {np.abs(action_zero).mean():.4f}")

# Test 2: "standing still" observation (gravity=[0,0,-1], everything else zero)
obs_stand = torch.zeros(1, num_obs)
obs_stand[0, 3] = 0.0   # gx
obs_stand[0, 4] = 0.0   # gy
obs_stand[0, 5] = -1.0  # gz (standing upright)
with torch.no_grad():
    action_stand = policy(obs_stand).numpy().squeeze()
print(f"\n=== Test 2: Standing observation (gravity=[0,0,-1]) ===")
print(f"action: {action_stand}")
print(f"action range: [{action_stand.min():.4f}, {action_stand.max():.4f}]")
print(f"action abs mean: {np.abs(action_stand).mean():.4f}")

# Test 3: Check if actions are reasonable (should be small for standing)
print(f"\n=== Sanity check ===")
if np.abs(action_stand).max() > 5.0:
    print("WARNING: Actions are very large for standing pose! Policy might be broken.")
elif np.abs(action_stand).max() > 2.0:
    print("CAUTION: Actions are moderately large for standing pose.")
else:
    print("OK: Actions are reasonable for standing pose.")

# Test 4: Check policy input/output dimensions
print(f"\nExpected: input={num_obs}, output={num_actions}")
print(f"Actual:   input={obs_zero.shape[1]}, output={action_zero.shape[0]}")
if action_zero.shape[0] != num_actions:
    print(f"ERROR: Output dimension mismatch! Expected {num_actions}, got {action_zero.shape[0]}")

# Test 5: Scaled action -> joint target
action_scale = 0.25
default_angles_bfs = np.array([
    0.0,                          # pelvis_link
    -0.2, -0.2,                   # LHIPp, RHIPp
    0.0, 0.0,                     # LHIPy, RHIPy
    0.0, 0.0,                     # LHIPr, RHIPr
    -0.4, 0.4,                    # LKNEEp, RKNEEp
    0.2, -0.2,                    # LANKLEp, RANKLEp
    0.0, 0.0,                     # LANKLEy, RANKLEy
])
bfs_names = [
    'pelvis_link',
    'LHIPp', 'RHIPp',
    'LHIPy', 'RHIPy',
    'LHIPr', 'RHIPr',
    'LKNEEp', 'RKNEEp',
    'LANKLEp', 'RANKLEp',
    'LANKLEy', 'RANKLEy',
]

targets = action_stand * action_scale + default_angles_bfs
print(f"\n=== Joint targets from standing obs (action*0.25 + default) ===")
for i, name in enumerate(bfs_names):
    print(f"  [{i:2d}] {name:20s}  action={action_stand[i]:+.4f}  target={targets[i]:+.4f}  default={default_angles_bfs[i]:+.4f}")
