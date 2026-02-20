"""Quick test: feed known obs to exported policy.pt and check output."""
import torch
import numpy as np
import sys

policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-18_12-58-46/exported/policy.pt"

print(f"Loading policy: {policy_path}")
policy = torch.jit.load(policy_path)
policy.eval()

# Print model structure
print(f"\nModel structure:")
for name, param in policy.named_parameters():
    print(f"  {name}: {param.shape}")

for name, buf in policy.named_buffers():
    print(f"  [buffer] {name}: {buf.shape}  mean={buf.mean().item():.4f}  std={buf.std().item():.4f}")
    if buf.numel() <= 48:
        print(f"           values: {buf.flatten().numpy()}")

# Test 1: zero obs
print(f"\n{'='*60}")
print("Test 1: Zero obs (48 dims, no history)")
obs_zero = torch.zeros(1, 48)
with torch.no_grad():
    act_zero = policy(obs_zero).numpy().squeeze()
print(f"  action: {act_zero}")
print(f"  act_max: {np.max(np.abs(act_zero)):.4f}")

# Test 2: obs mimicking initial state (single frame, no history)
# ang_vel=[0,0,0]*0.2, gravity=[0,0,-1], cmd=[0,0,0], jpos_rel=[0]*13, jvel=[0]*13, act=[0]*13
print(f"\n{'='*60}")
print("Test 2: Initial standing obs (gravity=[0,0,-1], rest zero)")
frame = np.zeros(48, dtype=np.float32)
frame[3:6] = [0.0, 0.0, -1.0]  # projected gravity

# Single frame obs (no history stacking)
obs_init = frame.copy()
print(f"  obs shape: {obs_init.shape}")
print(f"  obs[0:3] (ang_vel): {obs_init[0:3]}")
print(f"  obs[3:6] (gravity): {obs_init[3:6]}")
print(f"  obs[6:9] (cmd):     {obs_init[6:9]}")

obs_t = torch.from_numpy(obs_init).unsqueeze(0)
with torch.no_grad():
    act_init = policy(obs_t).numpy().squeeze()
print(f"  action: {act_init}")
print(f"  act_max: {np.max(np.abs(act_init)):.4f}")

print(f"\n{'='*60}")
print("Done.")
