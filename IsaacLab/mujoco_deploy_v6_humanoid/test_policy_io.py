"""Quick test: feed known obs to exported policy.pt and check output."""
import torch
import numpy as np
import sys

policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-19_11-55-18/exported/policy.pt"

print(f"Loading policy: {policy_path}")
policy = torch.jit.load(policy_path)
policy.eval()

# Print model structure
print(f"\nModel structure:")
for name, param in policy.named_parameters():
    print(f"  {name}: {param.shape}")

for name, buf in policy.named_buffers():
    print(f"  [buffer] {name}: {buf.shape}  mean={buf.mean().item():.4f}  std={buf.std().item():.4f}")
    if buf.numel() <= 240:
        print(f"           values: {buf.flatten().numpy()}")

# Test 1: zero obs
print(f"\n{'='*60}")
print("Test 1: Zero obs (240 dims)")
obs_zero = torch.zeros(1, 240)
with torch.no_grad():
    act_zero = policy(obs_zero).numpy().squeeze()
print(f"  action: {act_zero}")
print(f"  act_max: {np.max(np.abs(act_zero)):.4f}")

# Test 2: obs mimicking initial state (all history slots same)
# ang_vel=[0,0,0]*0.2, gravity=[0,0,-1], cmd=[0,0,0], jpos_rel=[0]*13, jvel=[0]*13, act=[0]*13
print(f"\n{'='*60}")
print("Test 2: Initial standing obs (gravity=[0,0,-1], rest zero)")
frame = np.zeros(48, dtype=np.float32)
frame[3:6] = [0.0, 0.0, -1.0]  # projected gravity

# Build per-term history (5 copies each)
ang_vel_hist = np.tile(frame[0:3], 5)    # 15
gravity_hist = np.tile(frame[3:6], 5)    # 15
cmd_hist = np.tile(frame[6:9], 5)        # 15
jpos_hist = np.tile(frame[9:22], 5)      # 65
jvel_hist = np.tile(frame[22:35], 5)     # 65
act_hist = np.tile(frame[35:48], 5)      # 65

obs_init = np.concatenate([ang_vel_hist, gravity_hist, cmd_hist, jpos_hist, jvel_hist, act_hist])
print(f"  obs shape: {obs_init.shape}")
print(f"  obs[0:15] (ang_vel_hist): {obs_init[0:15]}")
print(f"  obs[15:30] (gravity_hist): {obs_init[15:30]}")

obs_t = torch.from_numpy(obs_init).unsqueeze(0)
with torch.no_grad():
    act_init = policy(obs_t).numpy().squeeze()
print(f"  action: {act_init}")
print(f"  act_max: {np.max(np.abs(act_init)):.4f}")

# Test 3: same obs but with frame-interleaved layout (WRONG layout for comparison)
print(f"\n{'='*60}")
print("Test 3: Frame-interleaved layout (for comparison - may be wrong)")
obs_interleaved = np.tile(frame, 5)  # [frame0, frame1, frame2, frame3, frame4]
print(f"  obs shape: {obs_interleaved.shape}")
obs_t2 = torch.from_numpy(obs_interleaved).unsqueeze(0)
with torch.no_grad():
    act_interleaved = policy(obs_t2).numpy().squeeze()
print(f"  action: {act_interleaved}")
print(f"  act_max: {np.max(np.abs(act_interleaved)):.4f}")

print(f"\n{'='*60}")
print("Done.")
