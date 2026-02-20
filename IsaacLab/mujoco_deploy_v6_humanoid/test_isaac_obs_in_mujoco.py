"""
Test: Feed exact IsaacLab observations into the policy and compare actions.
Then apply actions to MuJoCo and compare resulting states.
"""
import numpy as np
import torch
import mujoco
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load config
with open(os.path.join(current_dir, "v6_robot.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

policy_path = config["policy_path"]
xml_path = os.path.join(current_dir, config["xml_path"])
action_scale = config["action_scale"]
num_actions = config["num_actions"]

# Load policy
policy = torch.jit.load(policy_path)
print(f"Policy loaded: {policy_path}")

# Load MuJoCo model
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = config["simulation_dt"]

# Isaac joint orders from config
isaac_joint_order = config["isaac_joint_order"]
action_joint_order = config["action_joint_order"]

# MuJoCo joint names
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

isaac_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac_joint_order], dtype=np.int32)

default_angles = np.array(config["default_angles"], dtype=np.float32)
default_angles_isaac = default_angles[isaac_to_mujoco].copy()

# Actuator mapping
actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    pd_index = mj_joint_names.index(joint_name)
    actuator_to_joint_indices.append(pd_index)
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

# ================================================================
# IsaacLab Step 0 observation (from user's output)
# ================================================================
isaac_obs_step0 = np.array([
    -9.8639e-03,  2.9603e-02, -3.1111e-02,  4.1504e-02,  2.4069e-02, -1.0409e+00,
     0.0000e+00, -0.0000e+00,  0.0000e+00,
    -5.9646e-02, -5.7211e-02,  4.2106e-02, -7.5158e-02, -1.1511e-01, -7.1091e-02,
    -8.1525e-02, -1.4265e-01, -8.5768e-02,  7.0197e-02,  1.3073e-01, -9.5675e-02,
    -1.3902e-02, -4.1743e-02, -6.5665e-02, -2.9312e-03,  5.4938e-02, -6.7670e-02,
    -1.5498e-02, -3.2053e-02, -3.3386e-04,  4.9293e-02, -3.3653e-02,  2.9465e-03,
    -3.6386e-02, -1.0833e-02,
     0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
     0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
     0.0000e+00
], dtype=np.float32)

# IsaacLab Step 0 expected actions
isaac_actions_step0 = np.array([
    -0.030991,  0.331074, -0.490743,  0.285347,  0.051951,
    -0.121331, -0.109601, -0.636742,  0.114688, -0.328023,
    -0.851936, -0.966040,  0.682626
], dtype=np.float32)

# IsaacLab Step 1 observation
isaac_obs_step1 = np.array([
    -0.0249,  0.1405, -0.5294, -0.0188,  0.0387, -1.0497,
     0.0000, -0.0000,  0.0000,
    -0.0438, -0.0435,  0.0053, -0.0576, -0.0920, -0.0718,
    -0.0708, -0.1467, -0.0751,  0.0455,  0.0816, -0.1083,
     0.0174,  0.0125,  0.0735, -0.0127,  0.0429,  0.0023,
     0.0633, -0.0085,  0.0027,  0.0775, -0.1200, -0.2087,
    -0.0847,  0.1038,
    -0.0310,  0.3311, -0.4907,  0.2853,  0.0520,
    -0.1213, -0.1096, -0.6367,  0.1147, -0.3280,
    -0.8519, -0.9660,  0.6826
], dtype=np.float32)

print("=" * 70)
print("TEST 1: Feed IsaacLab Step 0 obs into policy")
print("=" * 70)

obs_tensor = torch.from_numpy(isaac_obs_step0).unsqueeze(0)
action_out = policy(obs_tensor).detach().numpy().squeeze()
action_out = np.clip(action_out, -100.0, 100.0)

print(f"\nPolicy output actions:")
for i, jname in enumerate(isaac_joint_order):
    diff = action_out[i] - isaac_actions_step0[i]
    match = "OK" if abs(diff) < 0.01 else "MISMATCH"
    print(f"  [{i:2d}] {jname:14s}  mujoco={action_out[i]:+.6f}  isaac={isaac_actions_step0[i]:+.6f}  diff={diff:+.6f}  {match}")

max_diff = np.max(np.abs(action_out - isaac_actions_step0))
print(f"\nMax action diff: {max_diff:.6f}")

print("\n" + "=" * 70)
print("TEST 2: Feed IsaacLab Step 1 obs into policy")
print("=" * 70)

obs_tensor1 = torch.from_numpy(isaac_obs_step1).unsqueeze(0)
action_out1 = policy(obs_tensor1).detach().numpy().squeeze()
action_out1 = np.clip(action_out1, -100.0, 100.0)

print(f"\nPolicy output actions (step 1):")
for i, jname in enumerate(isaac_joint_order):
    print(f"  [{i:2d}] {jname:14s}  action={action_out1[i]:+.6f}")

print("\n" + "=" * 70)
print("TEST 3: Apply Step 0 actions to MuJoCo, compare with IsaacLab Step 1 state")
print("=" * 70)

# Set MuJoCo to match IsaacLab Step 0 joint positions
# From Step 0 obs: joint_pos_rel = obs[9:22], so joint_pos = joint_pos_rel + default
isaac_jpos_rel_step0 = isaac_obs_step0[9:22]
isaac_jpos_step0 = isaac_jpos_rel_step0 + default_angles_isaac

print(f"\nIsaacLab Step 0 joint positions (Isaac order):")
for i, jname in enumerate(isaac_joint_order):
    print(f"  [{i:2d}] {jname:14s}  pos={isaac_jpos_step0[i]:+.6f}  (rel={isaac_jpos_rel_step0[i]:+.6f}  def={default_angles_isaac[i]:+.4f})")

# Set MuJoCo initial state
d.qpos[0:3] = [0, 0, 0.55]
d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
# Map Isaac joint positions back to MuJoCo order
for i_isaac in range(num_actions):
    mj_idx = isaac_to_mujoco[i_isaac]
    d.qpos[7 + mj_idx] = isaac_jpos_step0[i_isaac]
d.qvel[:] = 0

# Compute PD targets from Step 0 actions
target_dof_pos = np.array(default_angles, dtype=np.float32)
for i_isaac in range(num_actions):
    mj_idx = isaac_to_mujoco[i_isaac]
    target_dof_pos[mj_idx] = isaac_actions_step0[i_isaac] * action_scale + default_angles[mj_idx]

print(f"\nPD targets (MuJoCo order):")
for i, jname in enumerate(mj_joint_names):
    print(f"  [{i:2d}] {jname:14s}  target={target_dof_pos[i]:+.6f}  default={default_angles[i]:+.4f}")

# Step MuJoCo for 4 substeps (decimation=4)
decimation = config["control_decimation"]
for _ in range(decimation):
    d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
    mujoco.mj_step(m, d)

# Read resulting state in Isaac order
mj_qj_after = d.qpos[7:].copy()
mj_dqj_after = d.qvel[6:].copy()
mj_qj_isaac = mj_qj_after[isaac_to_mujoco]
mj_dqj_isaac = mj_dqj_after[isaac_to_mujoco]

# IsaacLab Step 1 joint positions (from obs)
isaac_jpos_rel_step1 = isaac_obs_step1[9:22]
isaac_jpos_step1 = isaac_jpos_rel_step1 + default_angles_isaac

# IsaacLab Step 1 joint velocities (from obs, need to un-scale by dof_vel_scale=0.05)
isaac_jvel_scaled_step1 = isaac_obs_step1[22:35]
isaac_jvel_step1 = isaac_jvel_scaled_step1 / config["dof_vel_scale"]

print(f"\n--- After {decimation} MuJoCo substeps ---")
print(f"\nJoint positions comparison (Isaac order):")
print(f"  {'Joint':14s}  {'MuJoCo':>10s}  {'IsaacLab':>10s}  {'Diff':>10s}")
for i, jname in enumerate(isaac_joint_order):
    diff = mj_qj_isaac[i] - isaac_jpos_step1[i]
    print(f"  {jname:14s}  {mj_qj_isaac[i]:+.6f}  {isaac_jpos_step1[i]:+.6f}  {diff:+.6f}")

print(f"\nJoint velocities comparison (Isaac order):")
print(f"  {'Joint':14s}  {'MuJoCo':>10s}  {'IsaacLab':>10s}  {'Diff':>10s}")
for i, jname in enumerate(isaac_joint_order):
    diff = mj_dqj_isaac[i] - isaac_jvel_step1[i]
    print(f"  {jname:14s}  {mj_dqj_isaac[i]:+.6f}  {isaac_jvel_step1[i]:+.6f}  {diff:+.6f}")

# Base state comparison
mj_quat = d.qpos[3:7].copy()
mj_omega = d.qvel[3:6].copy()
isaac_omega_scaled = isaac_obs_step1[0:3]
isaac_omega = isaac_omega_scaled / config["ang_vel_scale"]
isaac_gravity = isaac_obs_step1[3:6]

def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

mj_gravity = get_gravity_orientation(mj_quat)

print(f"\nBase state comparison:")
print(f"  ang_vel (body):  MuJoCo={mj_omega}  IsaacLab={isaac_omega}")
print(f"  gravity:         MuJoCo={mj_gravity}  IsaacLab={isaac_gravity}")
print(f"  height:          MuJoCo={d.qpos[2]:.4f}")
print(f"  quat:            MuJoCo={mj_quat}")

pos_diffs = np.abs(mj_qj_isaac - isaac_jpos_step1)
vel_diffs = np.abs(mj_dqj_isaac - isaac_jvel_step1)
print(f"\nSummary:")
print(f"  Max joint pos diff:  {np.max(pos_diffs):.6f} rad")
print(f"  Mean joint pos diff: {np.mean(pos_diffs):.6f} rad")
print(f"  Max joint vel diff:  {np.max(vel_diffs):.6f} rad/s")
print(f"  Mean joint vel diff: {np.mean(vel_diffs):.6f} rad/s")
print(f"  Gravity diff:        {np.max(np.abs(mj_gravity - isaac_gravity)):.6f}")
