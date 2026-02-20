"""
Replay exact IsaacLab Step 0 & Step 1 in MuJoCo with viewer.
Uses the exact obs from IsaacLab -> feeds to policy -> applies actions to MuJoCo.
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "v6_robot.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
policy_path = config["policy_path"]
action_scale = config["action_scale"]
num_actions = config["num_actions"]
simulation_dt = config["simulation_dt"]
decimation = config["control_decimation"]
default_angles = np.array(config["default_angles"], dtype=np.float32)
isaac_joint_order = config["isaac_joint_order"]
clip_actions = config["clip_actions"]

# Load policy
policy = torch.jit.load(policy_path)
print(f"Policy loaded: {policy_path}")

# Load MuJoCo
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# Joint mappings
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

isaac_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac_joint_order], dtype=np.int32)

actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    pd_index = mj_joint_names.index(joint_name)
    actuator_to_joint_indices.append(pd_index)
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

# ================================================================
# IsaacLab exact data
# ================================================================

# Step 0 obs (48 dims)
obs_step0 = np.array([
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

# Step 1 obs (48 dims)
obs_step1 = np.array([
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

# Step 0 joint positions (Isaac order, from obs)
default_angles_isaac = default_angles[isaac_to_mujoco].copy()
jpos_step0 = obs_step0[9:22] + default_angles_isaac
jpos_step1 = obs_step1[9:22] + default_angles_isaac

# Step 0 joint velocities (from obs, un-scale)
jvel_step0 = obs_step0[22:35] / 0.05  # dof_vel_scale
jvel_step1 = obs_step1[22:35] / 0.05

# ================================================================
# Run policy on both obs
# ================================================================
action_step0 = policy(torch.from_numpy(obs_step0).unsqueeze(0)).detach().numpy().squeeze()
action_step0 = np.clip(action_step0, -clip_actions, clip_actions)

action_step1 = policy(torch.from_numpy(obs_step1).unsqueeze(0)).detach().numpy().squeeze()
action_step1 = np.clip(action_step1, -clip_actions, clip_actions)

print(f"\nStep 0 actions: {action_step0}")
print(f"Step 1 actions: {action_step1}")

def set_state_from_isaac(d, jpos_isaac, jvel_isaac, isaac_to_mujoco, default_angles):
    """Set MuJoCo state from Isaac joint positions/velocities."""
    d.qpos[0:3] = [0, 0, 0.55]
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    for i_isaac in range(len(jpos_isaac)):
        mj_idx = isaac_to_mujoco[i_isaac]
        d.qpos[7 + mj_idx] = jpos_isaac[i_isaac]
        d.qvel[6 + mj_idx] = jvel_isaac[i_isaac]
    d.qvel[0:6] = 0  # base vel

def compute_targets(action_isaac, action_scale, default_angles, isaac_to_mujoco):
    """Compute PD targets from Isaac actions."""
    target = np.array(default_angles, dtype=np.float32)
    for i_isaac in range(len(action_isaac)):
        mj_idx = isaac_to_mujoco[i_isaac]
        target[mj_idx] = action_isaac[i_isaac] * action_scale + default_angles[mj_idx]
    return target

print("\n" + "=" * 60)
print("Launching MuJoCo viewer - replaying IsaacLab steps")
print("=" * 60)

with mujoco.viewer.launch_passive(m, d) as viewer:
    
    # ---- Step 0: Set initial state, apply action ----
    print("\n--- Setting Step 0 state ---")
    set_state_from_isaac(d, jpos_step0, jvel_step0, isaac_to_mujoco, default_angles)
    mujoco.mj_forward(m, d)
    viewer.sync()
    time.sleep(2.0)  # Let user see initial pose
    
    target0 = compute_targets(action_step0, action_scale, default_angles, isaac_to_mujoco)
    print(f"Applying Step 0 actions, stepping {decimation} substeps...")
    
    for sub in range(decimation):
        d.ctrl[:] = target0[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
    viewer.sync()
    
    # Print resulting state
    mj_qj = d.qpos[7:].copy()
    mj_dqj = d.qvel[6:].copy()
    mj_qj_isaac = mj_qj[isaac_to_mujoco]
    mj_dqj_isaac = mj_dqj[isaac_to_mujoco]
    
    print(f"\nAfter Step 0 -> MuJoCo vs IsaacLab Step 1:")
    print(f"  {'Joint':14s}  {'MJ pos':>10s}  {'IL pos':>10s}  {'diff':>8s}  |  {'MJ vel':>10s}  {'IL vel':>10s}  {'diff':>8s}")
    for i, jname in enumerate(isaac_joint_order):
        pdiff = mj_qj_isaac[i] - jpos_step1[i]
        vdiff = mj_dqj_isaac[i] - jvel_step1[i]
        print(f"  {jname:14s}  {mj_qj_isaac[i]:+.6f}  {jpos_step1[i]:+.6f}  {pdiff:+.4f}  |  "
              f"{mj_dqj_isaac[i]:+.4f}  {jvel_step1[i]:+.4f}  {vdiff:+.4f}")
    
    time.sleep(2.0)
    
    # ---- Step 1: Apply action from Step 1 obs ----
    print(f"\n--- Applying Step 1 actions, stepping {decimation} substeps ---")
    target1 = compute_targets(action_step1, action_scale, default_angles, isaac_to_mujoco)
    
    for sub in range(decimation):
        d.ctrl[:] = target1[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
    viewer.sync()
    
    mj_qj2 = d.qpos[7:].copy()[isaac_to_mujoco]
    print(f"\nAfter Step 1 joint positions (Isaac order):")
    for i, jname in enumerate(isaac_joint_order):
        print(f"  [{i:2d}] {jname:14s}  pos={mj_qj2[i]:+.6f}")
    
    time.sleep(2.0)
    
    # ---- Continue running with Step 1 targets ----
    print(f"\n--- Continuing with Step 1 targets. Close viewer to exit ---")
    while viewer.is_running():
        step_start = time.time()
        d.ctrl[:] = target1[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        viewer.sync()
        elapsed = time.time() - step_start
        sleep_time = simulation_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

print("Done.")
