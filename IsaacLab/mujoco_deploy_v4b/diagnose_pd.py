"""
Diagnose MuJoCo PD control: check if target positions are being tracked.
"""
import mujoco
import numpy as np
import torch
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "v4_robot.yaml")

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
policy_path = config["policy_path"]
simulation_dt = config["simulation_dt"]
control_decimation = config["control_decimation"]
default_angles = np.array(config["default_angles"], dtype=np.float32)
action_scale = config["action_scale"]
num_actions = config["num_actions"]
num_obs = config["num_obs"]
ang_vel_scale = config["ang_vel_scale"]
dof_pos_scale = config["dof_pos_scale"]
dof_vel_scale = config["dof_vel_scale"]
cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
v4_coordinate_remap = config.get("v4_coordinate_remap", False)

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# Joint names
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

# Actuator mapping
actuator_to_joint_indices = []
actuator_names = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    actuator_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    actuator_to_joint_indices.append(mj_joint_names.index(joint_name))
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

print(f"Actuator names: {actuator_names}")
print(f"Actuator -> joint indices: {actuator_to_joint_indices}")
print(f"MuJoCo joints: {mj_joint_names}")

# Print actuator properties
print(f"\nActuator properties:")
for i in range(m.nu):
    name = actuator_names[i]
    gainprm = m.actuator_gainprm[i]
    biasprm = m.actuator_biasprm[i]
    forcerange = m.actuator_forcerange[i]
    ctrlrange = m.actuator_ctrlrange[i]
    print(f"  {name}: gainprm={gainprm[:3]}, biasprm={biasprm[:3]}, forcerange={forcerange}, ctrlrange={ctrlrange}")

# Isaac ordering
# Isaac 13-joint order (must match USD/PhysX traversal order from DOG_V5.usd)
isaac13_joint_order = [
    'LHIPp', 'LHIPy', 'LKNEEp',
    'RHIPp', 'RHIPy', 'RKNEEP',
    'Waist_2',
    'LSDp', 'LSDy', 'LARMp',
    'RSDp', 'RSDy', 'RARMp',
]
# Isaac 12-action order (from flat_env_cfg.py ActionsCfg joint_names list)
isaac12_action_order = [
    'RSDp', 'RSDy', 'RARMp',
    'LSDp', 'LSDy', 'LARMp',
    'RHIPp', 'RHIPy', 'RKNEEP',
    'LHIPp', 'LHIPy', 'LKNEEp',
]
isaac13_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac13_joint_order], dtype=np.int32)
isaac12_action_to_mj = np.array([mj_joint_names.index(j) for j in isaac12_action_order], dtype=np.int32)
waist_mj_idx = mj_joint_names.index('Waist_2')
waist_default = default_angles[waist_mj_idx]

# Load policy
policy = torch.jit.load(policy_path)

# Initialize
d.qpos[2] = 0.3
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles

target_dof_pos = default_angles.copy()
action_raw = np.zeros(num_actions, dtype=np.float32)
obs = np.zeros(num_obs, dtype=np.float32)

# Warmup
for _ in range(1000):
    d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
    mujoco.mj_step(m, d)
d.qvel[:] = 0

print(f"\nAfter warmup: height={d.qpos[2]:.4f}m")
print(f"qpos[7:] = {d.qpos[7:]}")
print(f"default_angles = {default_angles}")
print(f"diff = {d.qpos[7:] - default_angles}")

cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

def v4_remap_ang_vel(ang_vel_body):
    return np.array([ang_vel_body[0], ang_vel_body[2], ang_vel_body[1]])

def v4_remap_gravity(gravity_body):
    return np.array([gravity_body[2], gravity_body[0], gravity_body[1]])

# Run 10 policy steps with detailed diagnostics
counter = 0
for policy_step in range(10):
    # Build obs
    quat = d.qpos[3:7]
    omega = d.qvel[3:6].copy()
    qj_mujoco = d.qpos[7:].copy()
    dqj_mujoco = d.qvel[6:].copy()

    qj_isaac13 = qj_mujoco[isaac13_to_mujoco]
    dqj_isaac13 = dqj_mujoco[isaac13_to_mujoco]
    default_angles_isaac13 = default_angles[isaac13_to_mujoco]

    gravity_orientation = get_gravity_orientation(quat)
    if v4_coordinate_remap:
        omega_obs = v4_remap_ang_vel(omega) * ang_vel_scale
        gravity_obs = v4_remap_gravity(gravity_orientation)
    else:
        omega_obs = omega * ang_vel_scale
        gravity_obs = gravity_orientation

    qj = (qj_isaac13 - default_angles_isaac13) * dof_pos_scale
    dqj = dqj_isaac13 * dof_vel_scale

    obs[0:3] = omega_obs
    obs[3:6] = gravity_obs
    obs[6:9] = cmd * cmd_scale
    obs[9:22] = qj
    obs[22:35] = dqj
    obs[35:47] = action_raw.astype(np.float32)

    # Run policy
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    action_raw = policy(obs_tensor).detach().numpy().squeeze()

    # Apply actions
    target_dof_pos[waist_mj_idx] = waist_default
    for i12 in range(num_actions):
        mj_idx = isaac12_action_to_mj[i12]
        target_dof_pos[mj_idx] = action_raw[i12] * action_scale + default_angles[mj_idx]

    print(f"\n{'='*70}")
    print(f"Policy Step {policy_step}")
    print(f"{'='*70}")
    print(f"  action_raw (12): {action_raw}")
    print(f"  action_raw * scale: {action_raw * action_scale}")
    print(f"\n  Target vs Actual joint positions:")
    for i, jname in enumerate(mj_joint_names):
        actual = d.qpos[7 + i]
        target = target_dof_pos[i]
        error = target - actual
        print(f"    {jname:10s}: target={target:+8.4f}  actual={actual:+8.4f}  error={error:+8.4f}")

    print(f"\n  ctrl being sent:")
    ctrl_values = target_dof_pos[actuator_to_joint_indices]
    for i in range(m.nu):
        print(f"    actuator[{i}] ({actuator_names[i]:10s}): ctrl={ctrl_values[i]:+8.4f}")

    print(f"\n  Actuator forces (qfrc_actuator):")
    # Step once to see forces
    d.ctrl[:] = ctrl_values
    mujoco.mj_step(m, d)
    print(f"    qfrc_actuator[6:] = {d.qfrc_actuator[6:]}")
    print(f"    actuator_force = {d.actuator_force}")

    # Step remaining decimation steps
    for _ in range(control_decimation - 1):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)

    print(f"\n  After {control_decimation} sim steps:")
    print(f"    height: {d.qpos[2]:.4f}m")
    for i, jname in enumerate(mj_joint_names):
        actual = d.qpos[7 + i]
        target = target_dof_pos[i]
        error = target - actual
        if abs(error) > 0.01:
            print(f"    {jname:10s}: target={target:+8.4f}  actual={actual:+8.4f}  error={error:+8.4f} ***")
