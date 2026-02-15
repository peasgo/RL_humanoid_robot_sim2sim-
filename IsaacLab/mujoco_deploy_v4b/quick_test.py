"""Quick long-run test after joint order fix."""
import mujoco
import numpy as np
import torch
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "v4_robot.yaml"), "r") as f:
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

mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    actuator_to_joint_indices.append(mj_joint_names.index(joint_name))
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

# FIXED: Isaac 13-joint order matching USD/PhysX traversal
isaac13_joint_order = [
    'LHIPp', 'LHIPy', 'LKNEEp',
    'RHIPp', 'RHIPy', 'RKNEEP',
    'Waist_2',
    'LSDp', 'LSDy', 'LARMp',
    'RSDp', 'RSDy', 'RARMp',
]
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

print(f"isaac13_to_mujoco: {isaac13_to_mujoco}")

policy = torch.jit.load(policy_path)

d.qpos[2] = 0.3
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles

target_dof_pos = default_angles.copy()
action_raw = np.zeros(num_actions, dtype=np.float32)
obs = np.zeros(num_obs, dtype=np.float32)

# Joint limits
joint_limits = {}
for i, jname in enumerate(mj_joint_names):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if m.jnt_limited[jid]:
        joint_limits[jname] = (float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1]))

# Warmup
for _ in range(1000):
    d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
    mujoco.mj_step(m, d)
d.qvel[:] = 0

print(f"After warmup: height={d.qpos[2]:.4f}m, x={d.qpos[0]:.4f}")

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

counter = 0
for policy_step in range(500):
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

    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    action_raw = policy(obs_tensor).detach().numpy().squeeze()

    target_dof_pos[waist_mj_idx] = waist_default
    for i12 in range(num_actions):
        mj_idx = isaac12_action_to_mj[i12]
        target_dof_pos[mj_idx] = action_raw[i12] * action_scale + default_angles[mj_idx]

    for i, jname in enumerate(mj_joint_names):
        if jname in joint_limits:
            low, high = joint_limits[jname]
            target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)

    for _ in range(control_decimation):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)

    if policy_step % 25 == 0:
        pos = d.qpos[0:3]
        print(f"Step {policy_step:4d} (t={policy_step*control_decimation*simulation_dt:.2f}s): "
              f"x={pos[0]:+.4f} y={pos[1]:+.4f} h={pos[2]:.4f} "
              f"act_max={np.max(np.abs(action_raw)):.3f}")

print(f"\nFinal: x={d.qpos[0]:+.4f} y={d.qpos[1]:+.4f} h={d.qpos[2]:.4f}")
print(f"Total x displacement: {d.qpos[0]:.4f}m in {500*control_decimation*simulation_dt:.1f}s")
