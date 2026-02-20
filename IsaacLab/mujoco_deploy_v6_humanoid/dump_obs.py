"""Headless diagnostic: dump first N policy steps' observations (no viewer needed)."""
import mujoco
import numpy as np
import torch
import yaml
import os

def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "v6_robot.yaml")

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = config["xml_path"]
if not os.path.isabs(xml_path):
    xml_path = os.path.join(current_dir, xml_path)

policy_path = config["policy_path"]
simulation_dt = config["simulation_dt"]
control_decimation = config["control_decimation"]
default_angles = np.array(config["default_angles"], dtype=np.float32)
ang_vel_scale = config["ang_vel_scale"]
dof_pos_scale = config["dof_pos_scale"]
dof_vel_scale = config["dof_vel_scale"]
action_scale = config["action_scale"]
cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
num_actions = config["num_actions"]
num_obs = config["num_obs"]
clip_obs = float(config.get("clip_obs", 100.0))
clip_actions = float(config.get("clip_actions", 100.0))
init_height = float(config.get("init_height", 0.55))

isaac_joint_order = config.get("isaac_joint_order", [
    'pelvis_link', 'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
    'LHIPr', 'RHIPr', 'LKNEEp', 'RKNEEp', 'LANKLEp', 'RANKLEp',
    'LANKLEy', 'RANKLEy',
])

# Load MuJoCo
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# Joint names
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

isaac_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac_joint_order], dtype=np.int32)
default_angles_isaac = default_angles[isaac_to_mujoco]

# Actuator mapping
actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    actuator_to_joint_indices.append(mj_joint_names.index(joint_name))
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

# Load policy
policy = torch.jit.load(policy_path)
print(f"Policy loaded: {policy_path}")

# Init pose
d.qpos[2] = init_height
d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
d.qpos[7:] = default_angles

target_dof_pos = default_angles.copy()
action_raw = np.zeros(num_actions, dtype=np.float32)

# Warmup
warmup_steps = int(5.0 / simulation_dt)
print(f"Warmup: {warmup_steps} steps ({warmup_steps * simulation_dt:.1f}s)...")
for _ in range(warmup_steps):
    d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
    mujoco.mj_step(m, d)

print(f"After warmup: height={d.qpos[2]:.4f}m, ncon={d.ncon}")
d.qvel[:] = 0

# Run policy for N steps
cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # forward command
counter = 0
N_STEPS = 20

print(f"\n{'='*80}")
print(f"Running {N_STEPS} policy steps with cmd={cmd}")
print(f"{'='*80}")

for step in range(N_STEPS * control_decimation):
    d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
    mujoco.mj_step(m, d)
    counter += 1

    if counter % control_decimation == 0:
        policy_step = counter // control_decimation - 1

        quat = d.qpos[3:7].copy()
        omega_body = d.qvel[3:6].copy()
        qj_mujoco = d.qpos[7:].copy()
        dqj_mujoco = d.qvel[6:].copy()

        qj_isaac = qj_mujoco[isaac_to_mujoco]
        dqj_isaac = dqj_mujoco[isaac_to_mujoco]

        omega_obs = omega_body * ang_vel_scale
        gravity_obs = get_gravity_orientation(quat)
        cmd_obs = cmd * cmd_scale
        qj_rel = (qj_isaac - default_angles_isaac) * dof_pos_scale
        dqj_obs = dqj_isaac * dof_vel_scale
        last_act = action_raw.copy()

        obs = np.concatenate([omega_obs, gravity_obs, cmd_obs, qj_rel, dqj_obs, last_act])
        obs = np.clip(obs, -clip_obs, clip_obs)

        print(f"\n--- Policy Step {policy_step} ---")
        print(f"  height:  {d.qpos[2]:.4f}m")
        print(f"  quat:    {quat}")
        print(f"  ncon:    {d.ncon}")
        print(f"  obs[ 0: 3] ang_vel*0.2:   {omega_obs}")
        print(f"  obs[ 3: 6] gravity:       {gravity_obs}")
        print(f"  obs[ 6: 9] cmd:           {cmd_obs}")
        print(f"  obs[ 9:22] joint_pos_rel: {qj_rel}")
        print(f"  obs[22:35] joint_vel*0.05:{dqj_obs}")
        print(f"  obs[35:48] last_action:   {last_act}")
        print(f"  FULL OBS: {np.array2string(obs, precision=4, separator=', ', max_line_width=200)}")

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action_raw = policy(obs_tensor).detach().numpy().squeeze()
        action_raw = np.clip(action_raw, -clip_actions, clip_actions)

        print(f"  action_raw: {np.array2string(action_raw, precision=4, separator=', ')}")

        for i_isaac in range(num_actions):
            mj_idx = isaac_to_mujoco[i_isaac]
            target_dof_pos[mj_idx] = action_raw[i_isaac] * action_scale + default_angles[mj_idx]

        print(f"  target_dof (MJ order): {np.array2string(target_dof_pos, precision=4, separator=', ')}")

print(f"\nFinal height: {d.qpos[2]:.4f}m")
print(f"Final ncon: {d.ncon}")
