"""诊断 policy 第一步：打印 obs 和 action，检查是否合理"""
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
with open(os.path.join(current_dir, "v6_robot.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = config["simulation_dt"]

default_angles = np.array(config["default_angles"], dtype=np.float32)
isaac_joint_order = config["isaac_joint_order"]
action_scale = config["action_scale"]
ang_vel_scale = config["ang_vel_scale"]
dof_vel_scale = config["dof_vel_scale"]

mj_joint_names = []
for jid in range(m.njnt):
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))

isaac_to_mujoco = np.array([mj_joint_names.index(n) for n in isaac_joint_order])
default_angles_isaac = default_angles[isaac_to_mujoco]

# Load policy
policy_path = config["policy_path"]
if not os.path.exists(policy_path):
    print(f"Policy not found: {policy_path}")
    exit(1)
policy = torch.jit.load(policy_path)
print(f"Policy loaded: {policy_path}")

# Setup initial state
d.qpos[2] = 0.55
d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
d.qpos[7:] = default_angles

# Actuator mapping
actuator_to_joint = []
for i in range(m.nu):
    jid = m.actuator_trnid[i, 0]
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    actuator_to_joint.append(mj_joint_names.index(jn))
actuator_to_joint = np.array(actuator_to_joint)

# Warmup
target = default_angles.copy()
for _ in range(int(2.0 / m.opt.timestep)):
    d.ctrl[:] = target[actuator_to_joint]
    mujoco.mj_step(m, d)

d.qvel[:] = 0
print(f"\nAfter warmup: h={d.qpos[2]:.4f}m  quat={d.qpos[3:7]}")

# Run policy for a few steps
action_raw = np.zeros(13, dtype=np.float32)
decimation = config["control_decimation"]

for policy_step in range(5):
    # Step simulation for decimation steps
    for _ in range(decimation):
        d.ctrl[:] = target[actuator_to_joint]
        mujoco.mj_step(m, d)
    
    # Build observation
    quat = d.qpos[3:7].copy()
    omega_body = d.qvel[3:6].copy()
    qj_mujoco = d.qpos[7:].copy()
    dqj_mujoco = d.qvel[6:].copy()
    
    qj_isaac = qj_mujoco[isaac_to_mujoco]
    dqj_isaac = dqj_mujoco[isaac_to_mujoco]
    
    omega_obs = omega_body * ang_vel_scale
    gravity_obs = get_gravity_orientation(quat)
    cmd_obs = np.zeros(3, dtype=np.float32)
    qj_rel = qj_isaac - default_angles_isaac
    dqj_obs = dqj_isaac * dof_vel_scale
    last_act = action_raw.copy()
    
    obs = np.concatenate([omega_obs, gravity_obs, cmd_obs, qj_rel, dqj_obs, last_act]).astype(np.float32)
    obs = np.clip(obs, -100, 100)
    
    # Run policy
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    action_raw = policy(obs_tensor).detach().numpy().squeeze()
    action_raw = np.clip(action_raw, -100, 100)
    
    # Apply actions
    for i_isaac in range(13):
        mj_idx = isaac_to_mujoco[i_isaac]
        target[mj_idx] = action_raw[i_isaac] * action_scale + default_angles[mj_idx]
    
    print(f"\n{'='*70}")
    print(f"Policy step {policy_step}: h={d.qpos[2]:.4f}m  ncon={d.ncon}")
    print(f"  obs[0:3]  ang_vel*0.2:  {omega_obs}")
    print(f"  obs[3:6]  gravity:      {gravity_obs}")
    print(f"  obs[6:9]  cmd:          {cmd_obs}")
    print(f"  obs[9:22] joint_pos_rel:")
    for i, jn in enumerate(isaac_joint_order):
        print(f"    [{i:2d}] {jn:14s}  pos={qj_isaac[i]:+.4f}  def={default_angles_isaac[i]:+.4f}  rel={qj_rel[i]:+.4f}")
    print(f"  obs[22:35] joint_vel*0.05: {dqj_obs}")
    print(f"  obs[35:48] last_action:    {last_act}")
    print(f"  action_raw: {action_raw}")
    print(f"  target_dof_pos (MJ order):")
    for i, jn in enumerate(mj_joint_names):
        print(f"    [{i:2d}] {jn:14s}  target={target[i]:+.4f}  default={default_angles[i]:+.4f}  delta={target[i]-default_angles[i]:+.4f}")
    
    # Check if action magnitudes are reasonable
    max_act = np.max(np.abs(action_raw))
    max_target_delta = np.max(np.abs(target - default_angles))
    print(f"  |action|_max={max_act:.4f}  |target-default|_max={max_target_delta:.4f} rad ({np.degrees(max_target_delta):.1f}°)")
