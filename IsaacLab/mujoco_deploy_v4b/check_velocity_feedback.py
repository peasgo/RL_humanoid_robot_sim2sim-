#!/usr/bin/env python3
"""Check if the velocity feedback is correct.
The policy sees obs[0:3] as remapped linear velocity.
obs[0] = body Z velocity = forward velocity.

If the policy sees LESS forward velocity than actual, it will keep pushing forward.
If the policy sees MORE forward velocity than actual, it will slow down.

Also: check if the cmd_scale is correct. The training might use different cmd scaling.
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=6, suppress=True, linewidth=150)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")
POLICY_PATH = os.path.join(SCRIPT_DIR,
    "../logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/exported/policy.pt")

ISAAC17 = ['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2','LSDp','RSDp',
           'LKNEEp','RKNEEP','LSDy','RSDy','LANKLEp','RANKLEp',
           'LARMp','RARMp','LARMAp','RARMAP']

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_body(v, q):
    return quat_to_rotmat(q).T @ v

def get_gravity_orientation(q):
    w, x, y, z = q
    gx = -2*(x*z - w*y)
    gy = -2*(y*z + w*x)
    gz = -(1 - 2*(x*x + y*y))
    return np.array([gx, gy, gz])

def v4_remap_lin_vel(v):  return np.array([v[2], v[0], v[1]])
def v4_remap_ang_vel(w):  return np.array([w[0], w[2], w[1]])
def v4_remap_gravity(g):  return np.array([g[2], g[0], g[1]])

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)
action_scale = cfg.get("action_scale", 0.25)
control_decimation = cfg.get("control_decimation", 4)

mj_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_names.append(jname)

i17_to_mj = np.array([mj_names.index(n) for n in ISAAC17])
i16_names = [n for n in ISAAC17 if n != 'Waist_2']
i16_to_mj = np.array([mj_names.index(n) for n in i16_names])
waist_mj_idx = mj_names.index('Waist_2')

act_to_joint = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    act_to_joint.append(mj_names.index(joint_name))
act_to_joint = np.array(act_to_joint)

policy = torch.jit.load(POLICY_PATH, map_location="cpu")
policy.eval()

action_filter_alpha = float(cfg.get("action_filter_alpha", 0.0))
obs_filter_alpha = float(cfg.get("obs_filter_alpha", 0.0))
obs_filter_mode = str(cfg.get("obs_filter_mode", "all"))
action_ramp_steps = int(cfg.get("action_ramp_steps", 0))
action_clip = cfg.get("action_clip", None)
if action_clip is not None:
    action_clip = float(action_clip)


# Run simulation and track velocity feedback
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

target_dof_pos = default_angles.copy()
action_isaac16 = np.zeros(16, dtype=np.float32)
action_isaac16_prev = np.zeros(16, dtype=np.float32)
prev_obs = np.zeros(62, dtype=np.float32)
obs = np.zeros(62, dtype=np.float32)
cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # STAND

warmup_steps = int(2.0 / m.opt.timestep)
for _ in range(warmup_steps):
    d.ctrl[:] = target_dof_pos[act_to_joint]
    mujoco.mj_step(m, d)
d.qvel[:] = 0

init_pos = d.qpos[0:3].copy()
counter = 0
policy_step = 0

print("="*120)
print("Velocity feedback analysis (STAND cmd, 100 steps)")
print("="*120)
print(f"{'step':>4s} | {'actual_fwd':>10s} | {'obs_fwd':>10s} | {'diff':>10s} | {'actual_lat':>10s} | {'obs_lat':>10s} | {'grav_y':>8s} | {'act_max':>8s}")
print("-" * 120)

prev_pos = d.qpos[0:3].copy()

while policy_step < 100:
    d.ctrl[:] = target_dof_pos[act_to_joint]
    mujoco.mj_step(m, d)
    counter += 1
    
    if counter % control_decimation == 0:
        quat = d.qpos[3:7]
        base_lin_vel_world = d.qvel[0:3].copy()
        base_ang_vel_body = d.qvel[3:6].copy()
        
        base_lin_vel = world_to_body(base_lin_vel_world, quat)
        omega = world_to_body(base_ang_vel_body, quat)  # double rotation
        
        # Actual forward velocity: body +Z component of world velocity
        # body +Z in world = R @ [0,0,1]
        R = quat_to_rotmat(quat)
        body_z_world = R @ np.array([0, 0, 1])
        actual_fwd_vel = np.dot(base_lin_vel_world, body_z_world)
        actual_lat_vel = np.dot(base_lin_vel_world, R @ np.array([1, 0, 0]))
        
        # Also compute from position change
        dt_policy = m.opt.timestep * control_decimation
        pos_now = d.qpos[0:3].copy()
        pos_vel_world = (pos_now - prev_pos) / dt_policy
        actual_fwd_pos = np.dot(pos_vel_world, body_z_world)
        prev_pos = pos_now.copy()
        
        qj_mujoco = d.qpos[7:].copy()
        dqj_mujoco = d.qvel[6:].copy()
        qj_isaac17 = qj_mujoco[i17_to_mj]
        dqj_isaac17 = dqj_mujoco[i17_to_mj]
        default_angles_isaac17 = default_angles[i17_to_mj]
        gravity_orientation = get_gravity_orientation(quat)
        
        obs[0:3] = v4_remap_lin_vel(base_lin_vel)
        obs[3:6] = v4_remap_ang_vel(omega)
        obs[6:9] = v4_remap_gravity(gravity_orientation)
        obs[9:12] = cmd
        obs[12:29] = qj_isaac17 - default_angles_isaac17
        obs[29:46] = dqj_isaac17
        obs[46:62] = action_isaac16.astype(np.float32)
        
        if obs_filter_alpha > 0 and policy_step > 0:
            if obs_filter_mode == "vel_only":
                obs[0:6] = obs_filter_alpha * prev_obs[0:6] + (1.0 - obs_filter_alpha) * obs[0:6]
                obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1.0 - obs_filter_alpha) * obs[29:46]
            else:
                obs[:] = obs_filter_alpha * prev_obs + (1.0 - obs_filter_alpha) * obs
        prev_obs[:] = obs
        
        # obs[0] = remapped forward velocity = base_lin_vel[2] = body Z velocity
        obs_fwd = obs[0]  # What the policy sees as forward velocity
        obs_lat = obs[1]  # What the policy sees as lateral velocity
        
        grav_y = obs[7]  # Gravity Y component (should be ~0 for upright)
        
        if policy_step < 50 or policy_step % 10 == 0:
            print(f"{policy_step:4d} | {actual_fwd_vel:+10.4f} | {obs_fwd:+10.4f} | {obs_fwd-actual_fwd_vel:+10.4f} | {actual_lat_vel:+10.4f} | {obs_lat:+10.4f} | {grav_y:+8.4f} | {np.max(np.abs(action_isaac16)):8.4f}")
        
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action_isaac16 = policy(obs_tensor).detach().numpy().squeeze()
        
        if action_clip is not None:
            action_isaac16 = np.clip(action_isaac16, -action_clip, action_clip)
        if action_ramp_steps > 0 and policy_step < action_ramp_steps:
            ramp_factor = float(policy_step) / float(action_ramp_steps)
            action_isaac16 = action_isaac16 * ramp_factor
        if action_filter_alpha > 0:
            action_isaac16 = action_filter_alpha * action_isaac16_prev + (1.0 - action_filter_alpha) * action_isaac16
        action_isaac16_prev[:] = action_isaac16
        
        target_dof_pos[waist_mj_idx] = default_angles[waist_mj_idx]
        for i16 in range(16):
            mj_idx = i16_to_mj[i16]
            target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]
        
        policy_step += 1

final_pos = d.qpos[0:3].copy()
disp = final_pos - init_pos
print(f"\nFinal displacement: dx={disp[0]:+.4f} dy={disp[1]:+.4f} dz={disp[2]:+.4f}")
print(f"Forward (-Y): {-disp[1]:+.4f}m")


# Now check: what if we CORRECT the linear velocity to use body frame directly?
# base_lin_vel = world_to_body(qvel[0:3], quat) gives body-frame velocity
# obs[0] = base_lin_vel[2] = body Z velocity
# actual_fwd_vel = dot(qvel[0:3], R @ [0,0,1]) = dot(qvel[0:3], R[:,2]) = (R^T @ qvel[0:3])[2] = base_lin_vel[2]
# So obs[0] SHOULD equal actual_fwd_vel!
# The difference comes from the obs_filter_alpha smoothing.

print("\n" + "="*120)
print("KEY INSIGHT: The obs_filter_alpha=0.3 smooths the velocity observation.")
print("This means the policy sees a DELAYED version of the actual velocity.")
print("When the robot starts moving forward, the policy doesn't see the full velocity")
print("immediately, so it keeps pushing forward.")
print("="*120)

# Test: what happens without obs filter?
print("\n" + "="*120)
print("TEST: Run without obs filter (obs_filter_alpha=0)")
print("="*120)

for test_label, test_obs_alpha, test_act_alpha, test_ramp in [
    ("current (alpha=0.3, ramp=50)", 0.3, 0.3, 50),
    ("no obs filter (alpha=0)", 0.0, 0.3, 50),
    ("no act filter (alpha=0)", 0.3, 0.0, 50),
    ("no filters at all", 0.0, 0.0, 50),
    ("no filters, no ramp", 0.0, 0.0, 0),
]:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target_dof_pos = default_angles.copy()
    action_isaac16 = np.zeros(16, dtype=np.float32)
    action_isaac16_prev = np.zeros(16, dtype=np.float32)
    prev_obs = np.zeros(62, dtype=np.float32)
    obs = np.zeros(62, dtype=np.float32)
    cmd_test = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    warmup_steps = int(2.0 / m.opt.timestep)
    for _ in range(warmup_steps):
        d.ctrl[:] = target_dof_pos[act_to_joint]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    
    init_pos = d.qpos[0:3].copy()
    counter = 0
    policy_step = 0
    
    while policy_step < 300:
        d.ctrl[:] = target_dof_pos[act_to_joint]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_decimation == 0:
            quat = d.qpos[3:7]
            base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
            omega = world_to_body(d.qvel[3:6].copy(), quat)
            
            qj_mujoco = d.qpos[7:].copy()
            dqj_mujoco = d.qvel[6:].copy()
            qj_isaac17 = qj_mujoco[i17_to_mj]
            dqj_isaac17 = dqj_mujoco[i17_to_mj]
            default_angles_isaac17 = default_angles[i17_to_mj]
            gravity_orientation = get_gravity_orientation(quat)
            
            obs[0:3] = v4_remap_lin_vel(base_lin_vel)
            obs[3:6] = v4_remap_ang_vel(omega)
            obs[6:9] = v4_remap_gravity(gravity_orientation)
            obs[9:12] = cmd_test
            obs[12:29] = qj_isaac17 - default_angles_isaac17
            obs[29:46] = dqj_isaac17
            obs[46:62] = action_isaac16.astype(np.float32)
            
            if test_obs_alpha > 0 and policy_step > 0:
                obs[0:6] = test_obs_alpha * prev_obs[0:6] + (1.0 - test_obs_alpha) * obs[0:6]
                obs[29:46] = test_obs_alpha * prev_obs[29:46] + (1.0 - test_obs_alpha) * obs[29:46]
            prev_obs[:] = obs
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_isaac16 = policy(obs_tensor).detach().numpy().squeeze()
            
            if action_clip is not None:
                action_isaac16 = np.clip(action_isaac16, -action_clip, action_clip)
            if test_ramp > 0 and policy_step < test_ramp:
                ramp_factor = float(policy_step) / float(test_ramp)
                action_isaac16 = action_isaac16 * ramp_factor
            if test_act_alpha > 0:
                action_isaac16 = test_act_alpha * action_isaac16_prev + (1.0 - test_act_alpha) * action_isaac16
            action_isaac16_prev[:] = action_isaac16
            
            target_dof_pos[waist_mj_idx] = default_angles[waist_mj_idx]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]
            
            policy_step += 1
    
    final_pos = d.qpos[0:3].copy()
    disp = final_pos - init_pos
    fwd = -disp[1]
    print(f"  {test_label:40s}: fwd={fwd:+.4f}m  lat={disp[0]:+.4f}m")
