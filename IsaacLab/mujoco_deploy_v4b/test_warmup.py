#!/usr/bin/env python3
"""Test: does the warmup cause the forward bias?
Compare: with warmup vs without warmup vs shorter warmup.
Also test: what if we reset qvel after warmup?
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


def run_sim(cmd_values, warmup_time=2.0, num_steps=300):
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
    cmd = np.array(cmd_values, dtype=np.float32)
    
    if warmup_time > 0:
        warmup_steps = int(warmup_time / m.opt.timestep)
        for _ in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[act_to_joint]
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
    
    # Record state after warmup
    quat_after_warmup = d.qpos[3:7].copy()
    grav_after_warmup = get_gravity_orientation(quat_after_warmup)
    grav_remapped = v4_remap_gravity(grav_after_warmup)
    qj_after_warmup = d.qpos[7:].copy()
    qj_isaac17 = qj_after_warmup[i17_to_mj]
    default_isaac17 = default_angles[i17_to_mj]
    qj_rel = qj_isaac17 - default_isaac17
    
    init_pos = d.qpos[0:3].copy()
    counter = 0
    policy_step = 0
    
    while policy_step < num_steps:
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
    return disp, grav_remapped, qj_rel


print("="*100)
print("TEST 1: Effect of warmup time on forward bias (STAND cmd)")
print("="*100)

warmup_times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
print(f"\n  {'Warmup':>8s} | {'fwd(-Y)':>8s} | {'lat(X)':>8s} | {'grav_remapped':>30s} | {'max|qj_rel|':>12s}")
print("  " + "-" * 80)

for wt in warmup_times:
    disp, grav, qj_rel = run_sim([0, 0, 0], warmup_time=wt)
    fwd = -disp[1]
    lat = disp[0]
    print(f"  {wt:8.1f} | {fwd:+8.4f} | {lat:+8.4f} | {grav} | {np.max(np.abs(qj_rel)):12.6f}")


print("\n" + "="*100)
print("TEST 2: Effect of warmup on command differentiation")
print("="*100)

for wt in [0.0, 2.0]:
    print(f"\n  Warmup = {wt}s:")
    for label, cmd in [("FWD", [0.5,0,0]), ("STAND", [0,0,0]), ("BWD", [-0.5,0,0])]:
        disp, grav, _ = run_sim(cmd, warmup_time=wt)
        fwd = -disp[1]
        print(f"    {label:8s}: fwd={fwd:+.4f}m  grav={grav}")


print("\n" + "="*100)
print("TEST 3: What does the robot look like after warmup?")
print("="*100)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

print(f"\n  Before warmup:")
print(f"    quat: {d.qpos[3:7]}")
print(f"    height: {d.qpos[2]:.4f}")
grav = get_gravity_orientation(d.qpos[3:7])
print(f"    gravity_raw: {grav}")
print(f"    gravity_remapped: {v4_remap_gravity(grav)}")

warmup_steps = int(2.0 / m.opt.timestep)
for i in range(warmup_steps):
    d.ctrl[:] = default_angles[act_to_joint]
    mujoco.mj_step(m, d)

print(f"\n  After 2s warmup:")
print(f"    quat: {d.qpos[3:7]}")
print(f"    height: {d.qpos[2]:.4f}")
grav = get_gravity_orientation(d.qpos[3:7])
print(f"    gravity_raw: {grav}")
print(f"    gravity_remapped: {v4_remap_gravity(grav)}")

# Compute tilt angle
R = quat_to_rotmat(d.qpos[3:7])
# Body +Z axis in world frame (forward direction)
body_z_world = R @ np.array([0, 0, 1])
# Body +Y axis in world frame (up direction)
body_y_world = R @ np.array([0, 1, 0])
print(f"    body +Z (fwd) in world: {body_z_world}")
print(f"    body +Y (up) in world: {body_y_world}")
tilt_angle = np.degrees(np.arctan2(body_z_world[2], -body_z_world[1]))
print(f"    forward tilt angle: {tilt_angle:.2f}°")

# Joint deviations
qj_after = d.qpos[7:].copy()
qj_dev = qj_after - default_angles
print(f"\n    Joint deviations from default (MuJoCo order):")
for i, name in enumerate(mj_names):
    if abs(qj_dev[i]) > 0.001:
        print(f"      {name:12s}: {qj_dev[i]:+.6f} rad ({np.degrees(qj_dev[i]):+.3f}°)")
