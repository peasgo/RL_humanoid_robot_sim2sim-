#!/usr/bin/env python3
"""Systematic test: combine zero warmup with different angvel fixes.
Also test: what if we use body->world->body for angvel (should be identity)?

Key insight from previous test: even with warmup=0, bias exists.
So the bias develops DURING simulation. Let's trace it step by step.
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

def body_to_world(v, q):
    return quat_to_rotmat(q) @ v

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


def run_sim(cmd_values, angvel_mode="double_rot", linvel_mode="normal", 
            warmup_time=0.0, num_steps=300, trace=False):
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
    
    init_pos = d.qpos[0:3].copy()
    counter = 0
    policy_step = 0
    trace_data = []
    
    while policy_step < num_steps:
        d.ctrl[:] = target_dof_pos[act_to_joint]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_decimation == 0:
            quat = d.qpos[3:7]
            
            # Linear velocity
            if linvel_mode == "normal":
                base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
            elif linvel_mode == "direct":
                # What if qvel[0:3] is already body frame? (it's not, but let's test)
                base_lin_vel = d.qvel[0:3].copy()
            
            # Angular velocity
            if angvel_mode == "double_rot":
                omega = world_to_body(d.qvel[3:6].copy(), quat)
            elif angvel_mode == "direct":
                omega = d.qvel[3:6].copy()
            elif angvel_mode == "correct":
                # Convert body->world first, then world->body
                ang_world = body_to_world(d.qvel[3:6].copy(), quat)
                omega = world_to_body(ang_world, quat)
            
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
            
            if trace and policy_step < 10:
                trace_data.append({
                    'step': policy_step,
                    'obs': obs.copy(),
                    'quat': quat.copy(),
                    'qvel_03': d.qvel[0:3].copy(),
                    'qvel_36': d.qvel[3:6].copy(),
                    'base_lin_vel': base_lin_vel.copy(),
                    'omega': omega.copy(),
                    'gravity': gravity_orientation.copy(),
                    'pos': d.qpos[0:3].copy(),
                })
            
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
    return final_pos - init_pos, trace_data


# Test all combinations
print("="*100)
print("TEST: All angvel modes × warmup × cmd (300 steps)")
print("="*100)

configs = [
    ("double_rot, warmup=0",  "double_rot", 0.0),
    ("double_rot, warmup=2",  "double_rot", 2.0),
    ("direct, warmup=0",      "direct",     0.0),
    ("direct, warmup=2",      "direct",     2.0),
]

cmds = [("FWD", [0.5,0,0]), ("STAND", [0,0,0]), ("BWD", [-0.5,0,0])]

print(f"\n  {'Config':30s} | {'Cmd':8s} | {'fwd(-Y)':>8s} | {'lat(X)':>8s}")
print("  " + "-" * 65)

for cfg_label, angvel, warmup in configs:
    for cmd_label, cmd_val in cmds:
        disp, _ = run_sim(cmd_val, angvel_mode=angvel, warmup_time=warmup)
        fwd = -disp[1]
        lat = disp[0]
        print(f"  {cfg_label:30s} | {cmd_label:8s} | {fwd:+8.4f} | {lat:+8.4f}")
    print()


# Trace first 10 steps to see what's happening
print("\n" + "="*100)
print("TRACE: First 10 steps with STAND cmd, no warmup, double_rot")
print("="*100)

disp, trace = run_sim([0,0,0], angvel_mode="double_rot", warmup_time=0.0, trace=True)
for t in trace:
    step = t['step']
    print(f"\n  Step {step}:")
    print(f"    pos_world:     {t['pos']}")
    print(f"    quat:          {t['quat']}")
    print(f"    qvel[0:3] (world lin): {t['qvel_03']}")
    print(f"    qvel[3:6] (body ang):  {t['qvel_36']}")
    print(f"    base_lin_vel (body):    {t['base_lin_vel']}")
    print(f"    omega (double-rot):     {t['omega']}")
    print(f"    gravity_raw:            {t['gravity']}")
    print(f"    obs[0:3] lin_vel_remap: {t['obs'][0:3]}")
    print(f"    obs[3:6] ang_vel_remap: {t['obs'][3:6]}")
    print(f"    obs[6:9] grav_remap:    {t['obs'][6:9]}")
    print(f"    obs[12:16] qj_rel[:4]:  {t['obs'][12:16]}")
    print(f"    obs[29:33] dqj[:4]:     {t['obs'][29:33]}")
    print(f"    obs[46:50] act[:4]:     {t['obs'][46:50]}")
