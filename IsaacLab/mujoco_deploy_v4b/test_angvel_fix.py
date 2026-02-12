#!/usr/bin/env python3
"""Comprehensive test: try multiple fix combinations and compare.
Focus on whether the robot DIFFERENTIATES commands, not just forward bias.

Key metric: does backward cmd produce less forward movement than forward cmd?
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=4, suppress=True, linewidth=120)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")
POLICY_PATH = os.path.join(SCRIPT_DIR,
    "../logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/exported/policy.pt")

ISAAC17 = ['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2','LSDp','RSDp',
           'LKNEEp','RKNEEP','LSDy','RSDy','LANKLEp','RANKLEp',
           'LARMp','RARMp','LARMAp','RARMAP']
ISAAC16 = ['LHIPp','RHIPp','LHIPy','RHIPy','LSDp','RSDp',
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
i16_to_mj = np.array([mj_names.index(n) for n in ISAAC16])

act_to_joint = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    act_to_joint.append(mj_names.index(joint_name))
act_to_joint = np.array(act_to_joint)
waist_mj_idx = mj_names.index('Waist_2')

policy = torch.jit.load(POLICY_PATH, map_location="cpu")
policy.eval()

action_filter_alpha = float(cfg.get("action_filter_alpha", 0.0))
obs_filter_alpha = float(cfg.get("obs_filter_alpha", 0.0))
obs_filter_mode = str(cfg.get("obs_filter_mode", "all"))
action_ramp_steps = int(cfg.get("action_ramp_steps", 0))
action_clip = cfg.get("action_clip", None)
if action_clip is not None:
    action_clip = float(action_clip)


def run_sim(cmd_values, angvel_mode="double_rot", num_steps=300):
    """
    angvel_mode:
      "double_rot" - current buggy: world_to_body(qvel[3:6])
      "direct"     - use qvel[3:6] directly (body frame)
      "world_first"- convert body->world first, then world->body (identity, same as direct)
    """
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
    
    warmup_steps = int(2.0 / m.opt.timestep)
    for _ in range(warmup_steps):
        d.ctrl[:] = target_dof_pos[act_to_joint]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    
    init_pos = d.qpos[0:3].copy()
    counter = 0
    policy_step = 0
    
    while policy_step < num_steps:
        d.ctrl[:] = target_dof_pos[act_to_joint]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_decimation == 0:
            quat = d.qpos[3:7]
            
            # Linear velocity: qvel[0:3] is WORLD frame -> convert to body
            base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
            
            # Angular velocity
            if angvel_mode == "double_rot":
                omega = world_to_body(d.qvel[3:6].copy(), quat)
            elif angvel_mode == "direct":
                omega = d.qvel[3:6].copy()
            elif angvel_mode == "world_first":
                # Convert body->world, then world->body (should be identity)
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
    return final_pos - init_pos


cmds = [
    ("STAND",  [0, 0, 0]),
    ("FWD",    [0.5, 0, 0]),
    ("BWD",    [-0.5, 0, 0]),
    ("LEFT",   [0, 0.3, 0]),
    ("TURN_L", [0, 0, 0.5]),
]

modes = [
    ("CURRENT (double-rot)", "double_rot"),
    ("FIX (direct body)",    "direct"),
    ("VERIFY (world_first)", "world_first"),
]

print("="*90)
print("COMPREHENSIVE ANGVEL FIX TEST (300 policy steps)")
print("="*90)
print(f"\n{'Mode':28s} | {'Cmd':8s} | {'dx':>8s} | {'dy':>8s} | {'dz':>8s} | {'fwd(-Y)':>8s} | {'lat(X)':>8s}")
print("-" * 90)

for mode_label, mode in modes:
    for cmd_label, cmd_val in cmds:
        disp = run_sim(cmd_val, angvel_mode=mode)
        print(f"  {mode_label:26s} | {cmd_label:8s} | {disp[0]:+8.4f} | {disp[1]:+8.4f} | {disp[2]:+8.4f} | {-disp[1]:+8.4f} | {disp[0]:+8.4f}")
    print()

# Summary analysis
print("\n" + "="*90)
print("ANALYSIS: Command differentiation quality")
print("="*90)

for mode_label, mode in modes:
    fwd_disp = run_sim([0.5, 0, 0], angvel_mode=mode)
    bwd_disp = run_sim([-0.5, 0, 0], angvel_mode=mode)
    stand_disp = run_sim([0, 0, 0], angvel_mode=mode)
    
    fwd_y = -fwd_disp[1]
    bwd_y = -bwd_disp[1]
    stand_y = -stand_disp[1]
    
    # Good differentiation: fwd > stand > bwd (or fwd > 0 > bwd)
    diff_fwd_bwd = fwd_y - bwd_y
    diff_fwd_stand = fwd_y - stand_y
    
    print(f"\n  {mode_label}:")
    print(f"    FWD forward distance:   {fwd_y:+.4f}m")
    print(f"    STAND forward distance: {stand_y:+.4f}m")
    print(f"    BWD forward distance:   {bwd_y:+.4f}m")
    print(f"    FWD-BWD difference:     {diff_fwd_bwd:+.4f}m (should be positive)")
    print(f"    FWD-STAND difference:   {diff_fwd_stand:+.4f}m (should be positive)")
    
    if fwd_y > stand_y > bwd_y:
        print(f"    ✓ Correct ordering: FWD > STAND > BWD")
    elif fwd_y > bwd_y:
        print(f"    ~ Partial: FWD > BWD but STAND not in between")
    else:
        print(f"    ✗ Wrong ordering")
