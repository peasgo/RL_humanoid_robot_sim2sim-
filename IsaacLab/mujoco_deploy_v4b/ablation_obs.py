#!/usr/bin/env python3
"""Deep obs analysis: dump obs at step 0 (from default pose) and check each component.
Compare what MuJoCo deployment produces vs what IsaacLab would produce.

Key question: what in the obs is causing the forward bias?
Strategy: zero out different obs components and see which one eliminates the bias.
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


def run_sim_with_obs_mask(cmd_values, mask_name=None, num_steps=300):
    """Run sim, optionally zeroing out specific obs components."""
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
    obs_log = []
    
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
            
            # Apply mask AFTER filtering but BEFORE policy
            if mask_name == "zero_linvel":
                obs[0:3] = 0
            elif mask_name == "zero_angvel":
                obs[3:6] = 0
            elif mask_name == "zero_gravity":
                obs[6:9] = 0
            elif mask_name == "fix_gravity":
                # Set gravity to what it should be for upright robot
                # In IsaacLab, upright robot with quat=[0.7071, 0.7071, 0, 0]:
                # gravity_world = [0, 0, -1]
                # gravity_body = R^T @ [0, 0, -1]
                # For quat=[0.7071, 0.7071, 0, 0] (90° around X):
                # R rotates X-axis by 90°, so body Z -> world Y, body Y -> world -Z
                # gravity_body = R^T @ [0, 0, -1] = [0, 1, 0]
                # After v4_remap: [gz, gx, gy] = [0, 0, 1]
                # But let's compute it properly:
                q_upright = np.array([0.70710678, 0.70710678, 0.0, 0.0])
                g_upright = get_gravity_orientation(q_upright)
                obs[6:9] = v4_remap_gravity(g_upright)
            elif mask_name == "zero_cmd":
                obs[9:12] = 0
            elif mask_name == "zero_joint_pos":
                obs[12:29] = 0
            elif mask_name == "zero_joint_vel":
                obs[29:46] = 0
            elif mask_name == "zero_last_action":
                obs[46:62] = 0
            elif mask_name == "zero_all_vel":
                obs[0:6] = 0
                obs[29:46] = 0
            elif mask_name == "perfect_static":
                # What obs should look like for a perfectly static upright robot
                obs[0:3] = 0  # no linear vel
                obs[3:6] = 0  # no angular vel
                q_upright = np.array([0.70710678, 0.70710678, 0.0, 0.0])
                g_upright = get_gravity_orientation(q_upright)
                obs[6:9] = v4_remap_gravity(g_upright)
                # cmd stays as is
                obs[12:29] = 0  # at default pose
                obs[29:46] = 0  # no joint vel
                # last_action stays as is
            
            if policy_step < 3:
                obs_log.append(obs.copy())
            
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
    return final_pos - init_pos, obs_log


# First: dump obs at step 0 for STAND command
print("="*100)
print("STEP 1: Obs dump at first 3 steps (STAND cmd)")
print("="*100)
disp, obs_log = run_sim_with_obs_mask([0, 0, 0])
for i, obs in enumerate(obs_log):
    print(f"\n  Step {i}:")
    print(f"    lin_vel (remapped):  {obs[0:3]}")
    print(f"    ang_vel (remapped):  {obs[3:6]}")
    print(f"    gravity (remapped):  {obs[6:9]}")
    print(f"    cmd:                 {obs[9:12]}")
    print(f"    joint_pos_rel:       {obs[12:29]}")
    print(f"    joint_vel:           {obs[29:46]}")
    print(f"    last_action:         {obs[46:62]}")

# Expected gravity for upright robot
q_upright = np.array([0.70710678, 0.70710678, 0.0, 0.0])
g_raw = get_gravity_orientation(q_upright)
g_remapped = v4_remap_gravity(g_raw)
print(f"\n  Expected gravity (raw):      {g_raw}")
print(f"  Expected gravity (remapped): {g_remapped}")

# Second: ablation study - zero out each obs component
print("\n" + "="*100)
print("STEP 2: Ablation study - which obs component causes forward bias?")
print("="*100)

masks = [
    ("baseline (no mask)", None),
    ("zero_linvel",        "zero_linvel"),
    ("zero_angvel",        "zero_angvel"),
    ("zero_gravity",       "zero_gravity"),
    ("fix_gravity",        "fix_gravity"),
    ("zero_cmd",           "zero_cmd"),
    ("zero_joint_pos",     "zero_joint_pos"),
    ("zero_joint_vel",     "zero_joint_vel"),
    ("zero_last_action",   "zero_last_action"),
    ("zero_all_vel",       "zero_all_vel"),
    ("perfect_static",     "perfect_static"),
]

print(f"\n  {'Mask':25s} | {'fwd(-Y)':>8s} | {'lat(X)':>8s} | {'dz':>8s}")
print("  " + "-" * 60)

for label, mask in masks:
    disp, _ = run_sim_with_obs_mask([0, 0, 0], mask_name=mask)
    fwd = -disp[1]
    lat = disp[0]
    print(f"  {label:25s} | {fwd:+8.4f} | {lat:+8.4f} | {disp[2]:+8.4f}")

# Third: with FWD cmd, same ablation
print(f"\n  With FWD cmd [0.5, 0, 0]:")
print(f"  {'Mask':25s} | {'fwd(-Y)':>8s} | {'lat(X)':>8s} | {'dz':>8s}")
print("  " + "-" * 60)

for label, mask in [("baseline", None), ("perfect_static", "perfect_static")]:
    disp, _ = run_sim_with_obs_mask([0.5, 0, 0], mask_name=mask)
    fwd = -disp[1]
    lat = disp[0]
    print(f"  {label:25s} | {fwd:+8.4f} | {lat:+8.4f} | {disp[2]:+8.4f}")
