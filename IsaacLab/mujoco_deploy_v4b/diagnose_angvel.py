#!/usr/bin/env python3
"""验证MuJoCo qvel[3:6]是body frame还是world frame
以及world_to_body双重旋转的影响

同时测试：不用world_to_body转换ang_vel时的行为
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=6, suppress=True, linewidth=120)

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


# ============================================================
# TEST 1: Verify MuJoCo angular velocity frame
# ============================================================
print("="*70)
print("TEST 1: MuJoCo qvel[3:6] frame verification")
print("="*70)

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = 0.005

# Set initial pose
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]

# Give a known angular velocity in world frame (around world Z axis)
# For quat (w,x,y,z) = (0.7071, 0.7071, 0, 0), this is X+90° rotation
# World Z axis in body frame = body Y axis
# So world angvel [0,0,1] should appear as body angvel [0,1,0]

# Method: set qvel directly and check what MuJoCo reports
d.qvel[3:6] = [0, 0, 1.0]  # Set angular velocity
mujoco.mj_forward(m, d)

quat = d.qpos[3:7]
R = quat_to_rotmat(quat)

print(f"  quat: {quat}")
print(f"  R:\n{R}")
print(f"  qvel[3:6] = {d.qvel[3:6]}")

# Check: is qvel[3:6] in body or world frame?
# If body frame: qvel[3:6] = [0,0,1] means rotation around body Z
# If world frame: qvel[3:6] = [0,0,1] means rotation around world Z

# For a free joint in MuJoCo, qvel[3:6] is angular velocity in LOCAL (body) frame
# See: https://mujoco.readthedocs.io/en/stable/modeling.html#cangvel
print(f"\n  MuJoCo docs: For free joints, qvel[3:6] is angular velocity in LOCAL frame")
print(f"  So qvel[3:6]=[0,0,1] means rotation around body Z axis")

# What does world_to_body do to this?
angvel_double_rotated = world_to_body(d.qvel[3:6], quat)
print(f"\n  world_to_body(qvel[3:6]) = R^T @ [0,0,1] = {angvel_double_rotated}")
print(f"  This is WRONG - it's double-rotating a body-frame vector")
print(f"  Correct: just use qvel[3:6] directly as body-frame angular velocity")

# What about linear velocity?
print(f"\n  For linear velocity:")
print(f"  MuJoCo docs: qvel[0:3] is linear velocity in WORLD frame")
print(f"  So world_to_body(qvel[0:3]) is CORRECT for lin_vel")

# ============================================================
# TEST 2: Impact of angular velocity double-rotation
# ============================================================
print("\n" + "="*70)
print("TEST 2: Impact of ang_vel double-rotation on obs")
print("="*70)

# At the initial pose (X+90°), R^T maps:
# body [1,0,0] -> [1,0,0] (X stays X)
# body [0,1,0] -> [0,0,1] (Y maps to Z)
# body [0,0,1] -> [0,-1,0] (Z maps to -Y)

# So if true body angvel is [wx, wy, wz]:
# double-rotated = R^T @ [wx, wy, wz] = [wx, -wz, wy]
# Then v4_remap_ang_vel takes [a,b,c] -> [a, c, b]
# So: v4_remap(double_rotated) = v4_remap([wx, -wz, wy]) = [wx, wy, -wz]
# Correct: v4_remap(body) = v4_remap([wx, wy, wz]) = [wx, wz, wy]

# The difference: 
# With double rotation: [wx, wy, -wz]
# Without (correct):    [wx, wz, wy]
# Components 1 and 2 are SWAPPED and component 2 is NEGATED!

print(f"  At initial pose (X+90°):")
print(f"  True body angvel [wx, wy, wz]:")
print(f"    With double rotation: v4_remap(R^T @ [wx,wy,wz]) = [wx, wy, -wz]")
print(f"    Without (correct):    v4_remap([wx,wy,wz])        = [wx, wz, wy]")
print(f"  ")
print(f"  obs[3:6] mapping:")
print(f"    obs[3] = roll rate  (correct: wx, double: wx)  -- SAME")
print(f"    obs[4] = pitch rate (correct: wz, double: wy)  -- SWAPPED")
print(f"    obs[5] = yaw rate   (correct: wy, double: -wz) -- SWAPPED & NEGATED")
print(f"  ")
print(f"  This means the policy sees WRONG pitch and yaw rates!")
print(f"  When the robot pitches forward, the policy thinks it's yawing (and vice versa)")

# ============================================================
# TEST 3: Run simulation with FIXED ang_vel (no double rotation)
# ============================================================
print("\n" + "="*70)
print("TEST 3: Simulation with corrected ang_vel (no double rotation)")
print("="*70)

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

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

def run_sim(cmd_values, fix_angvel=False, fix_linvel=False, num_steps=250, label=""):
    """Run simulation with optional fixes"""
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
            
            # Linear velocity: qvel[0:3] is WORLD frame -> need world_to_body
            if fix_linvel:
                # Already body frame (wrong assumption test)
                base_lin_vel = d.qvel[0:3].copy()
            else:
                base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
            
            # Angular velocity: qvel[3:6] is BODY frame
            if fix_angvel:
                # Use directly (correct)
                omega = d.qvel[3:6].copy()
            else:
                # Double rotation (current buggy code)
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
    return disp

# Test all combinations
cmds = [
    ("STAND", [0, 0, 0]),
    ("FWD",   [0.5, 0, 0]),
    ("BWD",   [-0.5, 0, 0]),
    ("LEFT",  [0, 0.3, 0]),
    ("TURN",  [0, 0, 0.5]),
]

print(f"\n{'Mode':30s} | {'Cmd':6s} | {'dx':>8s} | {'dy':>8s} | {'dz':>8s} | {'fwd(-Y)':>8s}")
print("-" * 80)

for mode_label, fix_ang, fix_lin in [
    ("CURRENT (double-rot angvel)", False, False),
    ("FIX angvel (no rotation)",    True,  False),
]:
    for cmd_label, cmd_val in cmds:
        disp = run_sim(cmd_val, fix_angvel=fix_ang, fix_linvel=fix_lin)
        print(f"  {mode_label:28s} | {cmd_label:6s} | {disp[0]:+8.4f} | {disp[1]:+8.4f} | {disp[2]:+8.4f} | {-disp[1]:+8.4f}")
    print()
