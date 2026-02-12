#!/usr/bin/env python3
"""
Deep diagnosis: compare MuJoCo obs construction vs IsaacLab expected obs.
Focus on finding the exact obs components that differ.

Key question: at the initial pose (default angles, zero velocity),
what does each obs component look like?
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

# Isaac17/16 joint order (verified by user)
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


def main():
    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)
    
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    m.opt.timestep = cfg["simulation_dt"]
    
    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()
    
    default_angles = np.array(cfg["default_angles"], dtype=np.float64)
    action_scale = cfg.get("action_scale", 0.25)
    control_decimation = cfg.get("control_decimation", 4)
    
    # Joint mappings
    mj_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_names.append(jname)
    
    i17_to_mj = np.array([mj_names.index(n) for n in ISAAC17])
    i16_to_mj = np.array([mj_names.index(n) for n in ISAAC16])
    
    # Actuator mapping
    act_to_joint = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        act_to_joint.append(mj_names.index(joint_name))
    act_to_joint = np.array(act_to_joint)
    
    waist_mj_idx = mj_names.index('Waist_2')
    
    print("=" * 70)
    print("JOINT ORDER VERIFICATION")
    print("=" * 70)
    print(f"MuJoCo joint order: {mj_names}")
    print(f"Isaac17 order:      {ISAAC17}")
    print(f"i17_to_mj mapping:  {i17_to_mj}")
    print()
    
    # Verify: isaac17_to_mj should map each isaac17 index to the correct mj index
    print("Isaac17 -> MuJoCo mapping:")
    for i, name in enumerate(ISAAC17):
        mj_idx = i17_to_mj[i]
        print(f"  Isaac17[{i:2d}] {name:12s} -> MJ[{mj_idx:2d}] {mj_names[mj_idx]:12s}  {'✓' if name == mj_names[mj_idx] else '✗'}")
    
    print()
    print("=" * 70)
    print("INITIAL STATE OBS ANALYSIS")
    print("=" * 70)
    
    # Setup initial pose
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    # Warmup
    target_dof_pos = default_angles.copy()
    warmup_steps = int(2.0 / m.opt.timestep)
    for _ in range(warmup_steps):
        d.ctrl[:] = target_dof_pos[act_to_joint]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    
    quat = d.qpos[3:7]
    print(f"\nAfter warmup:")
    print(f"  Position: {d.qpos[0:3]}")
    print(f"  Quaternion: {quat}")
    print(f"  Height: {d.qpos[2]:.4f}")
    
    # Check joint positions after warmup
    qj_mj = d.qpos[7:].copy()
    print(f"\n  Joint positions (MuJoCo order):")
    for i, name in enumerate(mj_names):
        print(f"    MJ[{i:2d}] {name:12s}: qpos={qj_mj[i]:+.6f}  default={default_angles[i]:+.6f}  diff={qj_mj[i]-default_angles[i]:+.6f}")
    
    # Build obs the same way as deploy code
    base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
    omega = world_to_body(d.qvel[3:6].copy(), quat)
    
    qj_isaac17 = qj_mj[i17_to_mj]
    dqj_isaac17 = d.qvel[6:].copy()[i17_to_mj]
    default_angles_isaac17 = default_angles[i17_to_mj]
    
    grav = get_gravity_orientation(quat)
    
    lv = v4_remap_lin_vel(base_lin_vel)
    av = v4_remap_ang_vel(omega)
    gv = v4_remap_gravity(grav)
    qj_rel = qj_isaac17 - default_angles_isaac17
    dqj = dqj_isaac17
    
    print(f"\n  Obs components:")
    print(f"    lin_vel (body):     {base_lin_vel}")
    print(f"    lin_vel (v4 remap): {lv}")
    print(f"    ang_vel (body):     {omega}")
    print(f"    ang_vel (v4 remap): {av}")
    print(f"    gravity (body):     {grav}")
    print(f"    gravity (v4 remap): {gv}")
    print(f"    Expected gravity:   [0, 0, -1] (body -Y = up, so remap should be [0, 0, -1])")
    
    print(f"\n  Joint pos relative (Isaac17 order):")
    for i, name in enumerate(ISAAC17):
        print(f"    Isaac17[{i:2d}] {name:12s}: rel={qj_rel[i]:+.6f}")
    
    # Now run policy for a few steps with different cmds
    print()
    print("=" * 70)
    print("POLICY RESPONSE TO DIFFERENT COMMANDS (first action)")
    print("=" * 70)
    
    for cmd_label, cmd_val in [("STAND", [0,0,0]), ("FWD", [0.5,0,0]), ("BWD", [-0.5,0,0])]:
        # Reset
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.22
        d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
        d.qpos[7:] = default_angles
        d.qvel[:] = 0
        
        target_dof_pos = default_angles.copy()
        for _ in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[act_to_joint]
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
        
        quat = d.qpos[3:7]
        base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
        omega = world_to_body(d.qvel[3:6].copy(), quat)
        qj_mj = d.qpos[7:].copy()
        qj_isaac17 = qj_mj[i17_to_mj]
        dqj_isaac17 = d.qvel[6:].copy()[i17_to_mj]
        default_angles_isaac17 = default_angles[i17_to_mj]
        grav = get_gravity_orientation(quat)
        
        obs = np.zeros(62, dtype=np.float32)
        obs[0:3] = v4_remap_lin_vel(base_lin_vel)
        obs[3:6] = v4_remap_ang_vel(omega)
        obs[6:9] = v4_remap_gravity(grav)
        obs[9:12] = np.array(cmd_val, dtype=np.float32)
        obs[12:29] = qj_isaac17 - default_angles_isaac17
        obs[29:46] = dqj_isaac17
        obs[46:62] = 0  # last action = 0
        
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        
        print(f"\n  [{cmd_label}] cmd={cmd_val}")
        print(f"    obs[0:3]  lin_vel:  {obs[0:3]}")
        print(f"    obs[3:6]  ang_vel:  {obs[3:6]}")
        print(f"    obs[6:9]  gravity:  {obs[6:9]}")
        print(f"    obs[9:12] cmd:      {obs[9:12]}")
        print(f"    action (first 8):   {action[:8]}")
        print(f"    action (last 8):    {action[8:]}")
        print(f"    action mean: {action.mean():+.4f}  abs_mean: {np.abs(action).mean():.4f}")
        
        # Show what joint targets would be
        print(f"    Joint targets (action * {action_scale} + default):")
        for i, name in enumerate(ISAAC16):
            mj_idx = i16_to_mj[i]
            target = action[i] * action_scale + default_angles[mj_idx]
            print(f"      Isaac16[{i:2d}] {name:12s} -> MJ[{mj_idx:2d}]: action={action[i]:+.4f}  target={target:+.4f}  default={default_angles[mj_idx]:+.4f}")
    
    # Check if the angular velocity convention matters
    print()
    print("=" * 70)
    print("ANGULAR VELOCITY CONVENTION CHECK")
    print("=" * 70)
    print("MuJoCo qvel[3:6] is in BODY frame (verified by gyro sensor)")
    print("IsaacLab root_ang_vel_b is also in body frame")
    print("Current code: omega = world_to_body(qvel[3:6], quat) = R^T @ body_angvel")
    print("This is a DOUBLE rotation (wrong in theory)")
    print()
    print("But at initial pose with zero velocity, this doesn't matter.")
    print("The forward bias appears even at step 0 with zero velocities.")
    print("So angular velocity convention is NOT the primary cause.")
    print()
    
    # Check: does the policy produce forward-biased actions even with perfect obs?
    print("=" * 70)
    print("POLICY BIAS CHECK: zero obs except gravity")
    print("=" * 70)
    
    for cmd_label, cmd_val in [("STAND", [0,0,0]), ("FWD", [0.5,0,0]), ("BWD", [-0.5,0,0])]:
        obs_clean = np.zeros(62, dtype=np.float32)
        obs_clean[6:9] = [0, 0, -1]  # ideal gravity for V4 standing
        obs_clean[9:12] = cmd_val
        
        with torch.no_grad():
            action = policy(torch.from_numpy(obs_clean).unsqueeze(0)).squeeze(0).numpy()
        
        print(f"  [{cmd_label}] action mean={action.mean():+.4f}  abs_mean={np.abs(action).mean():.4f}")
        print(f"    action: {action}")


if __name__ == "__main__":
    main()
