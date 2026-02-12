#!/usr/bin/env python3
"""Test: does the robot drift forward even without policy?
Just hold default pose with PD controller and see if it moves.
"""
import numpy as np
import mujoco
import yaml
import os

np.set_printoptions(precision=6, suppress=True, linewidth=150)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)

mj_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_names.append(jname)

act_to_joint = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    act_to_joint.append(mj_names.index(joint_name))
act_to_joint = np.array(act_to_joint)

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def get_gravity_orientation(q):
    w, x, y, z = q
    gx = -2*(x*z - w*y)
    gy = -2*(y*z + w*x)
    gz = -(1 - 2*(x*x + y*y))
    return np.array([gx, gy, gz])

# Test 1: Hold default pose for 10 seconds
print("="*80)
print("TEST 1: Hold default pose (no policy) for 10 seconds")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

init_pos = d.qpos[0:3].copy()
total_steps = int(10.0 / m.opt.timestep)

for step in range(total_steps):
    d.ctrl[:] = default_angles[act_to_joint]
    mujoco.mj_step(m, d)
    
    if step % int(1.0 / m.opt.timestep) == 0:
        t = step * m.opt.timestep
        pos = d.qpos[0:3]
        disp = pos - init_pos
        quat = d.qpos[3:7]
        grav = get_gravity_orientation(quat)
        R = quat_to_rotmat(quat)
        body_z_world = R @ np.array([0, 0, 1])
        tilt = np.degrees(np.arcsin(body_z_world[2]))
        print(f"  t={t:5.1f}s: pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:.4f}) "
              f"fwd(-Y)={-disp[1]:+.4f}m  tilt={tilt:+.2f}°  "
              f"grav_y={grav[1]:+.4f}")

final_pos = d.qpos[0:3].copy()
disp = final_pos - init_pos
print(f"\n  Final: fwd(-Y)={-disp[1]:+.4f}m  lat(X)={disp[0]:+.4f}m")

# Test 2: Check CoM position relative to support polygon
print("\n" + "="*80)
print("TEST 2: CoM analysis at default pose")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0
mujoco.mj_forward(m, d)

# Get body positions
print("\n  Body positions (world frame):")
for bi in range(m.nbody):
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bi)
    pos = d.xpos[bi]
    mass = m.body_mass[bi]
    if mass > 0.01:
        print(f"    {bname:20s}: pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:.4f})  mass={mass:.3f}kg")

# Compute total CoM
total_mass = 0
com = np.zeros(3)
for bi in range(m.nbody):
    mass = m.body_mass[bi]
    com += mass * d.xpos[bi]
    total_mass += mass
com /= total_mass
print(f"\n  Total CoM: ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:.4f})")
print(f"  Total mass: {total_mass:.3f}kg")

# Get foot positions (contact points)
print("\n  Foot contact positions:")
for gi in range(m.ngeom):
    gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
    if gname and ('foot' in gname.lower() or 'ankle' in gname.lower() or 'sole' in gname.lower()):
        pos = d.geom_xpos[gi]
        print(f"    {gname:20s}: ({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:.4f})")

# After warmup
print("\n  After 2s warmup:")
warmup_steps = int(2.0 / m.opt.timestep)
for _ in range(warmup_steps):
    d.ctrl[:] = default_angles[act_to_joint]
    mujoco.mj_step(m, d)

mujoco.mj_forward(m, d)
total_mass = 0
com = np.zeros(3)
for bi in range(m.nbody):
    mass = m.body_mass[bi]
    com += mass * d.xpos[bi]
    total_mass += mass
com /= total_mass
print(f"  Total CoM: ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:.4f})")

# Check contacts
print(f"\n  Active contacts: {d.ncon}")
for ci in range(d.ncon):
    c = d.contact[ci]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"    {g1} <-> {g2}: pos=({c.pos[0]:+.4f},{c.pos[1]:+.4f},{c.pos[2]:.4f})")
