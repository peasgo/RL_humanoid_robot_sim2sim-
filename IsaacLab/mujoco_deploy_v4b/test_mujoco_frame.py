#!/usr/bin/env python3
"""Empirically verify MuJoCo qvel[3:6] frame convention.

Method: Apply a known world-frame angular velocity by stepping the simulation
and checking how qvel[3:6] changes.
"""
import numpy as np
import mujoco
import os

np.set_printoptions(precision=6, suppress=True, linewidth=120)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = 0.005
m.opt.gravity[:] = 0  # Disable gravity for clean test

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

# Test 1: Set qvel[3:6] and see what happens to quaternion
print("="*70)
print("TEST: MuJoCo qvel[3:6] frame convention")
print("="*70)

# Initial pose: X+90° rotation
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5  # High up to avoid ground contact
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qvel[:] = 0

# Disable all actuators
d.ctrl[:] = 0

quat_before = d.qpos[3:7].copy()
R_before = quat_to_rotmat(quat_before)
print(f"\nInitial quat: {quat_before}")
print(f"Initial R:\n{R_before}")

# Set angular velocity to [0, 0, 1] in qvel
d.qvel[3:6] = [0, 0, 1.0]
print(f"\nSet qvel[3:6] = [0, 0, 1]")

# Step once
mujoco.mj_step(m, d)
quat_after = d.qpos[3:7].copy()
R_after = quat_to_rotmat(quat_after)

# Compute the rotation that happened
dR = R_after @ R_before.T  # World-frame rotation
dR_body = R_before.T @ R_after  # Body-frame rotation

# Extract axis-angle from dR
# For small angle: dR ≈ I + [w]× * dt
# So [w]× ≈ (dR - I) / dt
dt = m.opt.timestep
skew_world = (dR - np.eye(3)) / dt
skew_body = (dR_body - np.eye(3)) / dt

# Extract angular velocity from skew-symmetric matrix
w_world = np.array([skew_world[2,1], skew_world[0,2], skew_world[1,0]])
w_body = np.array([skew_body[2,1], skew_body[0,2], skew_body[1,0]])

print(f"\nAfter 1 step (dt={dt}):")
print(f"  quat_after: {quat_after}")
print(f"  Inferred world-frame angular velocity: {w_world}")
print(f"  Inferred body-frame angular velocity:  {w_body}")
print(f"  qvel[3:6] after step: {d.qvel[3:6]}")

if np.allclose(w_body, [0, 0, 1], atol=0.1):
    print(f"\n  ✓ qvel[3:6] is in BODY frame")
    print(f"    world_to_body(qvel[3:6]) would be WRONG (double rotation)")
elif np.allclose(w_world, [0, 0, 1], atol=0.1):
    print(f"\n  ✓ qvel[3:6] is in WORLD frame")
    print(f"    world_to_body(qvel[3:6]) would be CORRECT")
else:
    print(f"\n  ⚠ Neither matches cleanly")

# Test 2: Try with a different initial orientation
print(f"\n{'='*70}")
print(f"TEST 2: Identity orientation")
print(f"{'='*70}")

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5
d.qpos[3:7] = [1, 0, 0, 0]  # Identity
d.qvel[:] = 0
d.qvel[3:6] = [0, 0, 1.0]

mujoco.mj_step(m, d)
quat_after2 = d.qpos[3:7].copy()
R_after2 = quat_to_rotmat(quat_after2)
dR2 = R_after2  # R_before is identity
skew2 = (dR2 - np.eye(3)) / dt
w2 = np.array([skew2[2,1], skew2[0,2], skew2[1,0]])
print(f"  Identity orientation: qvel[3:6]=[0,0,1] -> inferred w = {w2}")
print(f"  (At identity, body=world, so this doesn't distinguish)")

# Test 3: X+90° orientation, set qvel to world Z rotation
print(f"\n{'='*70}")
print(f"TEST 3: X+90° orientation, various qvel settings")
print(f"{'='*70}")

# At X+90°: body X = world X, body Y = world Z, body Z = world -Y
# So:
# body [0,0,1] = world [0,-1,0] (rotation around world -Y)
# world [0,0,1] = body [0,1,0] (rotation around body Y)

for label, qvel_set in [
    ("qvel=[0,0,1]", [0, 0, 1]),
    ("qvel=[0,1,0]", [0, 1, 0]),
    ("qvel=[1,0,0]", [1, 0, 0]),
]:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.5
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qvel[:] = 0
    d.qvel[3:6] = qvel_set
    
    quat_b = d.qpos[3:7].copy()
    R_b = quat_to_rotmat(quat_b)
    
    mujoco.mj_step(m, d)
    quat_a = d.qpos[3:7].copy()
    R_a = quat_to_rotmat(quat_a)
    
    dR = R_a @ R_b.T
    skew = (dR - np.eye(3)) / dt
    w_world = np.array([skew[2,1], skew[0,2], skew[1,0]])
    
    dR_body = R_b.T @ R_a
    skew_b = (dR_body - np.eye(3)) / dt
    w_body = np.array([skew_b[2,1], skew_b[0,2], skew_b[1,0]])
    
    print(f"  {label}: world_w={w_world}  body_w={w_body}")

print(f"\n  At X+90°: body Z = world -Y")
print(f"  If qvel is BODY frame: qvel=[0,0,1] should give world_w=[0,-1,0]")
print(f"  If qvel is WORLD frame: qvel=[0,0,1] should give world_w=[0,0,1]")

# Also check linear velocity
print(f"\n{'='*70}")
print(f"TEST 4: Linear velocity frame check")
print(f"{'='*70}")

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qvel[:] = 0
d.qvel[0:3] = [0, 0, 1.0]  # Set linear velocity

pos_before = d.qpos[0:3].copy()
mujoco.mj_step(m, d)
pos_after = d.qpos[0:3].copy()
dpos = (pos_after - pos_before) / dt

print(f"  qvel[0:3] = [0, 0, 1]")
print(f"  Position change / dt = {dpos}")
print(f"  If WORLD frame: should move in world Z -> dpos ≈ [0, 0, 1]")
print(f"  If BODY frame: body Z = world -Y -> dpos ≈ [0, -1, 0]")
