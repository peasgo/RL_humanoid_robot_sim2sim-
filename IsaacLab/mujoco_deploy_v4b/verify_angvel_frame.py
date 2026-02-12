#!/usr/bin/env python3
"""Verify MuJoCo angular velocity frame convention with actual simulation data.

MuJoCo docs say qvel[3:6] for free joint is angular velocity in LOCAL (body) frame.
But let's verify by comparing with numerical differentiation of quaternion.

Also: check if the issue is that MuJoCo uses a DIFFERENT body frame convention.
"""
import numpy as np
import mujoco
import yaml
import os

np.set_printoptions(precision=8, suppress=True, linewidth=150)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")

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

def quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both in (w,x,y,z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)

# Setup
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

# Give it a known angular velocity to test
# Set body-frame angular velocity
d.qvel[3:6] = [0.5, 0.0, 0.0]  # rotation around body X axis

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

d.ctrl[:] = default_angles[act_to_joint]

print("="*100)
print("TEST 1: Verify angular velocity frame")
print("="*100)

# Record state before step
q_before = d.qpos[3:7].copy()
omega_mj = d.qvel[3:6].copy()

print(f"\n  Before step:")
print(f"    quat: {q_before}")
print(f"    qvel[3:6]: {omega_mj}")

# Take one step
mujoco.mj_step(m, d)

q_after = d.qpos[3:7].copy()
dt = m.opt.timestep

print(f"\n  After step (dt={dt}):")
print(f"    quat: {q_after}")

# Compute angular velocity from quaternion change
# dq/dt = 0.5 * q * omega_quat (body frame)
# omega_quat = 2 * q_conj * dq/dt
dq = (q_after - q_before) / dt
omega_quat = 2 * quat_multiply(quat_conjugate(q_before), dq)
omega_body_from_quat = omega_quat[1:]  # xyz components

# Or: dq/dt = 0.5 * omega_world_quat * q (world frame)
omega_world_quat = 2 * quat_multiply(dq, quat_conjugate(q_before))
omega_world_from_quat = omega_world_quat[1:]

# Also compute using rotation matrix
R_before = quat_to_rotmat(q_before)
omega_world_from_body = R_before @ omega_mj
omega_body_from_world = R_before.T @ omega_world_from_quat

print(f"\n  Angular velocity analysis:")
print(f"    qvel[3:6] (MuJoCo):                    {omega_mj}")
print(f"    omega_body (from quat diff):            {omega_body_from_quat}")
print(f"    omega_world (from quat diff):           {omega_world_from_quat}")
print(f"    R @ qvel[3:6] (body->world):            {omega_world_from_body}")
print(f"    R^T @ omega_world (world->body):        {omega_body_from_world}")

# Check which one matches
diff_body = np.linalg.norm(omega_mj - omega_body_from_quat)
diff_world = np.linalg.norm(omega_mj - omega_world_from_quat)
print(f"\n    |qvel[3:6] - omega_body|:  {diff_body:.8f}")
print(f"    |qvel[3:6] - omega_world|: {diff_world:.8f}")

if diff_body < diff_world:
    print(f"    → qvel[3:6] is BODY frame ✓")
else:
    print(f"    → qvel[3:6] is WORLD frame ✓")


print("\n" + "="*100)
print("TEST 2: What does IsaacLab's quat_apply_inverse do?")
print("="*100)

# IsaacLab: root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
# This is R^T @ ang_vel_world

# If MuJoCo gives body-frame angular velocity, then:
# To match IsaacLab, we need: R^T @ (R @ omega_body) = omega_body
# So we should use qvel[3:6] directly!

# But if MuJoCo gives world-frame angular velocity, then:
# To match IsaacLab, we need: R^T @ omega_world = world_to_body(qvel[3:6])

# The current code does: world_to_body(qvel[3:6])
# If qvel[3:6] is body frame, this gives R^T @ omega_body (WRONG - double rotation)
# If qvel[3:6] is world frame, this gives R^T @ omega_world (CORRECT)

print(f"\n  For quat = {q_before}:")
print(f"  qvel[3:6] = {omega_mj}")
print(f"  world_to_body(qvel[3:6]) = {world_to_body(omega_mj, q_before)}")
print(f"  body_to_world(qvel[3:6]) = {body_to_world(omega_mj, q_before)}")

# Now let's check with a more complex rotation
print("\n" + "="*100)
print("TEST 3: Verify with free-falling robot (no control)")
print("="*100)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5  # Higher up
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0
d.qvel[3] = 1.0  # Angular velocity around body X
d.qvel[4] = 0.5  # Angular velocity around body Y
d.qvel[5] = -0.3  # Angular velocity around body Z

# Don't apply any control
d.ctrl[:] = 0

for step in range(5):
    q_before = d.qpos[3:7].copy()
    omega_mj = d.qvel[3:6].copy()
    
    mujoco.mj_step(m, d)
    
    q_after = d.qpos[3:7].copy()
    dq = (q_after - q_before) / dt
    
    # Body frame angular velocity from quaternion
    omega_body_quat = 2 * quat_multiply(quat_conjugate(q_before), dq)[1:]
    # World frame angular velocity from quaternion
    omega_world_quat = 2 * quat_multiply(dq, quat_conjugate(q_before))[1:]
    
    R = quat_to_rotmat(q_before)
    omega_world_from_body = R @ omega_mj
    
    diff_body = np.linalg.norm(omega_mj - omega_body_quat)
    diff_world = np.linalg.norm(omega_mj - omega_world_quat)
    
    frame = "BODY" if diff_body < diff_world else "WORLD"
    
    print(f"\n  Step {step}:")
    print(f"    qvel[3:6]:        {omega_mj}")
    print(f"    omega_body(quat): {omega_body_quat}")
    print(f"    omega_world(quat):{omega_world_quat}")
    print(f"    R @ qvel[3:6]:    {omega_world_from_body}")
    print(f"    diff_body={diff_body:.6f}  diff_world={diff_world:.6f}  → {frame}")


print("\n" + "="*100)
print("TEST 4: Check MuJoCo documentation flag")
print("="*100)

# Check if there's a flag for angular velocity representation
print(f"  m.opt.flag: {m.opt.flag}")
# Check if there's a frame flag
print(f"  Number of free joints: {sum(1 for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)}")

# MuJoCo uses "local" angular velocity by default for free joints
# But this can be changed with the "global" flag
# Let's check the XML for any such setting
print(f"\n  Checking MuJoCo version and defaults...")
print(f"  MuJoCo version: {mujoco.__version__}")
