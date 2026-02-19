"""
Test: Does MuJoCo stack body quat and freejoint qpos[3:7]?

v6_robot.xml has:
  <body name="base_link" pos="0 0 0.55" quat="0.7071068 0.0 0.0 0.7071068" ...>
      <freejoint/>

This quat is a 90-degree rotation around X axis.

Question: If we set d.qpos[3:7] = [0.7071068, 0, 0, 0.7071068] (same quat),
does the world orientation become 180 degrees (double rotation)?
Or does freejoint qpos override the body quat?

Expected MuJoCo behavior: freejoint qpos IS the world pose, it REPLACES
(not stacks with) the body quat. The body quat only sets the initial
default value of qpos[3:7].

Let's verify.
"""

import mujoco
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "v6_scene.xml")

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Find base_link body id
base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
print(f"base_link body id: {base_id}")

# ============================================================
# Test 1: Default state after mj_resetData
# ============================================================
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)

print(f"\n{'='*60}")
print("Test 1: After mj_resetData (default state)")
print(f"  d.qpos[0:7] = {d.qpos[0:7]}")
print(f"  d.xpos[base_id] = {d.xpos[base_id]}")
print(f"  d.xquat[base_id] = {d.xquat[base_id]}")
print(f"  (xquat is the WORLD orientation of the body)")

# ============================================================
# Test 2: Set qpos quat to identity [1,0,0,0]
# ============================================================
mujoco.mj_resetData(m, d)
d.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)

print(f"\n{'='*60}")
print("Test 2: qpos[3:7] = [1, 0, 0, 0] (identity)")
print(f"  d.qpos[0:7] = {d.qpos[0:7]}")
print(f"  d.xpos[base_id] = {d.xpos[base_id]}")
print(f"  d.xquat[base_id] = {d.xquat[base_id]}")

# ============================================================
# Test 3: Set qpos quat to the body quat [0.7071068, 0, 0, 0.7071068]
# ============================================================
mujoco.mj_resetData(m, d)
d.qpos[3:7] = [0.7071068, 0, 0, 0.7071068]
mujoco.mj_forward(m, d)

print(f"\n{'='*60}")
print("Test 3: qpos[3:7] = [0.7071068, 0, 0, 0.7071068] (90deg around X)")
print(f"  d.qpos[0:7] = {d.qpos[0:7]}")
print(f"  d.xpos[base_id] = {d.xpos[base_id]}")
print(f"  d.xquat[base_id] = {d.xquat[base_id]}")

# ============================================================
# Analysis
# ============================================================
print(f"\n{'='*60}")
print("Analysis:")
print("  If MuJoCo REPLACES body quat with freejoint qpos:")
print("    Test 2 xquat should be [1,0,0,0] (identity)")
print("    Test 3 xquat should be [0.707,0,0,0.707] (90deg X)")
print("  If MuJoCo STACKS body quat with freejoint qpos:")
print("    Test 2 xquat should be [0.707,0,0,0.707] (body quat only)")
print("    Test 3 xquat should be [0,0,0,1] or similar (180deg)")
print()

# Also check: what does mj_resetData set qpos[3:7] to?
mujoco.mj_resetData(m, d)
print(f"After mj_resetData, qpos[3:7] = {d.qpos[3:7]}")
print(f"  This tells us what MuJoCo considers the 'default' qpos for the freejoint.")
print(f"  If it's [0.707,0,0,0.707], then MuJoCo copies body quat as default qpos.")
print(f"  If it's [1,0,0,0], then body quat is separate from qpos.")

# ============================================================
# Gravity projection test
# ============================================================
print(f"\n{'='*60}")
print("Gravity projection test:")
print("  The robot's URDF has base_link rotated 90deg around X.")
print("  In IsaacLab, the initial orientation is (0.7071068, 0, 0, 0.7071068).")
print("  get_gravity_orientation() uses d.qpos[3:7] to compute projected gravity.")
print()

def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

# What gravity does the policy expect when standing upright?
# In IsaacLab, projected_gravity for upright robot should be [0, 0, -1]
# But the robot's "upright" in world frame has quat = (0.7071068, 0, 0, 0.7071068)

quat_90x = np.array([0.7071068, 0, 0, 0.7071068])
quat_identity = np.array([1, 0, 0, 0])

g_90x = get_gravity_orientation(quat_90x)
g_identity = get_gravity_orientation(quat_identity)

print(f"  gravity with quat [0.707,0,0,0.707]: {g_90x}")
print(f"  gravity with quat [1,0,0,0]:          {g_identity}")
print()
print(f"  For the robot to be 'upright' (gravity=[0,0,-1]),")
print(f"  which quat gives [0,0,-1]?")
print(f"    quat [0.707,0,0,0.707] -> gravity = {g_90x}")
print(f"    quat [1,0,0,0]         -> gravity = {g_identity}")
