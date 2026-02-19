"""
Test: Is MuJoCo d.qvel[3:6] in world frame or body frame for free joints?

Strategy:
  1. Create a simple free-floating box
  2. Set a non-trivial orientation (45° around Y axis)
  3. Set angular velocity in world frame via mj_forward + manual qvel
  4. Compare qvel[3:6] with expected body-frame and world-frame values

If qvel[3:6] is WORLD frame:
  Setting qvel[3:6] = [0, 0, 1] means omega_world = [0, 0, 1]
  Body-frame omega would be R^T @ [0, 0, 1]

If qvel[3:6] is BODY frame:
  Setting qvel[3:6] = [0, 0, 1] means omega_body = [0, 0, 1]
  World-frame omega would be R @ [0, 0, 1]

We verify by checking d.cvel (which MuJoCo documents as [ang_vel_world, lin_vel_world])
after mj_forward.

MuJoCo docs on cvel:
  "com-based velocity [3D angular ; 3D linear] in global frame"
  So d.cvel[body_id, 0:3] = angular velocity in WORLD frame.

If qvel[3:6] is body frame, then after mj_forward:
  d.cvel[body_id, 0:3] should equal R @ qvel[3:6]

If qvel[3:6] is world frame, then:
  d.cvel[body_id, 0:3] should equal qvel[3:6]
"""

import mujoco
import numpy as np

# Create a minimal model with a free-floating box
XML = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

# Rotation: 45° around Y axis
# quat (w,x,y,z) for 45° around Y: (cos(22.5°), 0, sin(22.5°), 0)
angle = np.pi / 4  # 45 degrees
w = np.cos(angle / 2)
y = np.sin(angle / 2)
d.qpos[3:7] = [w, 0, y, 0]  # (w, x, y, z)

# Build rotation matrix from quaternion
def quat_to_rotmat(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])

R = quat_to_rotmat(d.qpos[3:7])
print(f"Quaternion (w,x,y,z): {d.qpos[3:7]}")
print(f"Rotation matrix R (body-to-world):\n{R}")

# Set qvel[3:6] = [1, 0, 0] (angular velocity around X)
test_omega = np.array([1.0, 0.0, 0.0])
d.qvel[3:6] = test_omega

# Run mj_forward to compute cvel
mujoco.mj_forward(m, d)

# Get cvel for body 1 (the box, body 0 is world)
# cvel layout: [ang_vel_world(3), lin_vel_world(3)]
cvel = d.cvel[1].copy()
omega_from_cvel = cvel[0:3]

# What we'd expect if qvel[3:6] is body frame:
omega_world_if_body = R @ test_omega

# What we'd expect if qvel[3:6] is world frame:
omega_world_if_world = test_omega

print(f"\n--- Test: qvel[3:6] = {test_omega} ---")
print(f"d.cvel[1, 0:3] (omega in world frame from MuJoCo): {omega_from_cvel}")
print(f"Expected if qvel is BODY frame:  R @ qvel = {omega_world_if_body}")
print(f"Expected if qvel is WORLD frame: qvel     = {omega_world_if_world}")

diff_body = np.linalg.norm(omega_from_cvel - omega_world_if_body)
diff_world = np.linalg.norm(omega_from_cvel - omega_world_if_world)

print(f"\nError if BODY frame hypothesis:  {diff_body:.6f}")
print(f"Error if WORLD frame hypothesis: {diff_world:.6f}")

if diff_body < 1e-6:
    print("\n>>> CONCLUSION: qvel[3:6] is in BODY (local) frame <<<")
elif diff_world < 1e-6:
    print("\n>>> CONCLUSION: qvel[3:6] is in WORLD frame <<<")
else:
    print(f"\n>>> INCONCLUSIVE: neither hypothesis matches well <<<")

# Additional test with different omega direction
print(f"\n{'='*60}")
test_omega2 = np.array([0.0, 0.0, 1.0])
d.qvel[3:6] = test_omega2
mujoco.mj_forward(m, d)
cvel2 = d.cvel[1].copy()
omega_from_cvel2 = cvel2[0:3]
omega_world_if_body2 = R @ test_omega2
omega_world_if_world2 = test_omega2

print(f"--- Test 2: qvel[3:6] = {test_omega2} ---")
print(f"d.cvel[1, 0:3] (omega in world frame from MuJoCo): {omega_from_cvel2}")
print(f"Expected if qvel is BODY frame:  R @ qvel = {omega_world_if_body2}")
print(f"Expected if qvel is WORLD frame: qvel     = {omega_world_if_world2}")

diff_body2 = np.linalg.norm(omega_from_cvel2 - omega_world_if_body2)
diff_world2 = np.linalg.norm(omega_from_cvel2 - omega_world_if_world2)
print(f"Error if BODY frame:  {diff_body2:.6f}")
print(f"Error if WORLD frame: {diff_world2:.6f}")

# Also test with the V6 humanoid's initial quaternion
print(f"\n{'='*60}")
print(f"--- Test 3: Using V6 humanoid initial quat [0.7071, 0, 0, 0.7071] ---")
d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
R3 = quat_to_rotmat(d.qpos[3:7])
print(f"R3 (body-to-world):\n{R3}")

test_omega3 = np.array([0.5, 0.3, 0.1])
d.qvel[3:6] = test_omega3
mujoco.mj_forward(m, d)
cvel3 = d.cvel[1].copy()
omega_from_cvel3 = cvel3[0:3]
omega_world_if_body3 = R3 @ test_omega3

print(f"qvel[3:6] = {test_omega3}")
print(f"cvel omega (world): {omega_from_cvel3}")
print(f"R @ qvel (if body): {omega_world_if_body3}")
print(f"Error if BODY frame: {np.linalg.norm(omega_from_cvel3 - omega_world_if_body3):.6f}")
print(f"Error if WORLD frame: {np.linalg.norm(omega_from_cvel3 - test_omega3):.6f}")

# Also check: what does IsaacLab's base_ang_vel return?
# IsaacLab: root_ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
# So if MuJoCo qvel[3:6] is already body frame, we can use it directly.
# If it's world frame, we need: R^T @ qvel[3:6]
print(f"\n{'='*60}")
print("SUMMARY for V6 humanoid deployment:")
print("  IsaacLab base_ang_vel = root_ang_vel_b (body frame)")
print("  If MuJoCo qvel[3:6] is BODY frame -> use directly (current code)")
print("  If MuJoCo qvel[3:6] is WORLD frame -> need R^T @ qvel[3:6]")
