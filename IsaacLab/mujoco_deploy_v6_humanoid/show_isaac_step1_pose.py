"""
Set MuJoCo robot to exact IsaacLab Step 1 joint positions and freeze.
Zoom in and inspect the pose.
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "v6_robot.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
default_angles = np.array(config["default_angles"], dtype=np.float32)
isaac_joint_order = config["isaac_joint_order"]

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = config["simulation_dt"]

# Joint mappings
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

isaac_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac_joint_order], dtype=np.int32)
default_angles_isaac = default_angles[isaac_to_mujoco].copy()

# Actuator mapping
actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    pd_index = mj_joint_names.index(joint_name)
    actuator_to_joint_indices.append(pd_index)
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

# ================================================================
# IsaacLab Step 1 joint positions (from obs[9:22] + default)
# obs_step1[9:22] = [-0.0438, -0.0435, 0.0053, -0.0576, -0.0920, -0.0718,
#                     -0.0708, -0.1467, -0.0751, 0.0455, 0.0816, -0.1083,
#                      0.0174]
# ================================================================
isaac_jpos_rel_step1 = np.array([
    -0.0438, -0.0435,  0.0053, -0.0576, -0.0920, -0.0718,
    -0.0708, -0.1467, -0.0751,  0.0455,  0.0816, -0.1083,
     0.0174
], dtype=np.float32)

isaac_jpos_step1 = isaac_jpos_rel_step1 + default_angles_isaac

print("=" * 60)
print("IsaacLab Step 1 joint positions (Isaac BFS order):")
print("=" * 60)
for i, jname in enumerate(isaac_joint_order):
    print(f"  [{i:2d}] {jname:14s}  pos={isaac_jpos_step1[i]:+.6f}  "
          f"(rel={isaac_jpos_rel_step1[i]:+.4f}  def={default_angles_isaac[i]:+.4f})")

# Set robot state
d.qpos[0:3] = [0, 0, 0.55]
d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
for i_isaac in range(len(isaac_jpos_step1)):
    mj_idx = isaac_to_mujoco[i_isaac]
    d.qpos[7 + mj_idx] = isaac_jpos_step1[i_isaac]
d.qvel[:] = 0

# Set ctrl to hold this pose
target = np.array(default_angles, dtype=np.float32)
for i_isaac in range(len(isaac_jpos_step1)):
    mj_idx = isaac_to_mujoco[i_isaac]
    target[mj_idx] = isaac_jpos_step1[i_isaac]

mujoco.mj_forward(m, d)

print(f"\nRobot set to IsaacLab Step 1 pose. Viewer open - zoom in to inspect.")
print(f"Close viewer to exit.\n")

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # Hold position with PD
        d.ctrl[:] = target[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.01)

print("Done.")
