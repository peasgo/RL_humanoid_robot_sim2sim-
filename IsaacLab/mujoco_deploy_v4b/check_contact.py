"""Quick check: is the robot making ground contact?"""
import mujoco
import numpy as np
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "v4_robot.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
default_angles = np.array(config["default_angles"], dtype=np.float32)

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = 0.005

# Print all geom names and their contype/conaffinity
print("Geom collision properties:")
for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
    contype = m.geom_contype[i]
    conaffinity = m.geom_conaffinity[i]
    body_id = m.geom_bodyid[i]
    body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    if contype > 0 or conaffinity > 0:
        print(f"  {name:30s} body={body_name:15s} contype={contype} conaffinity={conaffinity}")

# Init
d.qpos[2] = 0.3
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles

# Actuator mapping
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    actuator_to_joint_indices.append(mj_joint_names.index(joint_name))
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

target_dof_pos = default_angles.copy()

# Warmup
for _ in range(1000):
    d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
    mujoco.mj_step(m, d)

print(f"\nAfter warmup:")
print(f"  height: {d.qpos[2]:.4f}m")
print(f"  ncon: {d.ncon}")

# Print contact details
for i in range(d.ncon):
    c = d.contact[i]
    geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
    print(f"  contact[{i}]: {geom1_name} <-> {geom2_name}  pos={c.pos}  dist={c.dist:.6f}")

# Check: which feet geoms exist?
print("\nFeet-related geoms:")
for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if "feet" in name.lower() or "foot" in name.lower() or "Feet" in name:
        body_id = m.geom_bodyid[i]
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        contype = m.geom_contype[i]
        conaffinity = m.geom_conaffinity[i]
        print(f"  {name:30s} body={body_name:15s} contype={contype} conaffinity={conaffinity}")
