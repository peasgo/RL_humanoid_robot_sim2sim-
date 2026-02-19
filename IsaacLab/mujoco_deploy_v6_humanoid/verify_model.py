"""Compare URDF joint axes with MuJoCo XML joint axes.
Parses URDF XML directly and compares with compiled MuJoCo model.
"""
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import os
from scipy.spatial.transform import Rotation

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load hand-crafted MuJoCo XML
xml_path = os.path.join(current_dir, "v6_scene.xml")
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Parse URDF
urdf_path = os.path.join(current_dir, "../../URDF_Humanoid_legs_V6/urdf/URDF_Humanoid_legs_V6.urdf")
urdf_path = os.path.abspath(urdf_path)
tree = ET.parse(urdf_path)
root = tree.getroot()

print(f"URDF: {urdf_path}")
print(f"XML:  {xml_path}")

# Extract URDF joint info
urdf_joints = {}
for joint_elem in root.findall('.//joint'):
    jname = joint_elem.get('name')
    jtype = joint_elem.get('type')
    if jtype != 'revolute':
        continue
    
    origin = joint_elem.find('origin')
    xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
    rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
    
    axis_elem = joint_elem.find('axis')
    axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()]
    
    limit_elem = joint_elem.find('limit')
    lower = float(limit_elem.get('lower', '0'))
    upper = float(limit_elem.get('upper', '0'))
    
    parent = joint_elem.find('parent').get('link')
    child = joint_elem.find('child').get('link')
    
    urdf_joints[jname] = {
        'xyz': xyz, 'rpy': rpy, 'axis': axis,
        'lower': lower, 'upper': upper,
        'parent': parent, 'child': child,
    }

# Get MuJoCo joint info
mj_joints = {}
for jid in range(m.njnt):
    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    axis = m.jnt_axis[jid].copy()
    jrange = m.jnt_range[jid].copy()
    body_id = m.jnt_bodyid[jid]
    body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
    mj_joints[jname] = {
        'axis': axis, 'range': jrange, 'body': body_name
    }

# ================================================================
# Compare joint axes
# ================================================================
print(f"\n{'='*80}")
print(f"JOINT AXIS COMPARISON (URDF local axis vs MuJoCo local axis)")
print(f"{'='*80}")
print(f"Note: URDF axis is in child frame. MuJoCo axis is in body frame.")
print(f"These should match if the URDF-to-MJCF conversion is correct.")

for jname in urdf_joints:
    if jname not in mj_joints:
        print(f"\n  {jname}: NOT FOUND in MuJoCo XML!")
        continue
    
    u = urdf_joints[jname]
    m_j = mj_joints[jname]
    
    u_axis = np.array(u['axis'])
    m_axis = m_j['axis']
    
    # Normalize
    u_axis = u_axis / np.linalg.norm(u_axis)
    m_axis = m_axis / np.linalg.norm(m_axis)
    
    dot = np.dot(u_axis, m_axis)
    match = abs(abs(dot) - 1.0) < 0.01
    sign_match = dot > 0
    
    u_range = [u['lower'], u['upper']]
    m_range = m_j['range']
    
    status = "OK" if (match and sign_match) else ("SIGN FLIP" if (match and not sign_match) else "*** AXIS MISMATCH ***")
    
    print(f"\n  {jname:14s}  {status}")
    print(f"    URDF axis: [{u_axis[0]:+.6f}, {u_axis[1]:+.6f}, {u_axis[2]:+.6f}]  range: [{u_range[0]:+.4f}, {u_range[1]:+.4f}]")
    print(f"    MJ   axis: [{m_axis[0]:+.6f}, {m_axis[1]:+.6f}, {m_axis[2]:+.6f}]  range: [{m_range[0]:+.4f}, {m_range[1]:+.4f}]")
    print(f"    dot={dot:+.6f}  parent={u['parent']}  child={u['child']}  mj_body={m_j['body']}")

# ================================================================
# Kinematic test: set default pose, compute foot positions
# ================================================================
print(f"\n{'='*80}")
print(f"KINEMATIC TEST: Default pose foot positions")
print(f"{'='*80}")

default_angles_dict = {
    "pelvis_link": 0.0,
    "RHIPp": -0.2, "RHIPy": 0.0, "RHIPr": 0.0,
    "RKNEEp": 0.4, "RANKLEp": -0.2, "RANKLEy": 0.0,
    "LHIPp": -0.2, "LHIPy": 0.0, "LHIPr": 0.0,
    "LKNEEp": -0.4, "LANKLEp": 0.2, "LANKLEy": 0.0,
}

# Set default pose
d.qpos[0:3] = [0, 0, 0.55]
d.qpos[3:7] = [0.7071068, 0, 0, 0.7071068]
for jid in range(m.njnt):
    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr] = default_angles_dict[jname]

mujoco.mj_forward(m, d)

print(f"\nBody positions at default pose:")
for bid in range(m.nbody):
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
    if bname:
        pos = d.xpos[bid]
        quat = d.xquat[bid]
        print(f"  {bname:14s}  pos=[{pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f}]"
              f"  quat=[{quat[0]:+.5f}, {quat[1]:+.5f}, {quat[2]:+.5f}, {quat[3]:+.5f}]")

# Check if feet are on the ground
ranky_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "RANKLEy")
lanky_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "LANKLEy")
print(f"\n  Right foot (RANKLEy) height: {d.xpos[ranky_bid][2]:.4f}m")
print(f"  Left foot  (LANKLEy) height: {d.xpos[lanky_bid][2]:.4f}m")
print(f"  Base height: {d.xpos[1][2]:.4f}m")

# ================================================================
# Gravity vector test
# ================================================================
print(f"\n{'='*80}")
print(f"GRAVITY VECTOR TEST")
print(f"{'='*80}")

quat = d.qpos[3:7].copy()
# Compute projected gravity in body frame
def get_gravity_orientation(quat_wxyz):
    w, x, y, z = quat_wxyz
    gx = 2.0 * (-x * z + w * y)
    gy = 2.0 * (-y * z - w * x)
    gz = 1.0 - 2.0 * (x * x + y * y)
    return np.array([gx, gy, gz])

gravity = get_gravity_orientation(quat)
print(f"  quat (w,x,y,z): {quat}")
print(f"  projected_gravity_b: {gravity}")
print(f"  Expected (upright): ~[0, 0, -1]")

print(f"\n{'='*80}")
print("Done.")
