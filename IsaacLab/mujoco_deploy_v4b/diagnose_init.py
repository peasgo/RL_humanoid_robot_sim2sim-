#!/usr/bin/env python3
"""诊断初始化阶段的运动：为什么step 0就有大速度？

检查：
1. 初始化后mj_forward的状态
2. 4个sim步后（第一个policy步前）的状态
3. 是否是重力导致的下落冲击
4. 对比不同初始高度的影响
"""
import numpy as np
import mujoco
import yaml
import os

np.set_printoptions(precision=6, suppress=True, linewidth=150)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_body(v, q): return quat_to_rotmat(q).T @ v
def get_gravity(q):
    w, x, y, z = q
    return np.array([-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))])
def v4_remap_lin(v): return np.array([v[2], v[0], v[1]])
def v4_remap_ang(v): return np.array([v[0], v[2], v[1]])

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)

# 获取关节名
mj_names = []
for jid in range(m.njnt):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_names.append(jn)

act_to_jnt = []
for i in range(m.nu):
    jid = m.actuator_trnid[i, 0]
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    act_to_jnt.append(mj_names.index(jn))
act_to_jnt = np.array(act_to_jnt)

# ============================================================
# Test: 逐步跟踪初始化阶段
# ============================================================
print("="*80)
print("逐步跟踪初始化阶段（无policy，只有PD控制到default_angles）")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

target = default_angles.copy()
d.ctrl[:] = target[act_to_jnt]

# 先做一次forward看初始状态
mujoco.mj_forward(m, d)
print(f"\n初始化后 mj_forward:")
print(f"  pos = {d.qpos[0:3]}")
print(f"  quat = {d.qpos[3:7]}")
print(f"  qvel[0:6] = {d.qvel[0:6]}")
print(f"  ncon = {d.ncon}")

# 检查接触
for i in range(d.ncon):
    c = d.contact[i]
    geom1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    geom2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"  contact[{i}]: {geom1} <-> {geom2}, dist={c.dist:.6f}, pos={c.pos}")

# 逐步仿真
for step in range(20):
    d.ctrl[:] = target[act_to_jnt]
    mujoco.mj_step(m, d)
    
    quat = d.qpos[3:7]
    lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
    omega = world_to_body(d.qvel[3:6].copy(), quat)
    
    if step < 8 or step == 19:
        print(f"\nStep {step+1} (t={d.time:.4f}s):")
        print(f"  pos = {d.qpos[0:3]}")
        print(f"  quat = {quat}")
        print(f"  world_vel = {d.qvel[0:3]}")
        print(f"  body_lin_vel = {lin_vel_b}")
        print(f"  body_ang_vel = {omega}")
        print(f"  remap_lin = {v4_remap_lin(lin_vel_b)}")
        print(f"  remap_ang = {v4_remap_ang(omega)}")
        print(f"  height = {d.qpos[2]:.6f}")
        print(f"  ncon = {d.ncon}")
        
        # 检查关节偏差
        qj_diff = d.qpos[7:] - default_angles
        max_diff_idx = np.argmax(np.abs(qj_diff))
        print(f"  max joint deviation: {mj_names[max_diff_idx]} = {qj_diff[max_diff_idx]:+.6f}")
        print(f"  joint vel max: {np.max(np.abs(d.qvel[6:])):.6f} at {mj_names[np.argmax(np.abs(d.qvel[6:]))]}")


# ============================================================
# Test 2: 检查不同初始高度
# ============================================================
print("\n" + "="*80)
print("不同初始高度对比（4步后的速度）")
print("="*80)

for h in [0.18, 0.20, 0.22, 0.24, 0.26]:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = h
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    for step in range(4):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
    
    quat = d.qpos[3:7]
    lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
    omega = world_to_body(d.qvel[3:6].copy(), quat)
    
    print(f"  h={h:.2f}: pos={d.qpos[0:3]}, "
          f"remap_lin={v4_remap_lin(lin_vel_b)}, "
          f"remap_ang={v4_remap_ang(omega)}, "
          f"ncon={d.ncon}")


# ============================================================
# Test 3: 检查是否有初始穿透
# ============================================================
print("\n" + "="*80)
print("检查初始穿透（mj_forward后的接触距离）")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0
mujoco.mj_forward(m, d)

print(f"ncon = {d.ncon}")
for i in range(d.ncon):
    c = d.contact[i]
    geom1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
    geom2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
    print(f"  contact[{i}]: {geom1} <-> {geom2}")
    print(f"    dist={c.dist:.6f}, pos={c.pos}")
    print(f"    frame={c.frame[:3]}")  # contact normal

# 检查所有geom的位置
print(f"\n所有geom位置:")
for i in range(m.ngeom):
    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
    pos = d.geom_xpos[i]
    if pos[2] < 0.1:  # 只显示低位的geom
        print(f"  {gn:20s}: pos={pos}")
