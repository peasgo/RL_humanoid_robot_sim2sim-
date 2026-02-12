#!/usr/bin/env python3
"""验证get_gravity_orientation与IsaacLab的projected_gravity_b是否一致。
同时检查MuJoCo中qvel[3:6]到底是什么坐标系。
"""
import numpy as np
import mujoco
import os

np.set_printoptions(precision=6, suppress=True, linewidth=200)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")

def get_gravity_orientation(quaternion):
    """从run_v4_robot.py复制"""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

def quat_apply_inverse(q, v):
    """IsaacLab的quat_apply_inverse: R^T @ v"""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return R.T @ np.array(v)

def world_to_body(v, q):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return R.T @ v

# 测试1：gravity函数对比
print("="*80)
print("  测试1：gravity函数对比")
print("="*80)

quats = [
    ("初始V4", [0.70710678, 0.70710678, 0.0, 0.0]),
    ("单位", [1, 0, 0, 0]),
    ("绕Y轴30°", [0.9659, 0, 0.2588, 0]),
    ("随机", [0.5, 0.5, 0.5, 0.5]),
]

for name, q in quats:
    g1 = get_gravity_orientation(q)
    g2 = quat_apply_inverse(q, [0, 0, -1])
    diff = np.linalg.norm(g1 - g2)
    print(f"  {name:12s}: get_gravity={g1}  quat_apply_inv={g2}  diff={diff:.10f}")

# 测试2：MuJoCo qvel[3:6]的坐标系
print(f"\n{'='*80}")
print("  测试2：MuJoCo qvel[3:6]的坐标系验证")
print("="*80)

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = 0.001

# 设置初始状态
d.qpos[2] = 0.5  # 高一点避免碰撞
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]

# 给一个已知的角速度
# 如果qvel[3:6]是世界坐标系，那么qvel[3]=1表示绕世界X轴旋转
# 如果qvel[3:6]是body坐标系，那么qvel[3]=1表示绕body X轴旋转
# V4的body X轴 = 世界X轴（因为只绕X旋转了90°）
# V4的body Y轴 = 世界-Z轴
# V4的body Z轴 = 世界Y轴（前进方向）

# 方法：设置一个角速度，看四元数如何变化
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qvel[:] = 0
d.qvel[5] = 1.0  # 设置qvel[5]（第3个角速度分量）= 1.0

# 记录初始四元数
q0 = d.qpos[3:7].copy()
print(f"\n  初始quat: {q0}")
print(f"  设置qvel[5]=1.0")

# 步进一小步
mujoco.mj_step(m, d)
q1 = d.qpos[3:7].copy()
print(f"  步进后quat: {q1}")

# 计算角速度方向
# 如果qvel[5]是世界Z轴角速度，那么机器人应该绕世界Z轴旋转
# 如果qvel[5]是body Z轴角速度，那么机器人应该绕body Z轴旋转
# body Z轴 = 世界-Y轴（V4的前进方向）

# 用四元数差来推断旋转轴
dq = q1 - q0
print(f"  dq: {dq}")

# 更直接的方法：用mj_objectVelocity
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qvel[:] = 0

# 设置不同的角速度分量，看mj_objectVelocity返回什么
base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

for axis in range(3):
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.5
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qvel[:] = 0
    d.qvel[3+axis] = 1.0
    
    # 不step，直接读取
    mujoco.mj_forward(m, d)
    
    vel6_world = np.zeros(6)
    vel6_body = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6_world, 0)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6_body, 1)
    
    w2b = world_to_body(d.qvel[3:6], d.qpos[3:7])
    
    print(f"\n  qvel[{3+axis}]=1.0:")
    print(f"    qvel[3:6]          = {d.qvel[3:6]}")
    print(f"    world_to_body(qvel)= {w2b}")
    print(f"    mj_objvel world ang= {vel6_world[0:3]}")
    print(f"    mj_objvel body ang = {vel6_body[0:3]}")

# 测试3：线速度对比
print(f"\n{'='*80}")
print("  测试3：线速度对比")
print("="*80)

for axis in range(3):
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.5
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qvel[:] = 0
    d.qvel[axis] = 1.0
    
    mujoco.mj_forward(m, d)
    
    vel6_world = np.zeros(6)
    vel6_body = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6_world, 0)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6_body, 1)
    
    w2b = world_to_body(d.qvel[0:3], d.qpos[3:7])
    
    print(f"\n  qvel[{axis}]=1.0:")
    print(f"    qvel[0:3]          = {d.qvel[0:3]}")
    print(f"    world_to_body(qvel)= {w2b}")
    print(f"    mj_objvel world lin= {vel6_world[3:6]}")
    print(f"    mj_objvel body lin = {vel6_body[3:6]}")

# 测试4：关键 - 检查IsaacLab中root_lin_vel_b的计算
# IsaacLab: root_lin_vel_b = quat_apply_inverse(root_link_quat_w, root_com_lin_vel_w)
# MuJoCo: world_to_body(qvel[0:3], quat) = R^T @ qvel[0:3]
# 问题：qvel[0:3]是世界坐标系的线速度吗？
print(f"\n{'='*80}")
print("  测试4：qvel[0:3]是否是世界坐标系线速度")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.5
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qvel[:] = 0
d.qvel[0] = 0.5  # 世界X方向速度

# 步进几步看位置变化
pos0 = d.qpos[0:3].copy()
for i in range(10):
    mujoco.mj_step(m, d)
pos1 = d.qpos[0:3].copy()
dp = pos1 - pos0
print(f"  设置qvel[0]=0.5, 10步后位移: dx={dp[0]:.6f}, dy={dp[1]:.6f}, dz={dp[2]:.6f}")
print(f"  预期: dx≈{0.5*0.001*10:.6f} (如果qvel[0]是世界X速度)")
