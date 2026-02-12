#!/usr/bin/env python3
"""验证 MuJoCo qvel[0:3] 的坐标系。

MuJoCo 文档说 free joint 的 qvel[0:3] 是世界坐标系的线速度。
但让我们实际验证一下。
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

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)

# 初始化
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

# 给一个已知的世界坐标系速度
d.qvel[0] = 0.1  # world X
d.qvel[1] = 0.0  # world Y
d.qvel[2] = 0.0  # world Z

print("初始状态:")
print(f"  qpos[0:3] = {d.qpos[0:3]}")
print(f"  qpos[3:7] = {d.qpos[3:7]}")
print(f"  qvel[0:3] = {d.qvel[0:3]}")

# 步进一步
mujoco.mj_step(m, d)

print(f"\n一步后:")
print(f"  qpos[0:3] = {d.qpos[0:3]}")
print(f"  位移 = {d.qpos[0:3] - np.array([0, 0, 0.22])}")
print(f"  期望位移(world X) ≈ [{0.1 * m.opt.timestep}, 0, ...]")

# 重置，测试 Y 方向
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0
d.qvel[0] = 0.0
d.qvel[1] = 0.1  # world Y (= V4 backward)

mujoco.mj_step(m, d)
print(f"\nY方向速度一步后:")
print(f"  qpos[0:3] = {d.qpos[0:3]}")
print(f"  位移 = {d.qpos[0:3] - np.array([0, 0, 0.22])}")
print(f"  期望位移(world Y) ≈ [0, {0.1 * m.opt.timestep}, ...]")

# 现在验证 body frame 的关系
print("\n" + "="*80)
print("验证 world_to_body 转换")
print("="*80)

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

quat = np.array([0.70710678, 0.70710678, 0.0, 0.0])
R = quat_to_rotmat(quat)
print(f"\nV4 初始四元数: {quat}")
print(f"旋转矩阵 R (world->body columns = body axes in world):")
print(f"  body X in world = {R[:, 0]}")
print(f"  body Y in world = {R[:, 1]}")
print(f"  body Z in world = {R[:, 2]}")

# 验证：body +Z 应该是 world -Y（前进方向）
print(f"\n  body +Z in world = {R[:, 2]} (期望 [0, -1, 0] = 前进)")
print(f"  body +X in world = {R[:, 0]} (期望 [1, 0, 0] = 左)")
print(f"  body +Y in world = {R[:, 1]} (期望 [0, 0, 1] = 上)")

# 测试 world_to_body
v_world = np.array([0.0, -0.3, 0.0])  # 世界-Y = 前进
v_body = R.T @ v_world
print(f"\n  world vel = {v_world} (前进0.3)")
print(f"  body vel  = {v_body}")
print(f"  期望: body [0, 0, +0.3] (body +Z = 前进)")

# V4 remap
v_remap = np.array([v_body[2], v_body[0], v_body[1]])
print(f"  V4 remap  = {v_remap}")
print(f"  期望: [+0.3, 0, 0] (remap[0] = 前进)")

# 测试角速度
print("\n" + "="*80)
print("验证角速度")
print("="*80)

# MuJoCo qvel[3:6] 是 body frame 角速度
# 如果机器人绕世界Z轴转（yaw），对应 body Y 轴
# 因为 body Y = world Z

omega_world_z = np.array([0, 0, 1.0])  # 绕世界Z轴
omega_body = R.T @ omega_world_z
print(f"\n  世界Z轴角速度 = {omega_world_z}")
print(f"  body frame = {omega_body}")
print(f"  期望: body [0, 1, 0] (body Y = world Z)")

# 如果 qvel[3:6] 已经是 body frame
# 那么 world_to_body(qvel[3:6]) = R.T @ qvel[3:6] 是错误的
# 因为这会再旋转一次

# 假设机器人绕世界Z轴以1 rad/s转
# qvel[3:6] (body frame) = [0, 1, 0]
omega_mj = np.array([0, 1, 0])  # body frame: 绕body Y
omega_double_rot = R.T @ omega_mj  # 错误的double rotation
print(f"\n  MuJoCo qvel[3:6] (body) = {omega_mj}")
print(f"  double rotation (错误) = {omega_double_rot}")
print(f"  V4 remap of double_rot = [{omega_double_rot[0]}, {omega_double_rot[2]}, {omega_double_rot[1]}]")
print(f"  V4 remap of correct    = [{omega_mj[0]}, {omega_mj[2]}, {omega_mj[1]}]")

# 训练中 IsaacLab 的 root_ang_vel_b 是怎么计算的？
# root_ang_vel_b = R.T @ root_ang_vel_w
# 如果 PhysX 返回的是世界坐标系角速度，那么 R.T @ ang_vel_w 是正确的
# 如果 MuJoCo 返回的是 body frame 角速度，那么直接用就行

# 但关键问题是：IsaacLab 的 root_ang_vel_b 和 MuJoCo 的 qvel[3:6] 是否一致？
# IsaacLab: root_ang_vel_b = R.T @ (PhysX world ang vel)
# MuJoCo: qvel[3:6] = body frame ang vel
# 如果 PhysX world ang vel = R @ body ang vel，那么：
# root_ang_vel_b = R.T @ R @ body_ang_vel = body_ang_vel
# 所以两者应该一致！

print("\n" + "="*80)
print("结论")
print("="*80)
print("MuJoCo qvel[3:6] = body frame 角速度")
print("IsaacLab root_ang_vel_b = R.T @ world_ang_vel = body frame 角速度")
print("两者一致，所以 MuJoCo 的 qvel[3:6] 应该直接使用，不需要 world_to_body()")
print("当前代码的 world_to_body(qvel[3:6]) 是 double rotation bug")
print()
print("但是！MuJoCo qvel[0:3] = world frame 线速度")
print("IsaacLab root_lin_vel_b = R.T @ world_lin_vel (但用的是 CoM 速度)")
print("所以 world_to_body(qvel[0:3]) 是正确的（忽略 CoM 差异）")
