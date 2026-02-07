"""
详细观测值调试脚本
用于检查每个观测分量是否合理
"""
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import mujoco

# 配置
POLICY_PATH = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-28_10-29-50/exported/policy.pt"
XML_PATH = "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy/WholeAssembleV2_mujoco.xml"

ACTION_JOINT_ORDER = [
    "RHipP", "RHipY", "RHipR", "RKneeP",
    "LHipP", "LHipY", "LHipR", "LKneeP",
    "RAankleP", "RAnkleR",
    "LAnkleP", "LAnkleR",
]

print("加载模型...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("加载策略...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = torch.jit.load(POLICY_PATH, map_location=device)
policy.eval()

# 获取关节索引
joint_qpos_adr = []
joint_qvel_adr = []
for name in ACTION_JOINT_ORDER:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_qpos_adr.append(model.jnt_qposadr[jid])
    joint_qvel_adr.append(model.jnt_dofadr[jid])

# 获取默认关节位置
default_joint_pos = np.zeros(len(ACTION_JOINT_ORDER))
for i, adr in enumerate(joint_qpos_adr):
    default_joint_pos[i] = model.qpos0[adr]

print(f"默认关节位置: {default_joint_pos}")

# 获取base_id
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

# 重置到初始状态
mujoco.mj_resetData(model, data)
for i, adr in enumerate(joint_qpos_adr):
    data.qpos[adr] = default_joint_pos[i]
data.qpos[2] = 0.5  # 设置高度
mujoco.mj_forward(model, data)

print("\n" + "="*70)
print("详细观测值分析 (初始静止状态)")
print("="*70)

# 构造观测
# Base rotation
quat = data.qpos[3:7]
print(f"\n1. Base Quaternion (w,x,y,z): {quat}")
r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
base_rot_mat = r.as_matrix()
print(f"   Rotation Matrix:\n{base_rot_mat}")

# Velocities
vel_global = np.zeros(6)
mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel_global, 0)
ang_vel_global = vel_global[0:3]
lin_vel_global = vel_global[3:6]
ang_vel_local = base_rot_mat.T @ ang_vel_global
lin_vel_local = base_rot_mat.T @ lin_vel_global

print(f"\n2. Linear Velocity (global): {lin_vel_global}")
print(f"   Linear Velocity (local):  {lin_vel_local}")
print(f"   Angular Velocity (global): {ang_vel_global}")
print(f"   Angular Velocity (local):  {ang_vel_local}")

# Gravity
gravity_global = np.array([0.0, 0.0, -1.0])
gravity_local = base_rot_mat.T @ gravity_global
print(f"\n3. Gravity (global): {gravity_global}")
print(f"   Gravity (local):  {gravity_local}")
print(f"   Expected: [0, 0, -1] for upright robot")

# Commands
command = np.zeros(3)
print(f"\n4. Commands: {command}")

# Joint states
qpos = data.qpos[joint_qpos_adr].flatten()
qvel = data.qvel[joint_qvel_adr].flatten()
print(f"\n5. Joint Positions: {qpos}")
print(f"   Joint Position Offsets (qpos - default): {qpos - default_joint_pos}")
print(f"   Joint Velocities: {qvel}")

# Last action
last_action = np.zeros(12)
print(f"\n6. Last Action: {last_action}")

# 组装观测
obs_components = {
    "lin_vel": lin_vel_local,
    "ang_vel": ang_vel_local,
    "gravity": gravity_local,
    "commands": command,
    "joint_pos_offset": qpos - default_joint_pos,
    "joint_vel": qvel,
    "last_action": last_action
}

obs = np.concatenate([
    lin_vel_local,
    ang_vel_local,
    gravity_local,
    command,
    qpos - default_joint_pos,
    qvel,
    last_action
])

print(f"\n" + "="*70)
print("完整观测向量 (48维)")
print("="*70)
print(f"Observation shape: {obs.shape}")
print(f"Observation: {obs}")
print(f"\n各分量范围:")
print(f"  lin_vel:      [{lin_vel_local.min():.4f}, {lin_vel_local.max():.4f}]")
print(f"  ang_vel:      [{ang_vel_local.min():.4f}, {ang_vel_local.max():.4f}]")
print(f"  gravity:      [{gravity_local.min():.4f}, {gravity_local.max():.4f}]")
print(f"  commands:     [{command.min():.4f}, {command.max():.4f}]")
print(f"  joint_pos:    [{(qpos-default_joint_pos).min():.4f}, {(qpos-default_joint_pos).max():.4f}]")
print(f"  joint_vel:    [{qvel.min():.4f}, {qvel.max():.4f}]")
print(f"  last_action:  [{last_action.min():.4f}, {last_action.max():.4f}]")

# 测试策略
obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
with torch.no_grad():
    action_raw = policy(obs_tensor.to(device)).cpu().numpy().squeeze()

# IsaacLab (RSL-RL wrapper) clips actions before stepping the env.
action = np.clip(action_raw, -1.0, 1.0)

print(f"\n" + "="*70)
print("策略输出")
print("="*70)
print(f"Action raw: {action_raw}")
print(f"Action clipped: {action}")
print(f"Action raw range:     [{action_raw.min():.4f}, {action_raw.max():.4f}]")
print(f"Action clipped range: [{action.min():.4f}, {action.max():.4f}]")
print(f"Action raw mean/std: {action_raw.mean():.4f} / {action_raw.std():.4f}")

print(f"\n" + "="*70)
print("诊断建议")
print("="*70)

# 检查异常值
issues = []
if np.abs(gravity_local - np.array([0, 0, -1])).max() > 0.1:
    issues.append("⚠️  重力向量异常! 应该接近 [0, 0, -1]")

if np.abs(lin_vel_local).max() > 0.01:
    issues.append("⚠️  线速度非零! 静止状态应该接近0")

if np.abs(ang_vel_local).max() > 0.01:
    issues.append("⚠️  角速度非零! 静止状态应该接近0")

if np.abs(qpos - default_joint_pos).max() > 0.01:
    issues.append("⚠️  关节位置偏移非零! 初始状态应该在默认位置")

if np.abs(qvel).max() > 0.01:
    issues.append("⚠️  关节速度非零! 静止状态应该接近0")

if len(issues) == 0:
    print("✓ 所有观测值看起来正常")
    print("\n如果动作幅度仍然很大,可能的原因:")
    print("  1. 策略本身在零指令下不稳定")
    print("  2. 训练时使用了不同的观测缩放")
    print("  3. 需要进一步降低 ACTION_SCALE")
else:
    print("发现以下问题:")
    for issue in issues:
        print(f"  {issue}")
    print("\n这些问题可能导致策略输出异常动作!")
