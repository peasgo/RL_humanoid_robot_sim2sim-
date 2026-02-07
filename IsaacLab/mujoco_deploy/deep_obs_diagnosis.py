"""深度诊断：观测空间构造差异"""
import sys
sys.path.insert(0, '/home/rl/RL-human_robot/IsaacLab')

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

POLICY_PATH = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-28_10-29-50/exported/policy.pt"
XML_PATH = "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy/WholeAssembleV2_mujoco.xml"

ACTION_JOINT_ORDER = [
    "RHipP", "RHipY", "RHipR", "RKneeP",
    "LHipP", "LHipY", "LHipR", "LKneeP",
    "RAankleP", "RAnkleR", "LAnkleP", "LAnkleR",
]

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = torch.jit.load(POLICY_PATH, map_location=device)
policy.eval()

joint_qpos_adr = []
joint_qvel_adr = []
for name in ACTION_JOINT_ORDER:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_qpos_adr.append(model.jnt_qposadr[jid])
    joint_qvel_adr.append(model.jnt_dofadr[jid])

default_joint_pos = np.zeros(len(ACTION_JOINT_ORDER))
for i, adr in enumerate(joint_qpos_adr):
    default_joint_pos[i] = model.qpos0[adr]

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

print("="*70)
print("深度诊断：观测空间逐项检查")
print("="*70)

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.35
mujoco.mj_forward(model, data)

# 详细检查每个观测分量
quat = data.qpos[3:7]
print(f"\n1. Base四元数: {quat}")
print(f"   是否单位四元数: {np.allclose(np.linalg.norm(quat), 1.0)}")

r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
base_rot_mat = r.as_matrix()
print(f"\n2. 旋转矩阵:\n{base_rot_mat}")

vel_global = np.zeros(6)
mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel_global, 0)
print(f"\n3. 全局速度: {vel_global}")

ang_vel_local = base_rot_mat.T @ vel_global[0:3]
lin_vel_local = base_rot_mat.T @ vel_global[3:6]
print(f"   局部角速度: {ang_vel_local}")
print(f"   局部线速度: {lin_vel_local}")

gravity_global = np.array([0.0, 0.0, -1.0])
gravity_local = base_rot_mat.T @ gravity_global
print(f"\n4. 重力向量:")
print(f"   全局: {gravity_global}")
print(f"   局部: {gravity_local}")
print(f"   是否接近[0,0,-1]: {np.allclose(gravity_local, [0,0,-1], atol=1e-6)}")

command = np.zeros(3)
print(f"\n5. 命令: {command}")

qpos = data.qpos[joint_qpos_adr].flatten()
qvel = data.qvel[joint_qvel_adr].flatten()
print(f"\n6. 关节位置: {qpos}")
print(f"   默认位置: {default_joint_pos}")
print(f"   位置偏移: {qpos - default_joint_pos}")
print(f"   是否全零: {np.allclose(qpos - default_joint_pos, 0, atol=1e-6)}")

print(f"\n7. 关节速度: {qvel}")
print(f"   是否全零: {np.allclose(qvel, 0, atol=1e-6)}")

last_action = np.zeros(12)
print(f"\n8. 上一步动作: {last_action}")

# 组装观测
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
print("完整观测向量分析")
print("="*70)
print(f"观测维度: {obs.shape}")
print(f"观测范围: [{obs.min():.6f}, {obs.max():.6f}]")
print(f"非零元素数量: {np.count_nonzero(obs)}")
print(f"非零元素索引: {np.where(np.abs(obs) > 1e-6)[0]}")
print(f"\n非零元素详情:")
for idx in np.where(np.abs(obs) > 1e-6)[0]:
    component = ""
    if idx < 3:
        component = f"lin_vel[{idx}]"
    elif idx < 6:
        component = f"ang_vel[{idx-3}]"
    elif idx < 9:
        component = f"gravity[{idx-6}]"
    elif idx < 12:
        component = f"command[{idx-9}]"
    elif idx < 24:
        component = f"joint_pos[{idx-12}] ({ACTION_JOINT_ORDER[idx-12]})"
    elif idx < 36:
        component = f"joint_vel[{idx-24}] ({ACTION_JOINT_ORDER[idx-24]})"
    else:
        component = f"last_action[{idx-36}] ({ACTION_JOINT_ORDER[idx-36]})"
    print(f"  [{idx:2d}] {component:30s} = {obs[idx]:.6f}")

# 测试策略
obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
with torch.no_grad():
    action_raw = policy(obs_tensor.to(device)).cpu().numpy().squeeze()

# IsaacLab (RSL-RL wrapper) clips actions before stepping the env.
action = np.clip(action_raw, -1.0, 1.0)

print(f"\n" + "="*70)
print("策略输出分析")
print("="*70)
print(f"动作 raw 范围:    [{action_raw.min():.3f}, {action_raw.max():.3f}]")
print(f"动作 clipped 范围: [{action.min():.3f}, {action.max():.3f}]")
print(f"动作均值: {action_raw.mean():.3f}")
print(f"动作标准差: {action_raw.std():.3f}")
print(f"\n各关节动作:")
for i, name in enumerate(ACTION_JOINT_ORDER):
    marker = " ← 异常" if abs(action_raw[i]) > 1.0 else ""
    print(f"  {name:12s}: raw={action_raw[i]:7.3f}  act={action[i]:7.3f}{marker}")

print(f"\n" + "="*70)
print("核心发现")
print("="*70)
if np.allclose(obs, 0, atol=1e-6):
    print("✓ 观测值完全为零")
else:
    print("✗ 观测值不为零 - 这是问题所在！")
    print("  即使在初始状态，观测空间也有非零值")
    print("  这可能导致策略输出异常动作")
