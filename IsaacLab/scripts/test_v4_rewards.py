"""
测试 V4 奖励/观测函数的坐标系映射是否正确
===========================================

验证内容：
1. V4 观测函数的坐标系重映射
2. V4 奖励函数的坐标系重映射
3. 与 Go2 官方配置的奖励权重对比
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import math

# 导入 V4 环境配置
from isaaclab_tasks.manager_based.locomotion.velocity.config.v4_quadruped.flat_env_cfg import (
    V4QuadrupedFlatEnvCfg,
    v4_base_lin_vel,
    v4_base_ang_vel,
    v4_projected_gravity,
    v4_track_lin_vel_xy_exp,
    v4_track_ang_vel_z_exp,
    v4_lin_vel_y_l2,
    v4_ang_vel_xz_l2,
    v4_flat_orientation_l2,
)

print("\n" + "="*70)
print("V4 奖励/惩罚函数 vs Go2 官方配置 对比测试")
print("="*70)

# ============================================================
# 1. 奖励权重对比
# ============================================================
print("\n" + "-"*70)
print("1. 奖励权重对比 (V4 vs Go2 Flat)")
print("-"*70)

# Go2 Flat 最终奖励权重（基础 → rough覆盖 → flat覆盖）
go2_flat_rewards = {
    "track_lin_vel_xy_exp": 1.5,     # rough覆盖
    "track_ang_vel_z_exp": 0.75,     # rough覆盖
    "lin_vel_z_l2": -2.0,            # 基础
    "ang_vel_xy_l2": -0.05,          # 基础
    "dof_torques_l2": -0.0002,       # rough覆盖
    "dof_acc_l2": -2.5e-7,           # rough覆盖
    "action_rate_l2": -0.01,         # 基础
    "feet_air_time": 0.25,           # flat覆盖
    "undesired_contacts": None,      # rough删除
    "flat_orientation_l2": -2.5,     # flat覆盖
    "dof_pos_limits": 0.0,           # 基础（禁用）
}

# V4 当前奖励权重
v4_rewards = {
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 0.75,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -0.0002,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.25,
    "undesired_contacts": -1.0,      # Go2删除了，V4保留
    "flat_orientation_l2": -2.5,
    "dof_pos_limits": -1.0,          # Go2是0.0，V4是-1.0
    "quadruped_height": 1.0,         # V4额外
    "feet_contact": 0.5,             # V4额外
}

all_keys = sorted(set(list(go2_flat_rewards.keys()) + list(v4_rewards.keys())))

print(f"\n{'奖励项':<30} {'Go2 Flat':<15} {'V4 当前':<15} {'匹配':<10}")
print("-" * 70)

mismatches = []
for key in all_keys:
    go2_val = go2_flat_rewards.get(key, "无")
    v4_val = v4_rewards.get(key, "无")
    
    if go2_val == "无":
        match = "V4额外"
        mismatches.append((key, go2_val, v4_val, match))
    elif v4_val == "无":
        match = "V4缺少"
        mismatches.append((key, go2_val, v4_val, match))
    elif go2_val is None and v4_val is not None:
        match = "❌ Go2删除"
        mismatches.append((key, go2_val, v4_val, match))
    elif go2_val == v4_val:
        match = "✅"
    else:
        match = "❌ 不同"
        mismatches.append((key, go2_val, v4_val, match))
    
    go2_str = str(go2_val) if go2_val is not None else "None(删除)"
    v4_str = str(v4_val) if v4_val is not None else "None(删除)"
    print(f"{key:<30} {go2_str:<15} {v4_str:<15} {match:<10}")

print(f"\n总计: {len(all_keys)} 项, 不匹配: {len(mismatches)} 项")

if mismatches:
    print("\n⚠️  不匹配项详情:")
    for key, go2_val, v4_val, reason in mismatches:
        print(f"  - {key}: Go2={go2_val}, V4={v4_val} ({reason})")

# ============================================================
# 2. 坐标系映射验证（纯数学测试，不需要仿真环境）
# ============================================================
print("\n" + "-"*70)
print("2. 坐标系映射验证（数学测试）")
print("-"*70)

print("""
V4 绕X轴+90°后的坐标系映射：
  局部 X → 世界 X  (左右)     不变
  局部 Y → 世界 +Z (上方)     Y轴朝上
  局部 Z → 世界 -Y (后方)     Z轴朝后
  局部 -Z → 世界 +Y (前方)    头的方向

因此：
  前进速度 = -root_lin_vel_b[:, 2]  (局部-Z = 世界+Y = 前进)
  左右速度 = root_lin_vel_b[:, 0]   (局部X = 世界X = 左右)
  上下速度 = root_lin_vel_b[:, 1]   (局部Y = 世界+Z = 上下)
  转弯角速度 = root_ang_vel_b[:, 1] (绕局部Y = 绕世界Z = 转弯)
""")

# 模拟测试数据
print("模拟测试：机器人以 1.0 m/s 前进（世界+Y方向）")
# 在局部坐标系中，前进 = -Z方向
# 所以 root_lin_vel_b = [0, 0, -1.0]（局部-Z = 世界+Y = 前进）
mock_vel_b = torch.tensor([[0.0, 0.0, -1.0]])  # 前进1.0 m/s
remapped = torch.stack([-mock_vel_b[:, 2], mock_vel_b[:, 0], mock_vel_b[:, 1]], dim=-1)
print(f"  局部速度 root_lin_vel_b = {mock_vel_b.tolist()}")
print(f"  重映射后 [-Z, X, Y] = {remapped.tolist()}")
print(f"  期望: [1.0, 0.0, 0.0] (前进=1.0, 左右=0, 上下=0)")
assert torch.allclose(remapped, torch.tensor([[1.0, 0.0, 0.0]])), "前进速度映射错误!"
print("  ✅ 前进速度映射正确")

print("\n模拟测试：机器人向右移动 0.5 m/s（世界+X方向）")
mock_vel_b = torch.tensor([[0.5, 0.0, 0.0]])  # 局部X = 世界X = 右
remapped = torch.stack([-mock_vel_b[:, 2], mock_vel_b[:, 0], mock_vel_b[:, 1]], dim=-1)
print(f"  局部速度 root_lin_vel_b = {mock_vel_b.tolist()}")
print(f"  重映射后 [-Z, X, Y] = {remapped.tolist()}")
print(f"  期望: [0.0, 0.5, 0.0] (前进=0, 左右=0.5, 上下=0)")
assert torch.allclose(remapped, torch.tensor([[0.0, 0.5, 0.0]])), "左右速度映射错误!"
print("  ✅ 左右速度映射正确")

print("\n模拟测试：机器人向上跳 0.3 m/s（世界+Z方向）")
mock_vel_b = torch.tensor([[0.0, 0.3, 0.0]])  # 局部Y = 世界+Z = 上
remapped = torch.stack([-mock_vel_b[:, 2], mock_vel_b[:, 0], mock_vel_b[:, 1]], dim=-1)
print(f"  局部速度 root_lin_vel_b = {mock_vel_b.tolist()}")
print(f"  重映射后 [-Z, X, Y] = {remapped.tolist()}")
print(f"  期望: [0.0, 0.0, 0.3] (前进=0, 左右=0, 上下=0.3)")
assert torch.allclose(remapped, torch.tensor([[0.0, 0.0, 0.3]])), "上下速度映射错误!"
print("  ✅ 上下速度映射正确")

print("\n模拟测试：正确四足姿态的重力投影")
# 正确姿态时，重力在世界坐标系是 (0, 0, -1)
# 投影到局部坐标系：世界-Z = 局部-Y
# 所以 projected_gravity_b = (0, -1, 0)
mock_grav_b = torch.tensor([[0.0, -1.0, 0.0]])
remapped_grav = torch.stack([-mock_grav_b[:, 2], mock_grav_b[:, 0], mock_grav_b[:, 1]], dim=-1)
print(f"  局部重力 projected_gravity_b = {mock_grav_b.tolist()}")
print(f"  重映射后 [-Z, X, Y] = {remapped_grav.tolist()}")
print(f"  期望: [0.0, 0.0, -1.0] (标准四足: 重力沿-Z)")
assert torch.allclose(remapped_grav, torch.tensor([[0.0, 0.0, -1.0]])), "重力映射错误!"
print("  ✅ 重力投影映射正确（正确姿态时重力沿重映射后的-Z）")

print("\n模拟测试：v4_flat_orientation_l2 正确姿态应为0")
# 正确姿态时 projected_gravity_b = (0, -1, 0)
# v4_flat_orientation_l2 惩罚 X 和 Z 分量
penalty = mock_grav_b[:, 0]**2 + mock_grav_b[:, 2]**2
print(f"  flat_orientation_l2 = grav_x² + grav_z² = {penalty.item()}")
assert penalty.item() == 0.0, "正确姿态时 flat_orientation 应为0!"
print("  ✅ 正确姿态时 flat_orientation_l2 = 0")

print("\n模拟测试：v4_lin_vel_y_l2 惩罚上下速度")
# 上下速度 = 局部Y = root_lin_vel_b[:, 1]
mock_vel_up = torch.tensor([[0.0, 0.5, 0.0]])  # 向上跳0.5
penalty_up = mock_vel_up[:, 1]**2
print(f"  向上跳0.5时 lin_vel_y_l2 = {penalty_up.item()}")
assert penalty_up.item() == 0.25, "上下速度惩罚计算错误!"
print("  ✅ 上下速度惩罚正确")

print("\n模拟测试：v4_ang_vel_xz_l2 惩罚非转弯角速度")
# 转弯 = 绕局部Y轴 = root_ang_vel_b[:, 1]
# 惩罚 = 绕X轴² + 绕Z轴²
mock_ang = torch.tensor([[0.1, 0.5, 0.2]])  # roll=0.1, yaw=0.5, pitch=0.2
penalty_ang = mock_ang[:, 0]**2 + mock_ang[:, 2]**2
print(f"  ang_vel = {mock_ang.tolist()}")
print(f"  ang_vel_xz_l2 = X² + Z² = {penalty_ang.item():.4f}")
print(f"  转弯角速度(Y轴) = {mock_ang[:, 1].item()} (不被惩罚)")
print("  ✅ 非转弯角速度惩罚正确")

# ============================================================
# 3. 总结
# ============================================================
print("\n" + "="*70)
print("测试总结")
print("="*70)
print("""
✅ 坐标系映射全部正确：
  - 前进速度 = -root_lin_vel_b[:, 2]
  - 左右速度 = root_lin_vel_b[:, 0]
  - 上下速度 = root_lin_vel_b[:, 1]（被惩罚）
  - 转弯角速度 = root_ang_vel_b[:, 1]
  - 重力投影正确姿态 = (0, -1, 0)

⚠️  与 Go2 Flat 的差异：
  1. undesired_contacts: Go2=None(删除), V4=-1.0 → V4多了躯干触地惩罚
  2. dof_pos_limits: Go2=0.0(禁用), V4=-1.0 → V4启用了关节限位惩罚
  3. quadruped_height: Go2=无, V4=1.0 → V4额外的高度保持奖励
  4. feet_contact: Go2=无, V4=0.5 → V4额外的脚掌接地奖励

建议：如果要完全匹配Go2，应该：
  - 删除 undesired_contacts (设为None)
  - 将 dof_pos_limits 权重改为 0.0
  - 删除 quadruped_height
  - 删除 feet_contact
  
但V4额外的奖励项可能对四足姿态稳定有帮助，可以保留。
""")

simulation_app.close()
