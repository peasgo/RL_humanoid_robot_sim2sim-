"""
V4 四足机器人姿态诊断脚本
========================
测试不同四元数下机器人的姿态，找到正确的"四脚朝下"配置。

关键问题：
1. URDF中机器人直立时，哪个轴朝上？（SolidWorks导出通常Y朝上）
2. 绕哪个轴旋转多少度才能让机器人变成四足姿态？
3. 关节角度配合四元数后，末端效应器是否朝下？
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG


def test_pose(robot, sim, label, root_pos, root_quat, joint_pos_dict=None, steps=50):
    """测试一个特定姿态"""
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"  root_pos = {root_pos}")
    print(f"  root_quat(w,x,y,z) = {root_quat}")
    print(f"{'='*60}")

    # 设置根状态
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = torch.tensor(root_pos, device=robot.device)
    root_state[:, 3:7] = torch.tensor(root_quat, device=robot.device)
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)

    # 设置关节
    if joint_pos_dict is not None:
        target_pos = torch.zeros_like(robot.data.default_joint_pos)
        for name, val in joint_pos_dict.items():
            if name in robot.data.joint_names:
                idx = robot.data.joint_names.index(name)
                target_pos[0, idx] = val
    else:
        target_pos = robot.data.default_joint_pos.clone()

    target_vel = torch.zeros_like(target_pos)
    robot.write_joint_state_to_sim(target_pos, target_vel)
    robot.set_joint_position_target(target_pos)
    robot.write_data_to_sim()

    # 步进
    for _ in range(steps):
        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)

    # 读取结果
    body_names = robot.data.body_names
    body_pos = robot.data.body_pos_w[0]

    # 打印所有body
    print(f"\n  {'Body':<20} {'X':>8} {'Y':>8} {'Z':>8}")
    print(f"  {'-'*46}")
    for i, name in enumerate(body_names):
        p = body_pos[i].cpu().numpy()
        marker = ""
        if name in ['RARMAy', 'LARMAy', 'RANKLEy', 'LANKLEy']:
            marker = " ← 末端"
        if name == 'base_link':
            marker = " ← 根"
        print(f"  {name:<20} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f}{marker}")

    # 末端效应器分析
    feet = ['RARMAy', 'LARMAy', 'RANKLEy', 'LANKLEy']
    feet_z = []
    for name in feet:
        if name in body_names:
            idx = body_names.index(name)
            feet_z.append((name, body_pos[idx, 2].item()))

    base_idx = body_names.index('base_link')
    base_z = body_pos[base_idx, 2].item()

    print(f"\n  base_link Z = {base_z:.4f}")
    for name, z in feet_z:
        print(f"  {name} Z = {z:.4f} ({'低于base' if z < base_z else '高于base ⚠️'})")

    all_below = all(z < base_z for _, z in feet_z)
    min_foot_z = min(z for _, z in feet_z) if feet_z else 999
    max_foot_z = max(z for _, z in feet_z) if feet_z else -999

    print(f"\n  ✅ 末端全部低于base" if all_below else f"\n  ❌ 有末端高于base（四脚朝天！）")
    print(f"  末端最低Z = {min_foot_z:.4f}, 最高Z = {max_foot_z:.4f}")
    print(f"  建议初始高度 = {base_z - min_foot_z + 0.02:.4f} m (末端离地2cm)")

    return all_below, base_z - min_foot_z


def main():
    # 无重力仿真，纯看姿态
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0", gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    robot_cfg = V4_QUADRUPED_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()

    # 默认关节角度
    default_joints = {}
    for i, name in enumerate(robot.data.joint_names):
        default_joints[name] = robot.data.default_joint_pos[0, i].item()

    print("\n默认关节角度:")
    for name, val in default_joints.items():
        print(f"  {name}: {val:.4f} rad ({math.degrees(val):.1f}°)")

    # ============================================================
    # 测试1: 当前配置 rot=(0.7071, -0.7071, 0, 0) 绕X轴-90°
    # ============================================================
    test_pose(robot, sim,
              "当前配置: 绕X轴-90°",
              [0, 0, 0.5],
              [0.7071, -0.7071, 0.0, 0.0])

    # ============================================================
    # 测试2: 绕X轴+90°
    # ============================================================
    test_pose(robot, sim,
              "绕X轴+90°",
              [0, 0, 0.5],
              [0.7071, 0.7071, 0.0, 0.0])

    # ============================================================
    # 测试3: 绕Y轴-90°
    # ============================================================
    test_pose(robot, sim,
              "绕Y轴-90°",
              [0, 0, 0.5],
              [0.7071, 0.0, -0.7071, 0.0])

    # ============================================================
    # 测试4: 绕Y轴+90°
    # ============================================================
    test_pose(robot, sim,
              "绕Y轴+90°",
              [0, 0, 0.5],
              [0.7071, 0.0, 0.7071, 0.0])

    # ============================================================
    # 测试5: 无旋转（直立）
    # ============================================================
    test_pose(robot, sim,
              "无旋转（直立）",
              [0, 0, 0.5],
              [1.0, 0.0, 0.0, 0.0])

    # ============================================================
    # 测试6: 绕X轴-90° + 关节全零（看URDF原始朝向）
    # ============================================================
    zero_joints = {name: 0.0 for name in robot.data.joint_names}
    test_pose(robot, sim,
              "绕X轴-90° + 关节全零",
              [0, 0, 0.5],
              [0.7071, -0.7071, 0.0, 0.0],
              joint_pos_dict=zero_joints)

    # ============================================================
    # 测试7: 无旋转 + 关节全零（URDF原始姿态）
    # ============================================================
    test_pose(robot, sim,
              "无旋转 + 关节全零（URDF原始）",
              [0, 0, 0.5],
              [1.0, 0.0, 0.0, 0.0],
              joint_pos_dict=zero_joints)

    print("\n" + "=" * 60)
    print("诊断完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()
