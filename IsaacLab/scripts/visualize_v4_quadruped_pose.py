"""
V4 四足机器人初始姿态可视化脚本
================================

用于检查四足初始关节角度是否正确。
关键：必须在每次 sim.step() 前设置 PD 控制器目标位置，
否则 ImplicitActuator 的目标默认为0，会把关节拉回零位。

使用方法:
    cd IsaacLab && conda run -n isaaclab python scripts/visualize_v4_quadruped_pose.py
    cd IsaacLab && conda run -n isaaclab python scripts/visualize_v4_quadruped_pose.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="V4 四足机器人初始姿态可视化")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 以下代码在 Isaac Sim 启动后执行 ----

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG


# 四足初始关节角度（DOG_V5，无踝关节）
QUADRUPED_JOINT_ANGLES = {
    "Waist_2": 3.14159,  # 上半身完全前倾 ≈π
            # 前腿（手臂）
    "RSDp": 0.6,
    "RSDy": 0.0,
    "RARMp": -1.4,
    "LSDp": 0.6,
    "LSDy": 0.0,
    "LARMp": 1.4,
            # 后腿
    "RHIPp": 0.78,
    "RHIPy": 0.0,
    "RKNEEP": 1.0,
    "LHIPp": -0.78,
    "LHIPy": 0.0,
    "LKNEEp": -1.0,              # 左膝 ~-57°
}


def main():
    """主函数：加载机器人并可视化初始姿态"""

    # 仿真配置
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.2])

    # 创建地面
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # 创建灯光
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    # 加载机器人
    robot_cfg = V4_QUADRUPED_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # 初始化仿真
    sim.reset()

    # 打印关节名称
    joint_names = robot.data.joint_names
    print("\n" + "=" * 60)
    print("V4 四足机器人 - 关节信息")
    print("=" * 60)
    print(f"关节数量: {len(joint_names)}")
    print(f"关节名称: {joint_names}")

    # ============================================================
    # 第一步：检查 default_joint_pos 是否正确设置
    # ============================================================
    print("\n" + "=" * 60)
    print("检查 default_joint_pos（来自 init_state.joint_pos 配置）")
    print("=" * 60)

    default_pos = robot.data.default_joint_pos[0]
    print(f"\n{'关节名称':<20} {'default_pos(rad)':<18} {'default_pos(deg)':<18} {'目标(deg)':<15}")
    print("-" * 70)
    all_match = True
    for i, name in enumerate(joint_names):
        default_rad = default_pos[i].item()
        default_deg = math.degrees(default_rad)
        target_deg = math.degrees(QUADRUPED_JOINT_ANGLES.get(name, 0.0))
        match = "✓" if abs(default_deg - target_deg) < 1.0 else "✗ MISMATCH"
        if abs(default_deg - target_deg) >= 1.0:
            all_match = False
        print(f"{name:<20} {default_rad:<18.4f} {default_deg:<18.2f} {target_deg:<15.2f} {match}")

    if all_match:
        print("\n✅ 所有 default_joint_pos 与目标角度匹配！")
    else:
        print("\n❌ 部分 default_joint_pos 与目标角度不匹配！")

    # ============================================================
    # 第二步：设置关节位置并写入仿真
    # ============================================================
    target_pos = robot.data.default_joint_pos.clone()
    target_vel = torch.zeros_like(target_pos)

    # 写入关节状态（位置+速度）
    robot.write_joint_state_to_sim(target_pos, target_vel)

    # 设置根状态（使用配置中的默认值，包括 rot=(0.7071, -0.7071, 0, 0)）
    root_state = robot.data.default_root_state.clone()
    root_state[:, 7:] = 0.0  # 零速度
    robot.write_root_state_to_sim(root_state)

    # ★★★ 关键：设置 PD 控制器的目标位置 ★★★
    # ImplicitActuator 使用 PhysX 内置的 PD 控制器
    # 如果不设置目标位置，PD 控制器目标默认为0，会把关节拉回零位！
    robot.set_joint_position_target(target_pos)
    robot.set_joint_velocity_target(target_vel)
    robot.write_data_to_sim()

    # 步进几步让物理稳定
    for i in range(20):
        # 每步都要重新设置 PD 目标并写入
        robot.set_joint_position_target(target_pos)
        robot.set_joint_velocity_target(target_vel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)

    # ============================================================
    # 第三步：读取稳定后的关节状态
    # ============================================================
    print("\n" + "=" * 60)
    print("设置关节角度 + PD目标后的状态（20步物理仿真后）")
    print("=" * 60)

    joint_pos_actual = robot.data.joint_pos[0]
    print(f"\n{'关节名称':<20} {'实际角度(rad)':<15} {'实际角度(deg)':<15} {'目标角度(deg)':<15} {'误差(deg)':<12}")
    print("-" * 75)
    for i, name in enumerate(joint_names):
        actual_rad = joint_pos_actual[i].item()
        actual_deg = math.degrees(actual_rad)
        target_deg = math.degrees(QUADRUPED_JOINT_ANGLES.get(name, 0.0))
        error_deg = actual_deg - target_deg
        status = "✓" if abs(error_deg) < 5.0 else "✗"
        print(f"{name:<20} {actual_rad:<15.4f} {actual_deg:<15.2f} {target_deg:<15.2f} {error_deg:<12.2f} {status}")

    print(f"\n根位置 (x, y, z): {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"根旋转 (w, x, y, z): {robot.data.root_quat_w[0].cpu().numpy()}")

    # 打印body位置
    body_names = robot.data.body_names
    body_pos = robot.data.body_pos_w[0]

    print(f"\n{'Body名称':<20} {'X(m)':<10} {'Y(m)':<10} {'Z(m)':<10}")
    print("-" * 50)
    for i, name in enumerate(body_names):
        pos = body_pos[i].cpu().numpy()
        print(f"{name:<20} {pos[0]:<10.4f} {pos[1]:<10.4f} {pos[2]:<10.4f}")

    # 末端效应器
    print("\n" + "=" * 60)
    print("末端效应器位置（四足接地点）:")
    print("=" * 60)
    end_effectors = ["R_ARM_feet", "L_arm_feet", "R_Feet", "L_Feet"]
    for name in end_effectors:
        if name in body_names:
            idx = body_names.index(name)
            pos = body_pos[idx].cpu().numpy()
            label = {
                "R_ARM_feet": "前右腿（右手）",
                "L_arm_feet": "前左腿（左手）",
                "R_Feet": "后右腿（右脚）",
                "L_Feet": "后左腿（左脚）",
            }.get(name, name)
            print(f"  {label}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m")

    print(f"\n质心高度: {robot.data.root_pos_w[0, 2].item():.4f} m")

    # ============================================================
    # 第四步：持续仿真，观察稳定性
    # ============================================================
    print("\n" + "=" * 60)
    print("开始持续物理仿真（PD控制器持续维持目标姿态）...")
    print("按 Ctrl+C 退出。")
    print("=" * 60)

    count = 0
    while simulation_app.is_running():
        # ★★★ 每步都设置 PD 目标 ★★★
        robot.set_joint_position_target(target_pos)
        robot.set_joint_velocity_target(target_vel)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_cfg.dt)
        count += 1

        # 每500步打印一次状态
        if count % 500 == 0:
            height = robot.data.root_pos_w[0, 2].item()
            print(f"\n[Step {count}] 质心高度: {height:.4f} m")

            # 打印关节角度
            for i, name in enumerate(joint_names):
                actual_rad = robot.data.joint_pos[0, i].item()
                target_deg = math.degrees(QUADRUPED_JOINT_ANGLES.get(name, 0.0))
                actual_deg = math.degrees(actual_rad)
                print(f"  {name:<15} 实际: {actual_deg:>8.1f}°  目标: {target_deg:>8.1f}°  误差: {actual_deg-target_deg:>8.1f}°")


if __name__ == "__main__":
    main()
    simulation_app.close()
