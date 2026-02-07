#!/usr/bin/env python3
"""
V3机器人坐标系调试脚本

用途：确认V3机器人base_link的坐标系定义
- 打印不同姿态下的gravity_body值
- 验证X/Y/Z轴的方向定义
"""

import torch
import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="V3机器人坐标系调试")
# parser.add_argument("--num_envs", type=int, default=1, help="环境数量") # Unused in script
# parser.add_argument("--headless", action="store_true", help="无头模式") # AppLauncher adds this
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动模拟器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat
from isaaclab_assets.robots.v3_humanoid import V3_HUMANOID_CFG


def print_coordinate_system_info(robot: Articulation, pose_name: str):
    """打印当前姿态下的坐标系信息"""
    root_quat = robot.data.root_quat_w[0]  # 取第一个环境
    root_pos = robot.data.root_pos_w[0]
    
    # 计算重力在机器人坐标系下的分量
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device)
    gravity_body = quat_apply_inverse(root_quat.unsqueeze(0), gravity_world.unsqueeze(0))[0]
    
    # 计算欧拉角（Roll, Pitch, Yaw）
    euler = euler_xyz_from_quat(root_quat.unsqueeze(0))[0]
    roll, pitch, yaw = euler[0].item(), euler[1].item(), euler[2].item()
    
    print(f"\n{'='*60}")
    print(f"姿态: {pose_name}")
    print(f"{'='*60}")
    print(f"位置 (x, y, z): ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
    print(f"四元数 (w, x, y, z): ({root_quat[0]:.3f}, {root_quat[1]:.3f}, {root_quat[2]:.3f}, {root_quat[3]:.3f})")
    print(f"欧拉角 (Roll, Pitch, Yaw): ({roll:.3f}, {pitch:.3f}, {yaw:.3f}) rad")
    print(f"欧拉角 (度): ({roll*57.3:.1f}°, {pitch*57.3:.1f}°, {yaw*57.3:.1f}°)")
    print(f"\n重力在机器人坐标系下的分量:")
    print(f"  gravity_body.x = {gravity_body[0]:.4f}")
    print(f"  gravity_body.y = {gravity_body[1]:.4f}")
    print(f"  gravity_body.z = {gravity_body[2]:.4f}")
    print(f"\n坐标系解释:")
    if abs(gravity_body[2] + 1.0) < 0.1:
        print("  ✓ Z轴指向天花板（标准人形机器人坐标系）")
    elif abs(gravity_body[2]) < 0.1:
        print("  ✗ Z轴水平（非标准定义）")
    if abs(gravity_body[0]) < 0.1:
        print("  ✓ X轴水平（机器人前方）")
    elif gravity_body[0] < -0.5:
        print("  ✓ X轴前倾（机器人前方向下）")


def main():
    """主函数"""
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # 创建场景配置
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    
    # 添加地面 (手动生成，避免 InteractiveScene 配置错误)
    # scene_cfg.terrain = sim_utils.GroundPlaneCfg()
    
    # 添加V3机器人
    scene_cfg.robot = V3_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 创建场景
    scene = InteractiveScene(scene_cfg)

    # 手动添加地面
    gp_cfg = sim_utils.GroundPlaneCfg()
    gp_cfg.func("/World/ground", gp_cfg)
    
    print(f"DEBUG: Robot type: {type(scene['robot'])}")
    print(f"DEBUG: CFG class type: {V3_HUMANOID_CFG.class_type}")
    
    # Hack to bypass AttributeError if initialization failed partly
    if not hasattr(scene['robot'], '_actuators'):
        print("DEBUG: '_actuators' missing on robot instance. Init issue?")
        scene['robot']._actuators = {}
    
    # 获取模拟上下文
    sim = SimulationContext.instance()
    
    # 重置场景
    scene.reset()
    
    print("\n" + "="*60)
    print("V3机器人坐标系调试")
    print("="*60)
    
    # ============================================================
    # 测试1: 标准站立姿态
    # ============================================================
    print("\n[测试1] 标准站立姿态（所有关节为0）")
    scene.reset()
    for _ in range(10):
        sim.step()
    print_coordinate_system_info(scene["robot"], "标准站立")
    
    # ============================================================
    # 测试2: 前倾30度
    # ============================================================
    print("\n[测试2] 躯干前倾30度（模拟趴下姿态）")
    scene.reset()
    # 设置髋关节前倾
    joint_pos = scene["robot"].data.default_joint_pos.clone()
    joint_indices = scene["robot"].find_joints(".*HIPp")[0]
    joint_pos[:, joint_indices] = 0.52  # 约30度
    scene["robot"].write_joint_state_to_sim(joint_pos, scene["robot"].data.default_joint_vel)
    for _ in range(50):
        sim.step()
    print_coordinate_system_info(scene["robot"], "前倾30度")
    
    # ============================================================
    # 测试3: 侧翻90度（测试Y轴）
    # ============================================================
    print("\n[测试3] 侧翻90度（测试Roll轴）")
    scene.reset()
    # 直接设置root姿态
    root_state = scene["robot"].data.default_root_state.clone()
    # 绕X轴旋转90度: quat = [cos(45°), sin(45°), 0, 0]
    root_state[:, 3:7] = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # w, x, y, z
    scene["robot"].write_root_state_to_sim(root_state)
    for _ in range(10):
        sim.step()
    print_coordinate_system_info(scene["robot"], "侧翻90度")
    
    # ============================================================
    # 测试4: 前翻90度（测试X轴）
    # ============================================================
    print("\n[测试4] 前翻90度（测试Pitch轴）")
    scene.reset()
    root_state = scene["robot"].data.default_root_state.clone()
    # 绕Y轴旋转90度: quat = [cos(45°), 0, sin(45°), 0]
    root_state[:, 3:7] = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # w, x, y, z
    scene["robot"].write_root_state_to_sim(root_state)
    for _ in range(10):
        sim.step()
    print_coordinate_system_info(scene["robot"], "前翻90度")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*60)
    print("坐标系定义总结")
    print("="*60)
    print("根据上述测试结果，你可以确认：")
    print("1. 如果'标准站立'时 gravity_body.z ≈ -1，说明Z轴指向天花板")
    print("2. 如果'前倾30度'时 gravity_body.x < 0，说明X轴指向前方")
    print("3. 如果'侧翻90度'时 gravity_body.y ≈ ±1，说明Y轴指向侧方")
    print("\n根据这些信息，你可以调整 orientation_prone_measure() 函数")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
