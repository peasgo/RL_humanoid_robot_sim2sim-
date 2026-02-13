"""
测试 V4 四足机器人姿态配置
==========================
验证 base_link Y轴朝下（趴着）的四元数配置是否正确。
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
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


def main():
    # 有重力仿真，测试落下效果
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # 直接内联配置，避免导入问题
    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rl/RL-human_robot/V4.SLDASM/urdf/V4_new.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.21),
            # 绕X轴+90°: Y轴朝下（面部朝地），Z轴朝前
            rot=(0.7071, 0.7071, 0.0, 0.0),
            joint_pos={
                "Waist_2": 3.14159,
                "RSDp": 0.7854, "RSDy": 0.0, "RARMp": -1.5708, "RARMAP": 0.7854,
                "LSDp": 0.7854, "LSDy": 0.0, "LARMp": 1.5708, "LARMAp": 0.7854,
                "RHIPp": 0.7854, "RHIPy": 0.0, "RKNEEP": 1.5708, "RANKLEp": -0.7854,
                "LHIPp": -0.7854, "LHIPy": 0.0, "LKNEEp": -1.5708, "LANKLEp": -0.7854,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["Waist_2"], effort_limit_sim=400.0, stiffness=400.0, damping=40.0,
            ),
            "front_legs_shoulder": ImplicitActuatorCfg(
                joint_names_expr=[".*SD.*"], effort_limit_sim=200.0, stiffness=200.0, damping=20.0,
            ),
            "front_legs_elbow": ImplicitActuatorCfg(
                joint_names_expr=[".*ARM.*"], effort_limit_sim=200.0, stiffness=200.0, damping=20.0,
            ),
            "rear_legs_hip": ImplicitActuatorCfg(
                joint_names_expr=[".*HIP.*"], effort_limit_sim=400.0,
                stiffness={".*HIPp": 400.0, ".*HIPy": 200.0},
                damping={".*HIPp": 40.0, ".*HIPy": 20.0},
            ),
            "rear_legs_knee": ImplicitActuatorCfg(
                joint_names_expr=[".*KNEE.*"], effort_limit_sim=400.0, stiffness=400.0, damping=40.0,
            ),
            "rear_legs_ankle": ImplicitActuatorCfg(
                joint_names_expr=[".*ANKLE.*"], effort_limit_sim=200.0, stiffness=200.0, damping=20.0,
            ),
        },
    )

    robot_cfg_prim = robot_cfg.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg_prim)
    sim.reset()

    print(f"\n关节名: {robot.data.joint_names}")
    print(f"Body名: {robot.data.body_names}")

    # 默认关节角
    print("\n默认关节角度:")
    for i, name in enumerate(robot.data.joint_names):
        val = robot.data.default_joint_pos[0, i].item()
        print(f"  {name}: {val:.4f} rad ({math.degrees(val):.1f}°)")

    # 设置初始状态
    root_state = robot.data.default_root_state.clone()
    robot.write_root_state_to_sim(root_state)
    target_pos = robot.data.default_joint_pos.clone()
    target_vel = torch.zeros_like(target_pos)
    robot.write_joint_state_to_sim(target_pos, target_vel)
    robot.set_joint_position_target(target_pos)
    robot.write_data_to_sim()

    # 步进200步（1秒），让机器人落下并稳定
    print("\n模拟落下中（200步 = 1秒）...")
    for step in range(200):
        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)

        if step % 50 == 0 or step == 199:
            body_names = robot.data.body_names
            body_pos = robot.data.body_pos_w[0]
            base_idx = body_names.index('base_link')
            base_pos = body_pos[base_idx].cpu().numpy()
            base_z = base_pos[2]

            # 获取root四元数
            root_quat = robot.data.root_quat_w[0].cpu().numpy()

            print(f"\n  Step {step}: base_link pos=({base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f})")
            print(f"    root_quat(w,x,y,z) = ({root_quat[0]:.4f}, {root_quat[1]:.4f}, {root_quat[2]:.4f}, {root_quat[3]:.4f})")

    # 最终状态分析
    print("\n" + "=" * 60)
    print("最终状态分析")
    print("=" * 60)

    body_names = robot.data.body_names
    body_pos = robot.data.body_pos_w[0]

    key_bodies = ['base_link', 'Waist', 'Waist_2', 'RARMAy', 'LARMAy', 'RANKLEy', 'LANKLEy']
    print(f"\n  {'Body':<20} {'X':>8} {'Y':>8} {'Z':>8}")
    print(f"  {'-'*46}")
    for name in key_bodies:
        if name in body_names:
            idx = body_names.index(name)
            p = body_pos[idx].cpu().numpy()
            print(f"  {name:<20} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f}")

    base_idx = body_names.index('base_link')
    base_z = body_pos[base_idx, 2].item()

    feet = ['RARMAy', 'LARMAy', 'RANKLEy', 'LANKLEy']
    print(f"\n  base_link Z = {base_z:.4f}")
    min_foot_z = 999
    for name in feet:
        if name in body_names:
            idx = body_names.index(name)
            z = body_pos[idx, 2].item()
            status = "低于base ✅" if z < base_z else "高于base ❌"
            print(f"  {name} Z = {z:.4f} ({status})")
            min_foot_z = min(min_foot_z, z)

    all_below = all(
        body_pos[body_names.index(n), 2].item() < base_z
        for n in feet if n in body_names
    )

    print(f"\n  {'✅ 四脚朝下，姿态正确！' if all_below else '❌ 有脚高于base，姿态可能不对'}")
    print(f"  base到地面距离 = {base_z:.4f} m")
    print(f"  末端最低Z = {min_foot_z:.4f} m")

    # 分析重力投影
    root_quat = robot.data.root_quat_w[0].cpu()
    print(f"\n  最终root四元数(w,x,y,z) = ({root_quat[0]:.4f}, {root_quat[1]:.4f}, {root_quat[2]:.4f}, {root_quat[3]:.4f})")

    print("\n诊断完成！")


if __name__ == "__main__":
    main()
    simulation_app.close()
