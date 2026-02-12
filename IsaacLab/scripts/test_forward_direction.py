"""测试 V4 四足姿态下的前进方向。

通过检查前腿末端（手部）和后腿末端（脚部）的世界坐标位置，
确定哪个方向是前进方向。

前腿 = 手臂末端 (RARMAy, LARMAy)
后腿 = 脚部末端 (RANKLEy, LANKLEy)

前进方向 = 前腿位置 - 后腿位置（在世界 XY 平面上的投影）
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG
import isaaclab.utils.math as math_utils

# 创建仿真
sim_cfg = SimulationCfg(dt=0.005)
sim = SimulationContext(sim_cfg)

# 创建地面
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)

# 创建机器人
robot_cfg = V4_QUADRUPED_CFG.copy()
robot_cfg.prim_path = "/World/Robot"
robot = Articulation(robot_cfg)

# 设置相机
sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.2])

# 初始化
sim.reset()
robot.reset()

# 运行几步让机器人稳定
for _ in range(100):
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.cfg.dt)

# 获取各个 body 的世界坐标
body_names = robot.body_names
print("\n" + "="*60)
print("V4 四足姿态 - Body 世界坐标位置")
print("="*60)

# 找到关键 body 的索引
key_bodies = ["base_link", "RARMAy", "LARMAy", "RANKLEy", "LANKLEy", "Waist_2"]
for name in key_bodies:
    if name in body_names:
        idx = body_names.index(name)
        pos = robot.data.body_pos_w[0, idx]
        print(f"  {name:12s}: world pos = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# 计算前腿和后腿的平均位置
front_bodies = ["RARMAy", "LARMAy"]
rear_bodies = ["RANKLEy", "LANKLEy"]

front_pos = torch.zeros(3)
rear_pos = torch.zeros(3)
for name in front_bodies:
    if name in body_names:
        idx = body_names.index(name)
        front_pos += robot.data.body_pos_w[0, idx].cpu()
front_pos /= len(front_bodies)

for name in rear_bodies:
    if name in body_names:
        idx = body_names.index(name)
        rear_pos += robot.data.body_pos_w[0, idx].cpu()
rear_pos /= len(rear_bodies)

print(f"\n  前腿平均位置: ({front_pos[0]:.4f}, {front_pos[1]:.4f}, {front_pos[2]:.4f})")
print(f"  后腿平均位置: ({rear_pos[0]:.4f}, {rear_pos[1]:.4f}, {rear_pos[2]:.4f})")

forward_dir = front_pos - rear_pos
forward_dir[2] = 0  # 只看 XY 平面
forward_dir = forward_dir / torch.norm(forward_dir)
print(f"\n  前进方向（世界坐标 XY 平面）: ({forward_dir[0]:.4f}, {forward_dir[1]:.4f})")

if abs(forward_dir[0]) > abs(forward_dir[1]):
    if forward_dir[0] > 0:
        print("  → 前进方向 ≈ 世界 +X")
    else:
        print("  → 前进方向 ≈ 世界 -X")
else:
    if forward_dir[1] > 0:
        print("  → 前进方向 ≈ 世界 +Y")
    else:
        print("  → 前进方向 ≈ 世界 -Y")

# 检查局部坐标系
print("\n" + "="*60)
print("V4 局部坐标系映射")
print("="*60)

quat = robot.data.root_quat_w[0]
print(f"  root_quat_w: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")

# 局部各轴在世界坐标系中的方向
local_x = torch.tensor([[1.0, 0.0, 0.0]], device=robot.device)
local_y = torch.tensor([[0.0, 1.0, 0.0]], device=robot.device)
local_z = torch.tensor([[0.0, 0.0, 1.0]], device=robot.device)

world_x = math_utils.quat_apply(quat.unsqueeze(0), local_x)[0]
world_y = math_utils.quat_apply(quat.unsqueeze(0), local_y)[0]
world_z = math_utils.quat_apply(quat.unsqueeze(0), local_z)[0]

print(f"  局部 X (1,0,0) → 世界 ({world_x[0]:.4f}, {world_x[1]:.4f}, {world_x[2]:.4f})")
print(f"  局部 Y (0,1,0) → 世界 ({world_y[0]:.4f}, {world_y[1]:.4f}, {world_y[2]:.4f})")
print(f"  局部 Z (0,0,1) → 世界 ({world_z[0]:.4f}, {world_z[1]:.4f}, {world_z[2]:.4f})")
print(f"  局部-Z (0,0,-1)→ 世界 ({-world_z[0]:.4f}, {-world_z[1]:.4f}, {-world_z[2]:.4f})")

# 检查 root_lin_vel_b 的含义
print("\n" + "="*60)
print("速度映射验证")
print("="*60)
vel_b = robot.data.root_lin_vel_b[0]
print(f"  root_lin_vel_b: ({vel_b[0]:.4f}, {vel_b[1]:.4f}, {vel_b[2]:.4f})")
print(f"  vel_b[0] (局部X) = 世界 ({world_x[0]:.2f}, {world_x[1]:.2f}, {world_x[2]:.2f}) 方向的速度")
print(f"  vel_b[1] (局部Y) = 世界 ({world_y[0]:.2f}, {world_y[1]:.2f}, {world_y[2]:.2f}) 方向的速度")
print(f"  vel_b[2] (局部Z) = 世界 ({world_z[0]:.2f}, {world_z[1]:.2f}, {world_z[2]:.2f}) 方向的速度")

print(f"\n  如果前进方向 ≈ 世界 +Y:")
print(f"    前进速度 = vel_b 在世界+Y方向的分量")
print(f"    = vel_b[0]*{world_x[1]:.2f} + vel_b[1]*{world_y[1]:.2f} + vel_b[2]*{world_z[1]:.2f}")
if abs(world_z[1]) > 0.5:
    if world_z[1] > 0:
        print(f"    ≈ vel_b[2] (局部+Z = 前进)")
    else:
        print(f"    ≈ -vel_b[2] (局部-Z = 前进)")
elif abs(world_x[1]) > 0.5:
    print(f"    ≈ vel_b[0] (局部X = 前进)")

# 检查 projected_gravity
grav = robot.data.projected_gravity_b[0]
print(f"\n  projected_gravity_b: ({grav[0]:.4f}, {grav[1]:.4f}, {grav[2]:.4f})")
print(f"  正确四足姿态时应为 (0, -1, 0)（重力沿局部-Y=世界-Z）")

# heading_w
heading = robot.data.heading_w[0]
print(f"\n  heading_w: {heading:.4f} rad = {torch.rad2deg(heading):.1f}°")
print(f"  heading_w 基于 FORWARD_VEC_B=(1,0,0)（局部X轴）")
print(f"  局部X轴在世界XY平面的角度 = atan2({world_x[1]:.4f}, {world_x[0]:.4f}) = {torch.atan2(world_x[1], world_x[0]):.4f} rad")

print("\n" + "="*60)
print("结论")
print("="*60)
print(f"  前腿在世界 Y={front_pos[1]:.4f} 方向")
print(f"  后腿在世界 Y={rear_pos[1]:.4f} 方向")
if front_pos[1] > rear_pos[1]:
    print(f"  → 前腿在世界 +Y 方向 → 前进方向 = 世界 +Y")
    print(f"  → 局部 Z 映射到世界 ({world_z[0]:.2f}, {world_z[1]:.2f}, {world_z[2]:.2f})")
    if world_z[1] < -0.5:
        print(f"  → 局部 -Z = 世界 +Y = 前进 ✅ (当前代码 actual_forward = -vel_z 正确)")
    elif world_z[1] > 0.5:
        print(f"  → 局部 +Z = 世界 +Y = 前进 ⚠️ (当前代码 actual_forward = -vel_z 错误！应该是 +vel_z)")
else:
    print(f"  → 前腿在世界 -Y 方向 → 前进方向 = 世界 -Y")

simulation_app.close()
