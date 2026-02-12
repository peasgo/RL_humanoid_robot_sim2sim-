#!/usr/bin/env python3
"""诊断V4坐标系转换是否正确。

在V4初始姿态下验证：
1. get_gravity_orientation() 的输出
2. world_to_body() 的输出
3. v4_remap_*() 的输出
4. 与IsaacLab训练代码的对比
"""
import numpy as np

def get_gravity_orientation(quaternion):
    """Convert quaternion [w,x,y,z] to gravity vector in body frame."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

def quat_to_rotmat_wxyz(quat_wxyz):
    w, x, y, z = quat_wxyz
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return R

def world_to_body(v_world, quat_wxyz):
    R_wb = quat_to_rotmat_wxyz(quat_wxyz)
    return R_wb.T @ v_world

# V4 初始四元数 (绕X轴旋转90°)
quat_v4 = np.array([0.7071068, 0.7071068, 0.0, 0.0])

print("=" * 60)
print("V4 初始姿态诊断 (quat = [0.7071, 0.7071, 0, 0])")
print("=" * 60)

# 1. 旋转矩阵
R = quat_to_rotmat_wxyz(quat_v4)
print(f"\n旋转矩阵 R (world->body columns = body axes in world):")
print(f"  body X in world: [{R[0,0]:.3f}, {R[1,0]:.3f}, {R[2,0]:.3f}]")
print(f"  body Y in world: [{R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f}]")
print(f"  body Z in world: [{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}]")

# 2. 重力方向
grav = get_gravity_orientation(quat_v4)
print(f"\n重力方向 (body frame): [{grav[0]:.3f}, {grav[1]:.3f}, {grav[2]:.3f}]")
print(f"  解释: 重力在body frame中指向 body {['X' if abs(grav[0])>0.5 else ''][0] or ''}"
      f"{'Y' if abs(grav[1])>0.5 else ''}"
      f"{'Z' if abs(grav[2])>0.5 else ''} 方向")

# 3. 验证: R^T @ [0,0,-1] 应该等于 get_gravity_orientation
grav_check = R.T @ np.array([0, 0, -1])
print(f"  验证 R^T @ [0,0,-1]: [{grav_check[0]:.3f}, {grav_check[1]:.3f}, {grav_check[2]:.3f}]")
print(f"  一致性: {'✓' if np.allclose(grav, grav_check) else '✗ 不一致!'}")

# 4. 速度转换测试
print(f"\n--- 速度转换测试 ---")

# 假设机器人在世界坐标系中向前走 (V4前进方向 = 世界-Y)
v_forward_world = np.array([0, -1, 0])  # 世界-Y = V4前进
v_forward_body = world_to_body(v_forward_world, quat_v4)
print(f"\n世界-Y方向速度 (V4前进):")
print(f"  world: {v_forward_world}")
print(f"  body:  [{v_forward_body[0]:.3f}, {v_forward_body[1]:.3f}, {v_forward_body[2]:.3f}]")

# 假设机器人在世界坐标系中向上运动
v_up_world = np.array([0, 0, 1])  # 世界+Z = 上
v_up_body = world_to_body(v_up_world, quat_v4)
print(f"\n世界+Z方向速度 (向上):")
print(f"  world: {v_up_world}")
print(f"  body:  [{v_up_body[0]:.3f}, {v_up_body[1]:.3f}, {v_up_body[2]:.3f}]")

# 假设机器人在世界坐标系中向右运动
v_right_world = np.array([1, 0, 0])  # 世界+X = 右
v_right_body = world_to_body(v_right_world, quat_v4)
print(f"\n世界+X方向速度 (向右):")
print(f"  world: {v_right_world}")
print(f"  body:  [{v_right_body[0]:.3f}, {v_right_body[1]:.3f}, {v_right_body[2]:.3f}]")

# 5. V4 重映射
print(f"\n--- V4 重映射 ---")
print(f"IsaacLab训练代码中:")
print(f"  v4_base_lin_vel: [vel[:,2], vel[:,0], vel[:,1]] = [+Z, X, Y]")
print(f"  v4_base_ang_vel: [ang[:,0], ang[:,2], ang[:,1]] = [X, +Z, Y]")
print(f"  v4_projected_gravity: [grav[:,2], grav[:,0], grav[:,1]] = [+Z, X, Y]")

print(f"\n对于V4初始姿态:")
print(f"  body frame 重力: [{grav[0]:.3f}, {grav[1]:.3f}, {grav[2]:.3f}]")

# 当前代码的重映射 [+Z, X, Y]
remap_grav = np.array([grav[2], grav[0], grav[1]])
print(f"  重映射后 [+Z,X,Y]: [{remap_grav[0]:.3f}, {remap_grav[1]:.3f}, {remap_grav[2]:.3f}]")

# 之前代码的重映射 [-Z, X, Y]
remap_grav_old = np.array([-grav[2], grav[0], grav[1]])
print(f"  旧代码 [-Z,X,Y]:   [{remap_grav_old[0]:.3f}, {remap_grav_old[1]:.3f}, {remap_grav_old[2]:.3f}]")

print(f"\n对于前进速度 (world -Y):")
print(f"  body frame: [{v_forward_body[0]:.3f}, {v_forward_body[1]:.3f}, {v_forward_body[2]:.3f}]")
remap_fwd = np.array([v_forward_body[2], v_forward_body[0], v_forward_body[1]])
print(f"  重映射后 [+Z,X,Y]: [{remap_fwd[0]:.3f}, {remap_fwd[1]:.3f}, {remap_fwd[2]:.3f}]")
remap_fwd_old = np.array([-v_forward_body[2], v_forward_body[0], v_forward_body[1]])
print(f"  旧代码 [-Z,X,Y]:   [{remap_fwd_old[0]:.3f}, {remap_fwd_old[1]:.3f}, {remap_fwd_old[2]:.3f}]")

print(f"\n--- 期望值 ---")
print(f"正确四足姿态时:")
print(f"  重力obs应该 ≈ [0, 0, -1] (重力指向下方=负的'上下'分量)")
print(f"  前进速度obs[0]应该 > 0 (正值=前进)")

print(f"\n--- 结论 ---")
if abs(remap_grav[2] - (-1.0)) < 0.1:
    print(f"✓ 当前重映射 [+Z,X,Y] 正确: 重力obs = {remap_grav} ≈ [0, 0, -1]")
elif abs(remap_grav_old[2] - (-1.0)) < 0.1:
    print(f"✗ 应该用旧重映射 [-Z,X,Y]: 重力obs = {remap_grav_old} ≈ [0, 0, -1]")
else:
    print(f"? 两种重映射都不对，需要进一步分析")
    
if remap_fwd[0] > 0:
    print(f"✓ 当前重映射 [+Z,X,Y] 正确: 前进速度obs[0] = {remap_fwd[0]:.3f} > 0")
elif remap_fwd_old[0] > 0:
    print(f"✗ 应该用旧重映射 [-Z,X,Y]: 前进速度obs[0] = {remap_fwd_old[0]:.3f} > 0")
else:
    print(f"? 两种重映射都不对，需要进一步分析")

# 6. 验证 IsaacLab 中的期望值
print(f"\n--- IsaacLab 训练中的期望值验证 ---")
print(f"在IsaacLab中，V4以quat=(0.7071,0.7071,0,0)初始化")
print(f"root_lin_vel_b 是 PhysX 直接给出的 body frame 速度")
print(f"projected_gravity_b 是 PhysX 直接给出的 body frame 重力投影")
print(f"")
print(f"PhysX body frame 和 MuJoCo body frame 应该一致（都是 R^T @ v_world）")
print(f"")
print(f"在正确四足站立时:")
print(f"  PhysX projected_gravity_b = R^T @ [0,0,-9.81] / 9.81 = [{grav[0]:.3f}, {grav[1]:.3f}, {grav[2]:.3f}]")
print(f"  v4_projected_gravity 重映射: [grav[2], grav[0], grav[1]] = [{grav[2]:.3f}, {grav[0]:.3f}, {grav[1]:.3f}]")
print(f"  这个值作为obs[6:9]输入策略")
print(f"")
print(f"  训练中 flat_orientation 惩罚检查: grav[:,0] 和 grav[:,2] 应该接近0")
print(f"  即重映射后的 [0] 和 [2] 分量应该接近0")
print(f"  重映射后: [{grav[2]:.3f}, {grav[0]:.3f}, {grav[1]:.3f}]")
print(f"  [0]={grav[2]:.3f}, [2]={grav[1]:.3f}")
