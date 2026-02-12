"""
诊断脚本：确认V3机器人的前方方向和重力投影
在训练环境中运行，打印站立时的 projected_gravity_b 值
"""

import torch
from isaaclab.utils.math import quat_apply_inverse

def debug_gravity_projection(env, asset_cfg_name="robot"):
    """
    在环境 step 中调用此函数，打印重力投影信息
    
    用法：在奖励函数中临时添加调用
    """
    robot = env.scene[asset_cfg_name]
    root_quat = robot.data.root_quat_w
    root_pos = robot.data.root_pos_w
    
    # 方法1：使用 Isaac Lab 内置的 projected_gravity_b
    pg_builtin = robot.data.projected_gravity_b
    
    # 方法2：手动计算（与奖励函数中的方式一致）
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device)
    gravity_world = gravity_world.unsqueeze(0).expand(root_quat.shape[0], -1)
    pg_manual = quat_apply_inverse(root_quat, gravity_world)
    
    # 获取前方向量（Isaac Lab 默认 X 前方）
    forward_b = torch.tensor([1.0, 0.0, 0.0], device=root_quat.device)
    forward_b = forward_b.unsqueeze(0).expand(root_quat.shape[0], -1)
    from isaaclab.utils.math import quat_apply
    forward_w = quat_apply(root_quat, forward_b)
    
    # 只打印第一个环境的值
    print(f"\n=== 重力投影诊断 (env 0) ===")
    print(f"  质心高度: {root_pos[0, 2]:.4f} m")
    print(f"  四元数 (w,x,y,z): {root_quat[0].tolist()}")
    print(f"  内置 projected_gravity_b: x={pg_builtin[0, 0]:.4f}, y={pg_builtin[0, 1]:.4f}, z={pg_builtin[0, 2]:.4f}")
    print(f"  手动 gravity_body:        x={pg_manual[0, 0]:.4f}, y={pg_manual[0, 1]:.4f}, z={pg_manual[0, 2]:.4f}")
    print(f"  前方向量(世界系):          x={forward_w[0, 0]:.4f}, y={forward_w[0, 1]:.4f}, z={forward_w[0, 2]:.4f}")
    print(f"  解读:")
    print(f"    站立时 gravity_b 应该 ≈ (0, 0, -1)")
    print(f"    前倾时 gravity_b 的哪个分量变化最大？那就是前方轴")
    print(f"================================\n")
    
    return pg_manual
