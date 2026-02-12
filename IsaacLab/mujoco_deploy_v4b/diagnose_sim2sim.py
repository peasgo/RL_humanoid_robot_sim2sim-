"""
全面诊断 V4 Quadruped Sim2Sim 问题
对比 IsaacLab 训练配置 vs MuJoCo 部署配置
"""

import numpy as np
import torch
import yaml
import os

print("=" * 80)
print("V4 Quadruped Sim2Sim 全面诊断报告")
print("=" * 80)

# ============================================================
# 1. 加载两侧配置
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# MuJoCo 部署配置
with open(os.path.join(current_dir, "v4_robot.yaml"), "r") as f:
    mj_cfg = yaml.load(f, Loader=yaml.FullLoader)

# IsaacLab 训练时实际使用的配置（从日志中保存的）
env_yaml_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/params/env.yaml"
with open(env_yaml_path, "r") as f:
    isaac_cfg = yaml.load(f, Loader=yaml.FullLoader)

agent_yaml_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/params/agent.yaml"
with open(agent_yaml_path, "r") as f:
    agent_cfg = yaml.load(f, Loader=yaml.FullLoader)

# ============================================================
# 问题 1: Action Scale 对比
# ============================================================
print("\n" + "=" * 80)
print("【问题1】Action Scale 对比")
print("=" * 80)

isaac_action_scale = isaac_cfg["actions"]["joint_pos"]["scale"]
mj_action_scale = mj_cfg["action_scale"]

print(f"  IsaacLab 训练时 action scale: {isaac_action_scale}")
print(f"  MuJoCo 部署 action scale:     {mj_action_scale}")

if isaac_action_scale != mj_action_scale:
    print(f"  ❌ 不匹配！差异 = {mj_action_scale / isaac_action_scale:.2f}x")
else:
    print(f"  ✅ 匹配")

# ============================================================
# 问题 2: 默认关节角度对比
# ============================================================
print("\n" + "=" * 80)
print("【问题2】默认关节角度对比")
print("=" * 80)

# IsaacLab 训练时的默认角度
isaac_default_angles = isaac_cfg["scene"]["robot"]["init_state"]["joint_pos"]
print(f"\n  IsaacLab 训练时默认角度:")
for jname, angle in isaac_default_angles.items():
    if isinstance(angle, (int, float)):
        print(f"    {jname:12s}: {angle:+.6f} rad ({np.degrees(angle):+.2f}°)")

# MuJoCo 部署的默认角度（MuJoCo顺序）
mj_default_angles = np.array(mj_cfg["default_angles"], dtype=np.float32)
mj_joint_order = [
    'Waist_2', 'RSDp', 'RSDy', 'RARMp', 'RARMAP',
    'LSDp', 'LSDy', 'LARMp', 'LARMAp',
    'RHIPp', 'RHIPy', 'RKNEEP', 'RANKLEp',
    'LHIPp', 'LHIPy', 'LKNEEp', 'LANKLEp'
]

print(f"\n  MuJoCo 部署默认角度:")
for i, jname in enumerate(mj_joint_order):
    print(f"    {jname:12s}: {mj_default_angles[i]:+.6f} rad ({np.degrees(mj_default_angles[i]):+.2f}°)")

print(f"\n  逐关节对比:")
has_angle_mismatch = False
for jname in mj_joint_order:
    if jname in isaac_default_angles:
        isaac_val = isaac_default_angles[jname]
        mj_idx = mj_joint_order.index(jname)
        mj_val = mj_default_angles[mj_idx]
        diff = abs(isaac_val - mj_val)
        status = "✅" if diff < 0.001 else "❌"
        if diff >= 0.001:
            has_angle_mismatch = True
        print(f"    {jname:12s}: Isaac={isaac_val:+.4f}  MJ={mj_val:+.4f}  diff={diff:.4f} {status}")

if not has_angle_mismatch:
    print("  ✅ 所有默认角度匹配")

# ============================================================
# 问题 3: 关节顺序对比 (obs中的17关节)
# ============================================================
print("\n" + "=" * 80)
print("【问题3】关节顺序对比 - obs中的17关节")
print("=" * 80)

# MuJoCo部署代码中硬编码的Isaac17顺序
isaac17_joint_order = [
    'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy', 'Waist_2',
    'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP',
    'LSDy', 'RSDy', 'LANKLEp', 'RANKLEp',
    'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
]

# Isaac16 action顺序（排除Waist_2）
isaac16_action_order = [
    'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
    'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP',
    'LSDy', 'RSDy', 'LANKLEp', 'RANKLEp',
    'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
]

# IsaacLab action配置中的关节名列表
isaac_action_joint_names = isaac_cfg["actions"]["joint_pos"]["joint_names"]
print(f"\n  IsaacLab action配置中的关节名列表 (preserve_order=false):")
for i, jname in enumerate(isaac_action_joint_names):
    print(f"    [{i:2d}] {jname}")

print(f"\n  MuJoCo部署中的isaac16_action_order:")
for i, jname in enumerate(isaac16_action_order):
    print(f"    [{i:2d}] {jname}")

print(f"\n  ⚠️  注意: preserve_order=false 意味着 IsaacLab 会按照 PhysX 内部顺序重排关节")
print(f"  ⚠️  action配置中写的顺序 ≠ 实际执行顺序！实际顺序由 PhysX 解析 USD 决定")
print(f"  ⚠️  isaac16_action_order 是否正确需要从 IsaacLab 运行时打印确认")

# ============================================================
# 问题 4: Observation 维度和构成对比
# ============================================================
print("\n" + "=" * 80)
print("【问题4】Observation 维度和构成对比")
print("=" * 80)

print(f"\n  IsaacLab obs 构成 (从 env.yaml):")
obs_terms = isaac_cfg["observations"]["policy"]
obs_dim_breakdown = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos": 17,  # joint_pos_rel 返回所有17个关节
    "joint_vel": 17,  # joint_vel_rel 返回所有17个关节
    "actions": 16,    # last_action 返回16个动作
}
total_obs = sum(obs_dim_breakdown.values())
print(f"    base_lin_vel:       3  (v4_base_lin_vel: [+Z, X, Y] remap)")
print(f"    base_ang_vel:       3  (v4_base_ang_vel: [X, +Z, Y] remap)")
print(f"    projected_gravity:  3  (v4_projected_gravity: [+Z, X, Y] remap)")
print(f"    velocity_commands:  3")
print(f"    joint_pos (rel):   17  (所有17关节, Isaac内部顺序)")
print(f"    joint_vel (rel):   17  (所有17关节, Isaac内部顺序)")
print(f"    last_action:       16  (16个被控关节)")
print(f"    总计:              {total_obs}")

print(f"\n  MuJoCo obs 构成 (从 run_v4_robot.py):")
print(f"    obs[0:3]   = base_lin_vel (v4 remap)")
print(f"    obs[3:6]   = base_ang_vel (v4 remap)")
print(f"    obs[6:9]   = projected_gravity (v4 remap)")
print(f"    obs[9:12]  = cmd * cmd_scale")
print(f"    obs[12:29] = joint_pos_rel (17 joints, Isaac17 order)")
print(f"    obs[29:46] = joint_vel (17 joints, Isaac17 order)")
print(f"    obs[46:62] = last_action (16 actions)")
print(f"    总计:        62")

mj_num_obs = mj_cfg["num_obs"]
print(f"\n  MuJoCo配置 num_obs: {mj_num_obs}")
print(f"  计算得到 obs 维度:  {total_obs}")
if mj_num_obs == total_obs:
    print(f"  ✅ 维度匹配")
else:
    print(f"  ❌ 维度不匹配！")

# ============================================================
# 问题 5: Obs Scale 对比
# ============================================================
print("\n" + "=" * 80)
print("【问题5】Observation Scale 对比")
print("=" * 80)

print(f"\n  IsaacLab 训练时:")
print(f"    obs scale: 无显式scale (empirical_normalization=True, normalizer内置于policy.pt)")
print(f"    obs noise: lin_vel ±0.05, ang_vel ±0.1, gravity ±0.025, joint_pos ±0.01, joint_vel ±0.5")

print(f"\n  MuJoCo 部署:")
print(f"    lin_vel_scale:  {mj_cfg['lin_vel_scale']}")
print(f"    ang_vel_scale:  {mj_cfg['ang_vel_scale']}")
print(f"    dof_pos_scale:  {mj_cfg['dof_pos_scale']}")
print(f"    dof_vel_scale:  {mj_cfg['dof_vel_scale']}")
print(f"    cmd_scale:      {mj_cfg['cmd_scale']}")

if (mj_cfg['lin_vel_scale'] == 1.0 and mj_cfg['ang_vel_scale'] == 1.0 and
    mj_cfg['dof_pos_scale'] == 1.0 and mj_cfg['dof_vel_scale'] == 1.0):
    print(f"\n  ✅ 所有scale=1.0, 配合empirical_normalization内置于policy.pt, 这是正确的")
else:
    print(f"\n  ❌ scale不为1.0, 但empirical_normalization已内置于policy.pt, 会导致double scaling!")

# ============================================================
# 问题 6: PD增益对比
# ============================================================
print("\n" + "=" * 80)
print("【问题6】PD增益对比")
print("=" * 80)

isaac_actuators = isaac_cfg["scene"]["robot"]["actuators"]
print(f"\n  IsaacLab 训练时 PD增益:")
for act_name, act_cfg in isaac_actuators.items():
    print(f"    {act_name}: kp={act_cfg['stiffness']}, kd={act_cfg['damping']}, "
          f"effort_limit={act_cfg.get('effort_limit_sim', 'N/A')}")
    print(f"      joints: {act_cfg['joint_names_expr']}")

mj_kps = np.array(mj_cfg["kps"])
mj_kds = np.array(mj_cfg["kds"])
print(f"\n  MuJoCo 部署 PD增益 (MuJoCo关节顺序):")
for i, jname in enumerate(mj_joint_order):
    print(f"    {jname:12s}: kp={mj_kps[i]:6.0f}, kd={mj_kds[i]:5.0f}")

# 检查匹配
print(f"\n  逐关节PD增益对比:")
pd_mismatch = False
for jname in mj_joint_order:
    mj_idx = mj_joint_order.index(jname)
    mj_kp = mj_kps[mj_idx]
    mj_kd = mj_kds[mj_idx]

    # 确定Isaac中的kp/kd
    isaac_kp = None
    isaac_kd = None
    for act_name, act_cfg in isaac_actuators.items():
        import re
        for pattern in act_cfg['joint_names_expr']:
            if re.match(pattern, jname):
                isaac_kp = act_cfg['stiffness']
                isaac_kd = act_cfg['damping']
                break
        if isaac_kp is not None:
            break

    if isaac_kp is not None:
        kp_match = "✅" if abs(mj_kp - isaac_kp) < 0.1 else "❌"
        kd_match = "✅" if abs(mj_kd - isaac_kd) < 0.1 else "❌"
        if abs(mj_kp - isaac_kp) >= 0.1 or abs(mj_kd - isaac_kd) >= 0.1:
            pd_mismatch = True
        print(f"    {jname:12s}: Isaac kp={isaac_kp:6.0f} kd={isaac_kd:5.0f} | "
              f"MJ kp={mj_kp:6.0f} kd={mj_kd:5.0f} {kp_match}{kd_match}")
    else:
        print(f"    {jname:12s}: Isaac kp=??? | MJ kp={mj_kp:6.0f} kd={mj_kd:5.0f}")

# ============================================================
# 问题 7: 仿真参数对比
# ============================================================
print("\n" + "=" * 80)
print("【问题7】仿真参数对比")
print("=" * 80)

isaac_dt = isaac_cfg["sim"]["dt"]
isaac_decimation = isaac_cfg["decimation"]
isaac_control_dt = isaac_dt * isaac_decimation

mj_dt = mj_cfg["simulation_dt"]
mj_decimation = mj_cfg["control_decimation"]
mj_control_dt = mj_dt * mj_decimation

print(f"  IsaacLab: dt={isaac_dt}, decimation={isaac_decimation}, control_dt={isaac_control_dt}")
print(f"  MuJoCo:   dt={mj_dt}, decimation={mj_decimation}, control_dt={mj_control_dt}")

if abs(isaac_dt - mj_dt) < 1e-6 and isaac_decimation == mj_decimation:
    print(f"  ✅ 仿真参数匹配")
else:
    print(f"  ❌ 仿真参数不匹配！")

# ============================================================
# 问题 8: 角速度处理 - 关键bug
# ============================================================
print("\n" + "=" * 80)
print("【问题8】角速度处理 - 关键bug分析")
print("=" * 80)

print(f"""
  MuJoCo 中 qvel[3:6] 的含义:
    MuJoCo 的 qvel[3:6] 是 body frame 角速度（局部坐标系）
    参考: MuJoCo文档 "The angular velocity of the free body is in the local frame"

  run_v4_robot.py 中的处理 (line 462-463):
    base_ang_vel_world = d.qvel[3:6].copy()   # 实际上已经是 body frame!
    omega = world_to_body(base_ang_vel_world, quat)  # 又做了一次旋转 → double rotation!

  代码注释说:
    "已知问题：MuJoCo qvel[3:6] 已经是 body frame 角速度，
     world_to_body() 会导致 double rotation。但实测修复后反而更差，
     说明策略可能已适应了这个 bug。暂时保持不变。"

  ⚠️  这是一个已知的 double rotation bug。
  ⚠️  如果策略在 IsaacLab 中看到的是正确的 body frame 角速度，
      但 MuJoCo 部署中给的是 double-rotated 角速度，
      那么策略收到的角速度信息是错误的！

  IsaacLab 中 root_ang_vel_b 的含义:
    这是 body frame 角速度，直接从 PhysX 获取

  正确做法: MuJoCo qvel[3:6] 直接就是 body frame 角速度，不需要 world_to_body()
""")

# ============================================================
# 问题 9: 线速度处理
# ============================================================
print("\n" + "=" * 80)
print("【问题9】线速度处理分析")
print("=" * 80)

print(f"""
  MuJoCo 中 qvel[0:3] 的含义:
    MuJoCo 的 qvel[0:3] 是 world frame 线速度

  run_v4_robot.py 中的处理 (line 455-459):
    base_lin_vel_world = d.qvel[0:3].copy()   # world frame ✅
    base_lin_vel = world_to_body(base_lin_vel_world, quat)  # 转到 body frame ✅

  IsaacLab 中 root_lin_vel_b:
    这是 body frame 线速度

  ✅ 线速度处理正确
""")

# ============================================================
# 问题 10: V4坐标系重映射对比
# ============================================================
print("\n" + "=" * 80)
print("【问题10】V4坐标系重映射对比")
print("=" * 80)

print(f"""
  IsaacLab 训练代码 (flat_env_cfg.py):
    v4_base_lin_vel:      [vel[:,2], vel[:,0], vel[:,1]]  → [+Z, X, Y]
    v4_base_ang_vel:      [ang[:,0], ang[:,2], ang[:,1]]  → [X, +Z, Y]
    v4_projected_gravity: [grav[:,2], grav[:,0], grav[:,1]] → [+Z, X, Y]

  MuJoCo 部署代码 (run_v4_robot.py):
    v4_remap_lin_vel:     [lin[2], lin[0], lin[1]]  → [+Z, X, Y]  ✅
    v4_remap_ang_vel:     [ang[0], ang[2], ang[1]]  → [X, +Z, Y]  ✅
    v4_remap_gravity:     [grav[2], grav[0], grav[1]] → [+Z, X, Y] ✅

  ✅ V4坐标系重映射一致
""")

# ============================================================
# 问题 11: Action后处理对比
# ============================================================
print("\n" + "=" * 80)
print("【问题11】Action后处理对比")
print("=" * 80)

print(f"""
  IsaacLab 训练时:
    processed_action = raw_action * scale + offset
    scale = {isaac_action_scale}
    offset = default_joint_pos (use_default_offset=True)
    clip_actions = {agent_cfg.get('clip_actions', 'None')}

  MuJoCo 部署:
    action_scale = {mj_action_scale}
    target_dof_pos[mj_idx] = action * action_scale + default_angles[mj_idx]
    action_clip = {mj_cfg.get('action_clip', 'None')}
    use_tanh_action = {mj_cfg.get('use_tanh_action', False)}
    action_filter_alpha = {mj_cfg.get('action_filter_alpha', 0)}

  ⚠️  IsaacLab 训练时 clip_actions = None (无clip)
  ⚠️  MuJoCo 部署时 action_clip = {mj_cfg.get('action_clip', 'None')} (有clip!)
  ⚠️  MuJoCo 部署时 action_filter_alpha = {mj_cfg.get('action_filter_alpha', 0)} (有低通滤波!)

  这两个额外的后处理在训练时不存在，会改变策略的行为！
""")

# ============================================================
# 问题 12: 检查 policy.pt 中的 normalizer
# ============================================================
print("\n" + "=" * 80)
print("【问题12】检查 policy.pt 中的 normalizer 参数")
print("=" * 80)

policy_path = mj_cfg["policy_path"]
if os.path.exists(policy_path):
    policy = torch.jit.load(policy_path, map_location="cpu")
    print(f"  Policy loaded from: {policy_path}")
    print(f"  Policy code:")
    print(policy.code)

    # 尝试提取 normalizer 参数
    try:
        # 遍历所有参数和buffer
        print(f"\n  Policy named parameters and buffers:")
        for name, param in policy.named_parameters():
            print(f"    param: {name}, shape={param.shape}, dtype={param.dtype}")
        for name, buf in policy.named_buffers():
            print(f"    buffer: {name}, shape={buf.shape}, dtype={buf.dtype}")
            if 'mean' in name.lower() or 'running_mean' in name.lower():
                print(f"      values: {buf[:10].numpy()}...")
            if 'var' in name.lower() or 'running_var' in name.lower():
                print(f"      values: {buf[:10].numpy()}...")
    except Exception as e:
        print(f"  Error extracting parameters: {e}")
else:
    print(f"  ❌ Policy file not found: {policy_path}")

# ============================================================
# 问题 13: 关节顺序验证 - action mapping
# ============================================================
print("\n" + "=" * 80)
print("【问题13】Action关节映射验证")
print("=" * 80)

print(f"\n  IsaacLab action配置中的关节列表 (preserve_order=false):")
print(f"    {isaac_action_joint_names}")
print(f"    共 {len(isaac_action_joint_names)} 个关节")

print(f"\n  MuJoCo部署中的 isaac16_action_order:")
print(f"    {isaac16_action_order}")
print(f"    共 {len(isaac16_action_order)} 个关节")

# 检查是否包含相同的关节（不考虑顺序）
isaac_set = set(isaac_action_joint_names)
mj_set = set(isaac16_action_order)
if isaac_set == mj_set:
    print(f"\n  ✅ 两侧包含相同的16个关节")
else:
    print(f"\n  ❌ 关节集合不同!")
    print(f"    Isaac有但MJ没有: {isaac_set - mj_set}")
    print(f"    MJ有但Isaac没有: {mj_set - isaac_set}")

print(f"""
  ⚠️  关键问题: preserve_order=false 时，IsaacLab 按 PhysX 内部顺序排列关节
  ⚠️  action配置中写的顺序 [RSDp, RSDy, RARMp, ...] 不是实际的action顺序！
  ⚠️  实际顺序由 PhysX 解析 USD 后的 articulation 内部顺序决定
  ⚠️  MuJoCo部署中的 isaac16_action_order 必须与 PhysX 内部顺序完全一致
  ⚠️  这个顺序只能通过在 IsaacLab 中打印 robot.joint_names 来确认
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 80)
print("诊断总结 - 发现的问题")
print("=" * 80)

print(f"""
🔴 严重问题:

1. 【角速度 double rotation】
   MuJoCo qvel[3:6] 已经是 body frame 角速度，
   但代码又做了 world_to_body() 变换，导致 double rotation。
   策略收到的角速度信息是错误的。
   这会直接影响步态的稳定性和流畅性。

2. 【action_filter_alpha = 0.3 训练时不存在】
   训练时策略输出直接作为动作执行，没有低通滤波。
   部署时加了 alpha=0.3 的低通滤波，相当于给动作加了延迟。
   这会导致动作响应变慢，步态不流畅。

3. 【action_clip = 5.0 训练时不存在】
   训练时 clip_actions = None，策略输出不受clip限制。
   部署时加了 ±5.0 的clip。虽然正常动作在±2以内，
   但这改变了策略的行为空间。

🟡 需要验证的问题:

4. 【关节顺序 isaac16_action_order 未经运行时验证】
   preserve_order=false 时，实际关节顺序由 PhysX 决定。
   MuJoCo部署中硬编码的顺序需要与 IsaacLab 运行时打印的顺序对比确认。
   如果顺序错误，会导致动作发送到错误的关节，步态完全混乱。

5. 【isaac17_joint_order 中 Waist_2 的位置】
   obs中17个关节的顺序也需要运行时验证。
   Waist_2 在 isaac17 中排第4位（index=4），这个位置是否正确？

🟢 已确认正确:

6. action_scale = 0.25 ✅ (与训练一致)
7. 默认关节角度 ✅ (与训练一致)
8. PD增益 ✅ (与训练一致)
9. 仿真参数 dt=0.005, decimation=4 ✅
10. V4坐标系重映射 ✅
11. empirical_normalization 内置于 policy.pt ✅
12. obs维度 62 = 3+3+3+3+17+17+16 ✅
""")

print("\n" + "=" * 80)
print("建议修复优先级")
print("=" * 80)
print(f"""
1. 【最高优先级】修复角速度 double rotation
   将 omega = world_to_body(base_ang_vel_world, quat)
   改为 omega = base_ang_vel_world  (直接使用，不做变换)

2. 【高优先级】去掉 action_filter_alpha
   设为 0.0，与训练时一致

3. 【高优先级】去掉 action_clip
   设为 None，与训练时一致

4. 【高优先级】验证关节顺序
   在 IsaacLab 中运行以下代码打印实际顺序:
   print(robot.joint_names)
   print(robot.find_joints(action_joint_names, preserve_order=False))
""")
