#!/usr/bin/env python3
"""V4b sim2sim 闭环诊断脚本

在 MuJoCo 中运行策略几步，记录完整的 obs/action 数据，
分析为什么不管什么 cmd 都往前走。

测试方案：
1. cmd=[0,0,0] (站立) - 应该不动
2. cmd=[0.5,0,0] (前进) - 应该前进
3. cmd=[-0.5,0,0] (后退) - 应该后退

对比三种 cmd 下的：
- 策略输出 action 的差异
- obs 中各分量的变化
- 机器人实际位移
"""

import time
import mujoco
import numpy as np
import torch
import yaml
import os

np.set_printoptions(precision=4, suppress=True, linewidth=120)


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


def v4_remap_lin_vel(v):
    return np.array([v[2], v[0], v[1]])

def v4_remap_ang_vel(w):
    return np.array([w[0], w[2], w[1]])

def v4_remap_gravity(g):
    return np.array([g[2], g[0], g[1]])


def run_test(cmd_name, cmd_values, m, d, policy, config, num_policy_steps=200):
    """运行一次测试，返回记录的数据"""
    
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    action_scale = config["action_scale"]
    control_decimation = config["control_decimation"]
    
    # Isaac17 顺序
    isaac17_joint_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy', 'Waist_2',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]
    isaac16_action_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP',
        'LSDy', 'RSDy', 'LANKLEp', 'RANKLEp',
        'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]
    
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)
    
    isaac17_to_mujoco17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac17_joint_order], dtype=np.int32
    )
    isaac16_action_to_mj17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac16_action_order], dtype=np.int32
    )
    
    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]
    
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)
    
    # Reset
    mujoco.mj_resetData(m, d)
    init_height = 0.22
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target_dof_pos = default_angles.copy()
    action_isaac16 = np.zeros(16, dtype=np.float32)
    action_isaac16_prev = np.zeros(16, dtype=np.float32)
    prev_obs = np.zeros(62, dtype=np.float32)
    obs = np.zeros(62, dtype=np.float32)
    
    cmd = np.array(cmd_values, dtype=np.float32)
    
    # Warmup
    warmup_steps = int(2.0 / m.opt.timestep)
    for _ in range(warmup_steps):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    
    # Record initial position
    init_pos = d.qpos[0:3].copy()
    
    # Run policy
    records = []
    counter = 0
    policy_step = 0
    
    action_filter_alpha = float(config.get("action_filter_alpha", 0.0))
    obs_filter_alpha = float(config.get("obs_filter_alpha", 0.0))
    obs_filter_mode = str(config.get("obs_filter_mode", "all"))
    action_ramp_steps = int(config.get("action_ramp_steps", 0))
    action_clip = config.get("action_clip", None)
    if action_clip is not None:
        action_clip = float(action_clip)
    
    while policy_step < num_policy_steps:
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_decimation == 0:
            quat = d.qpos[3:7]
            base_lin_vel_world = d.qvel[0:3].copy()
            base_ang_vel_world = d.qvel[3:6].copy()
            
            base_lin_vel = world_to_body(base_lin_vel_world, quat)
            omega = world_to_body(base_ang_vel_world, quat)
            
            qj_mujoco = d.qpos[7:].copy()
            dqj_mujoco = d.qvel[6:].copy()
            
            qj_isaac17 = qj_mujoco[isaac17_to_mujoco17]
            dqj_isaac17 = dqj_mujoco[isaac17_to_mujoco17]
            
            default_angles_isaac17 = default_angles[isaac17_to_mujoco17]
            gravity_orientation = get_gravity_orientation(quat)
            
            base_lin_vel_obs = v4_remap_lin_vel(base_lin_vel)
            omega_obs = v4_remap_ang_vel(omega)
            gravity_obs = v4_remap_gravity(gravity_orientation)
            
            qj = (qj_isaac17 - default_angles_isaac17)
            dqj = dqj_isaac17
            
            obs[0:3] = base_lin_vel_obs
            obs[3:6] = omega_obs
            obs[6:9] = gravity_obs
            obs[9:12] = cmd
            obs[12:29] = qj
            obs[29:46] = dqj
            obs[46:62] = action_isaac16.astype(np.float32)
            
            # Obs filter
            if obs_filter_alpha > 0 and policy_step > 0:
                if obs_filter_mode == "vel_only":
                    obs[0:6] = obs_filter_alpha * prev_obs[0:6] + (1.0 - obs_filter_alpha) * obs[0:6]
                    obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1.0 - obs_filter_alpha) * obs[29:46]
                else:
                    obs[:] = obs_filter_alpha * prev_obs + (1.0 - obs_filter_alpha) * obs
            prev_obs[:] = obs
            
            # Policy inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_isaac16 = policy(obs_tensor).detach().numpy().squeeze()
            
            if action_clip is not None:
                action_isaac16 = np.clip(action_isaac16, -action_clip, action_clip)
            
            if action_ramp_steps > 0 and policy_step < action_ramp_steps:
                ramp_factor = float(policy_step) / float(action_ramp_steps)
                action_isaac16 = action_isaac16 * ramp_factor
            
            if action_filter_alpha > 0:
                action_isaac16 = action_filter_alpha * action_isaac16_prev + (1.0 - action_filter_alpha) * action_isaac16
            action_isaac16_prev[:] = action_isaac16
            
            # Apply action
            target_dof_pos[waist_mj_idx] = waist_default
            for i16 in range(16):
                mj_idx = isaac16_action_to_mj17[i16]
                target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]
            
            # Record
            pos = d.qpos[0:3].copy()
            if policy_step % 20 == 0 or policy_step < 5:
                records.append({
                    'step': policy_step,
                    'pos': pos.copy(),
                    'displacement': pos - init_pos,
                    'obs_lin_vel': obs[0:3].copy(),
                    'obs_ang_vel': obs[3:6].copy(),
                    'obs_gravity': obs[6:9].copy(),
                    'obs_cmd': obs[9:12].copy(),
                    'obs_qj_rel': obs[12:29].copy(),
                    'action_raw': action_isaac16.copy(),
                    'action_mean': np.mean(action_isaac16),
                    'action_abs_mean': np.mean(np.abs(action_isaac16)),
                    'height': pos[2],
                })
            
            policy_step += 1
    
    final_pos = d.qpos[0:3].copy()
    displacement = final_pos - init_pos
    
    return {
        'cmd_name': cmd_name,
        'cmd': cmd_values,
        'init_pos': init_pos,
        'final_pos': final_pos,
        'displacement': displacement,
        'records': records,
    }


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "v4_robot.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)
    
    policy_path = config["policy_path"]
    
    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    print(f"Loading policy: {policy_path}")
    policy = torch.jit.load(policy_path)
    
    print(f"Action scale: {config['action_scale']}")
    print(f"Default angles (MJ order): {config['default_angles']}")
    print()
    
    # 测试三种 cmd
    tests = [
        ("站立 cmd=[0,0,0]", [0.0, 0.0, 0.0]),
        ("前进 cmd=[0.5,0,0]", [0.5, 0.0, 0.0]),
        ("后退 cmd=[-0.5,0,0]", [-0.5, 0.0, 0.0]),
        ("左转 cmd=[0,0,0.5]", [0.0, 0.0, 0.5]),
    ]
    
    results = []
    for name, cmd in tests:
        print(f"{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")
        result = run_test(name, cmd, m, d, policy, config, num_policy_steps=250)
        results.append(result)
        
        print(f"  初始位置: {result['init_pos']}")
        print(f"  最终位置: {result['final_pos']}")
        print(f"  位移: dx={result['displacement'][0]:+.4f}, dy={result['displacement'][1]:+.4f}, dz={result['displacement'][2]:+.4f}")
        print()
        
        # 打印关键时间步
        for r in result['records']:
            if r['step'] < 5 or r['step'] % 50 == 0:
                print(f"  step={r['step']:3d}: pos=({r['pos'][0]:+.3f},{r['pos'][1]:+.3f},{r['pos'][2]:.3f}) "
                      f"lin_vel={r['obs_lin_vel']} gravity={r['obs_gravity']} "
                      f"act_mean={r['action_mean']:+.3f} act_abs={r['action_abs_mean']:.3f}")
        print()
    
    # 对比分析
    print(f"\n{'='*60}")
    print("对比分析")
    print(f"{'='*60}")
    print()
    
    print("位移对比 (MuJoCo 世界坐标):")
    print(f"  {'测试':>20s} | {'dx':>8s} | {'dy':>8s} | {'dz':>8s} | {'水平距离':>10s}")
    print(f"  {'-'*60}")
    for r in results:
        dx, dy, dz = r['displacement']
        dist = np.sqrt(dx**2 + dy**2)
        print(f"  {r['cmd_name']:>20s} | {dx:+8.4f} | {dy:+8.4f} | {dz:+8.4f} | {dist:10.4f}")
    
    print()
    print("V4 坐标系说明:")
    print("  V4 前进方向 = 世界 -Y (因为 base_link 绕X轴旋转+90°)")
    print("  V4 左右方向 = 世界 X")
    print("  V4 上下方向 = 世界 Z")
    print()
    print("  所以 dy < 0 表示前进, dy > 0 表示后退")
    print()
    
    # 分析第一步的 obs 差异
    print("第一步 obs 对比 (step=0):")
    for r in results:
        rec = r['records'][0]
        print(f"  {r['cmd_name']:>20s}: lin_vel={rec['obs_lin_vel']} ang_vel={rec['obs_ang_vel']} "
              f"gravity={rec['obs_gravity']} cmd={rec['obs_cmd']}")
    
    print()
    print("第一步 action 对比 (step=0):")
    for r in results:
        rec = r['records'][0]
        print(f"  {r['cmd_name']:>20s}: action={rec['action_raw']}")
    
    # 分析 action 差异
    if len(results) >= 3:
        stand_action = results[0]['records'][0]['action_raw']
        fwd_action = results[1]['records'][0]['action_raw']
        back_action = results[2]['records'][0]['action_raw']
        
        print()
        print("Action 差异分析 (step=0):")
        print(f"  前进-站立: {fwd_action - stand_action}")
        print(f"  后退-站立: {back_action - stand_action}")
        print(f"  前进-后退: {fwd_action - back_action}")
    
    # 检查 gravity 向量是否正确
    print()
    print("重力向量检查:")
    print("  V4 四足姿态时，base_link 绕X轴旋转+90°")
    print("  quat = (0.7071, 0.7071, 0, 0)")
    quat_test = np.array([0.70710678, 0.70710678, 0.0, 0.0])
    grav_raw = get_gravity_orientation(quat_test)
    grav_remapped = v4_remap_gravity(grav_raw)
    print(f"  raw gravity (body frame): {grav_raw}")
    print(f"  remapped gravity [gz,gx,gy]: {grav_remapped}")
    print(f"  期望: 重力沿 body -Y 方向 → raw=(0,-1,0) → remapped=(0,0,-1)")
    
    # 检查 IsaacLab 的 projected_gravity
    print()
    print("IsaacLab projected_gravity 对比:")
    print(f"  IsaacLab 捕获: [0.025, 0.014, -1.023]")
    print(f"  这是 V4 remap 后的值: [gz, gx, gy]")
    print(f"  还原: gx=0.014, gy=-1.023, gz=0.025")
    print(f"  MuJoCo 计算 (理想): raw={grav_raw} → remapped={grav_remapped}")
    
    # 检查 lin_vel remap
    print()
    print("线速度 remap 检查:")
    print("  V4 前进 = body +Z → remap 后 obs[0] = body_vel_z")
    print("  V4 左右 = body X → remap 后 obs[1] = body_vel_x")
    print("  V4 上下 = body Y → remap 后 obs[2] = body_vel_y")
    print()
    print("  当机器人在世界 -Y 方向前进时:")
    print("    world_vel = (0, -v, 0)")
    print("    body_vel = R^T @ world_vel")
    
    # 计算 body frame velocity for forward motion
    R = quat_to_rotmat_wxyz(quat_test)
    world_vel_fwd = np.array([0, -1, 0])  # 世界 -Y = V4 前进
    body_vel = R.T @ world_vel_fwd
    remapped_vel = v4_remap_lin_vel(body_vel)
    print(f"    body_vel = {body_vel}")
    print(f"    remapped [vz,vx,vy] = {remapped_vel}")
    print(f"    期望: obs[0] (前进速度) > 0 当前进时")
    
    # 检查 ang_vel remap
    print()
    print("角速度 remap 检查:")
    print("  V4 yaw = 绕世界 Z 轴 = 绕 body Y 轴")
    world_angvel_yaw = np.array([0, 0, 1])  # 绕世界 Z 轴
    body_angvel = R.T @ world_angvel_yaw
    remapped_angvel = v4_remap_ang_vel(body_angvel)
    print(f"  world_angvel = {world_angvel_yaw}")
    print(f"  body_angvel = {body_angvel}")
    print(f"  remapped [wx, wz, wy] = {remapped_angvel}")
    print(f"  期望: obs[5] (yaw rate) > 0 当左转时")


if __name__ == "__main__":
    main()
