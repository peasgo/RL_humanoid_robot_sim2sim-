#!/usr/bin/env python3
"""综合诊断：测试多种修复组合，找出根本原因。

测试矩阵：
1. angvel: double_rot (当前) vs direct (修复)
2. last_action: filtered (当前) vs raw (修复)  
3. action_filter: 0.3 (当前) vs 0.0 (无滤波)
4. obs_filter: 0.3 (当前) vs 0.0 (无滤波)
5. action_ramp: 50 (当前) vs 0 (无ramp)

对每种组合，测试 FWD/STAND/BWD 三个cmd。
"""
import numpy as np
import mujoco
import torch
import yaml
import os
import itertools

np.set_printoptions(precision=4, suppress=True, linewidth=150)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

policy_path = cfg["policy_path"]
policy = torch.jit.load(policy_path, map_location="cpu")
policy.eval()

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_body(v, q):
    R = quat_to_rotmat(q)
    return R.T @ v

def get_gravity(q):
    w, x, y, z = q
    return np.array([-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))])

def v4_remap_lin(v): return np.array([v[2], v[0], v[1]])
def v4_remap_ang(v): return np.array([v[0], v[2], v[1]])
def v4_remap_grav(v): return np.array([v[2], v[0], v[1]])


def run_sim(cmd_vec, angvel_mode="double_rot", last_action_mode="filtered",
            action_filter_alpha=0.3, obs_filter_alpha=0.3, 
            action_ramp_steps=50, num_steps=500):
    """运行一次仿真，返回最终Y位移（前进方向=-Y）"""
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    m.opt.timestep = cfg["simulation_dt"]
    
    default_angles = np.array(cfg["default_angles"], dtype=np.float64)
    action_scale = cfg["action_scale"]
    control_dec = cfg["control_decimation"]
    
    # 关节映射
    mj_names = []
    for jid in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_names.append(jn)
    
    isaac17 = ['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2',
               'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
               'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']
    isaac16 = [j for j in isaac17 if j != 'Waist_2']
    
    i17_to_mj = np.array([mj_names.index(j) for j in isaac17])
    i16_to_mj = np.array([mj_names.index(j) for j in isaac16])
    waist_mj = mj_names.index('Waist_2')
    
    act_to_jnt = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        act_to_jnt.append(mj_names.index(jn))
    act_to_jnt = np.array(act_to_jnt)
    
    # 初始化
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target = default_angles.copy()
    action16 = np.zeros(16, dtype=np.float32)
    action16_prev = np.zeros(16, dtype=np.float32)
    raw_action16 = np.zeros(16, dtype=np.float32)  # 策略原始输出
    obs = np.zeros(62, dtype=np.float32)
    prev_obs = np.zeros(62, dtype=np.float32)
    
    cmd = np.array(cmd_vec, dtype=np.float32)
    
    init_pos = d.qpos[0:3].copy()
    counter = 0
    policy_step = 0
    
    total_physics_steps = num_steps * control_dec
    
    for step in range(total_physics_steps):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_dec == 0:
            quat = d.qpos[3:7]
            lin_vel_w = d.qvel[0:3].copy()
            ang_vel_raw = d.qvel[3:6].copy()
            
            # 线速度：始终 world->body
            lin_vel_b = world_to_body(lin_vel_w, quat)
            
            # 角速度
            if angvel_mode == "double_rot":
                omega = world_to_body(ang_vel_raw, quat)  # 当前代码（bug）
            else:
                omega = ang_vel_raw.copy()  # 修复：直接用body frame
            
            qj_mj = d.qpos[7:].copy()
            dqj_mj = d.qvel[6:].copy()
            qj_i17 = qj_mj[i17_to_mj]
            dqj_i17 = dqj_mj[i17_to_mj]
            def_i17 = default_angles[i17_to_mj]
            
            grav = get_gravity(quat)
            
            lin_obs = v4_remap_lin(lin_vel_b)
            ang_obs = v4_remap_ang(omega)
            grav_obs = v4_remap_grav(grav)
            
            qj_rel = qj_i17 - def_i17
            
            # last_action 选择
            if last_action_mode == "raw":
                la = raw_action16.copy()
            else:
                la = action16.copy()  # 当前代码：经过filter/ramp后的
            
            obs[0:3] = lin_obs
            obs[3:6] = ang_obs
            obs[6:9] = grav_obs
            obs[9:12] = cmd
            obs[12:29] = qj_rel.astype(np.float32)
            obs[29:46] = dqj_i17.astype(np.float32)
            obs[46:62] = la.astype(np.float32)
            
            # obs滤波
            if obs_filter_alpha > 0 and policy_step > 0:
                obs[0:6] = obs_filter_alpha * prev_obs[0:6] + (1-obs_filter_alpha) * obs[0:6]
                obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1-obs_filter_alpha) * obs[29:46]
            prev_obs[:] = obs
            
            # 策略推理
            with torch.no_grad():
                out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
            raw_action16[:] = out  # 保存原始输出
            action16[:] = out.copy()
            
            # clip
            action16 = np.clip(action16, -5.0, 5.0)
            
            # ramp
            if action_ramp_steps > 0 and policy_step < action_ramp_steps:
                action16 *= float(policy_step) / float(action_ramp_steps)
            
            # action filter
            if action_filter_alpha > 0:
                action16 = action_filter_alpha * action16_prev + (1-action_filter_alpha) * action16
            action16_prev[:] = action16
            
            policy_step += 1
            
            # 更新目标
            target[waist_mj] = default_angles[waist_mj]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
    
    final_pos = d.qpos[0:3]
    fwd = -(final_pos[1] - init_pos[1])  # -Y = forward
    return fwd


# ============================================================
# 测试矩阵
# ============================================================
print("="*100)
print("综合诊断：测试不同修复组合")
print("="*100)

cmds = {
    "FWD":   [0.3, 0, 0],
    "STAND": [0, 0, 0],
    "BWD":   [-0.3, 0, 0],
}

# 先测试当前配置（baseline）
print("\n--- BASELINE (当前代码配置) ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="double_rot", last_action_mode="filtered",
                  action_filter_alpha=0.3, obs_filter_alpha=0.3, action_ramp_steps=50)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试1: 只修复角速度
print("\n--- FIX 1: 角速度 direct (不做double rotation) ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="direct", last_action_mode="filtered",
                  action_filter_alpha=0.3, obs_filter_alpha=0.3, action_ramp_steps=50)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试2: 只修复last_action
print("\n--- FIX 2: last_action 用策略原始输出 ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="double_rot", last_action_mode="raw",
                  action_filter_alpha=0.3, obs_filter_alpha=0.3, action_ramp_steps=50)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试3: 去掉所有滤波和ramp
print("\n--- FIX 3: 无滤波无ramp ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="double_rot", last_action_mode="filtered",
                  action_filter_alpha=0.0, obs_filter_alpha=0.0, action_ramp_steps=0)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试4: 角速度修复 + raw last_action
print("\n--- FIX 4: 角速度direct + raw last_action ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="direct", last_action_mode="raw",
                  action_filter_alpha=0.3, obs_filter_alpha=0.3, action_ramp_steps=50)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试5: 角速度修复 + raw last_action + 无滤波
print("\n--- FIX 5: 角速度direct + raw last_action + 无滤波无ramp ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="direct", last_action_mode="raw",
                  action_filter_alpha=0.0, obs_filter_alpha=0.0, action_ramp_steps=0)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试6: 只去掉action filter（保留obs filter和ramp）
print("\n--- FIX 6: 无action filter ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="double_rot", last_action_mode="filtered",
                  action_filter_alpha=0.0, obs_filter_alpha=0.3, action_ramp_steps=50)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

# 测试7: raw last_action + 无滤波无ramp（不修角速度）
print("\n--- FIX 7: raw last_action + 无滤波无ramp ---")
for cname, cvec in cmds.items():
    fwd = run_sim(cvec, angvel_mode="double_rot", last_action_mode="raw",
                  action_filter_alpha=0.0, obs_filter_alpha=0.0, action_ramp_steps=0)
    print(f"  {cname:6s} cmd={cvec}: fwd={fwd:+.4f}m")

print("\n" + "="*100)
print("分析：找出哪个组合让 FWD > STAND > BWD 且 STAND ≈ 0")
print("="*100)
