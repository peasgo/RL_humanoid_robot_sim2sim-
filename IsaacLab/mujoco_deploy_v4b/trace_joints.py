#!/usr/bin/env python3
"""精确诊断：检查action到关节角度的映射是否正确。
对比FWD和BWD的target_dof_pos，看步态差异。
同时检查MuJoCo中关节实际角度是否跟踪target。
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=4, suppress=True, linewidth=200)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

policy = torch.jit.load(cfg["policy_path"], map_location="cpu")
policy.eval()

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_body(v, q): return quat_to_rotmat(q).T @ v
def get_gravity(q):
    w, x, y, z = q
    return np.array([-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))])
def v4_remap_lin(v): return np.array([v[2], v[0], v[1]])
def v4_remap_ang(v): return np.array([v[0], v[2], v[1]])
def v4_remap_grav(v): return np.array([v[2], v[0], v[1]])


def run_and_trace(cmd_vec, num_steps=100):
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    m.opt.timestep = cfg["simulation_dt"]
    
    default_angles = np.array(cfg["default_angles"], dtype=np.float64)
    action_scale = cfg["action_scale"]
    control_dec = cfg["control_decimation"]
    
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
    
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target = default_angles.copy()
    action16 = np.zeros(16, dtype=np.float32)
    obs = np.zeros(62, dtype=np.float32)
    cmd = np.array(cmd_vec, dtype=np.float32)
    
    counter = 0
    
    # 记录：target角度(MuJoCo order), 实际角度(MuJoCo order), action(Isaac16 order)
    target_history = []
    actual_history = []
    action_history = []
    torque_history = []
    
    for step in range(num_steps * control_dec):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_dec == 0:
            # 记录当前状态
            target_history.append(target.copy())
            actual_history.append(d.qpos[7:].copy())
            torque_history.append(d.qfrc_actuator[6:].copy())  # 关节力矩
            
            quat = d.qpos[3:7]
            lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
            omega = world_to_body(d.qvel[3:6].copy(), quat)
            
            qj_i17 = d.qpos[7:].copy()[i17_to_mj]
            dqj_i17 = d.qvel[6:].copy()[i17_to_mj]
            def_i17 = default_angles[i17_to_mj]
            grav = get_gravity(quat)
            
            obs[0:3] = v4_remap_lin(lin_vel_b)
            obs[3:6] = v4_remap_ang(omega)
            obs[6:9] = v4_remap_grav(grav)
            obs[9:12] = cmd
            obs[12:29] = (qj_i17 - def_i17).astype(np.float32)
            obs[29:46] = dqj_i17.astype(np.float32)
            obs[46:62] = action16.astype(np.float32)
            
            with torch.no_grad():
                out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
            action16[:] = np.clip(out, -5.0, 5.0)
            action_history.append(action16.copy())
            
            target[waist_mj] = default_angles[waist_mj]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
    
    return (np.array(target_history), np.array(actual_history), 
            np.array(action_history), np.array(torque_history), mj_names)


# 运行
print("运行 FWD...")
tgt_fwd, act_fwd, a16_fwd, trq_fwd, mj_names = run_and_trace([0.5, 0, 0])
print("运行 BWD...")
tgt_bwd, act_bwd, a16_bwd, trq_bwd, _ = run_and_trace([-0.5, 0, 0])
print("运行 STAND...")
tgt_stand, act_stand, a16_stand, trq_stand, _ = run_and_trace([0, 0, 0])

# 关键：检查target和actual的跟踪误差
print("\n" + "="*100)
print("  关节跟踪误差 (target - actual) 的均值和最大值 (步骤10-100)")
print("="*100)

print(f"\n  {'关节':10s} | {'FWD误差均值':>12s} | {'BWD误差均值':>12s} | {'FWD误差max':>12s} | {'BWD误差max':>12s}")
print(f"  {'-'*65}")
for i, jn in enumerate(mj_names):
    fwd_err = tgt_fwd[10:, i] - act_fwd[10:, i]
    bwd_err = tgt_bwd[10:, i] - act_bwd[10:, i]
    print(f"  {jn:10s} | {np.mean(fwd_err):+12.5f} | {np.mean(bwd_err):+12.5f} | {np.max(np.abs(fwd_err)):12.5f} | {np.max(np.abs(bwd_err)):12.5f}")

# 关键：检查target角度的差异
print("\n" + "="*100)
print("  target角度差异 (FWD - BWD) 的均值 (步骤10-100)")
print("="*100)

print(f"\n  {'关节':10s} | {'FWD target':>12s} | {'BWD target':>12s} | {'差异':>12s} | {'default':>12s}")
print(f"  {'-'*65}")
default_angles = np.array(cfg["default_angles"], dtype=np.float64)
for i, jn in enumerate(mj_names):
    fwd_mean = np.mean(tgt_fwd[10:, i])
    bwd_mean = np.mean(tgt_bwd[10:, i])
    diff = fwd_mean - bwd_mean
    print(f"  {jn:10s} | {fwd_mean:+12.5f} | {bwd_mean:+12.5f} | {diff:+12.5f} | {default_angles[i]:+12.5f}")

# 关键：检查力矩
print("\n" + "="*100)
print("  关节力矩均值 (步骤10-100)")
print("="*100)

print(f"\n  {'关节':10s} | {'FWD力矩':>12s} | {'BWD力矩':>12s} | {'差异':>12s}")
print(f"  {'-'*50}")
for i, jn in enumerate(mj_names):
    fwd_mean = np.mean(trq_fwd[10:, i])
    bwd_mean = np.mean(trq_bwd[10:, i])
    diff = fwd_mean - bwd_mean
    print(f"  {jn:10s} | {fwd_mean:+12.3f} | {bwd_mean:+12.3f} | {diff:+12.3f}")

# 最关键：检查action到target的映射是否正确
print("\n" + "="*100)
print("  验证action到target映射 (步骤20)")
print("="*100)

isaac16 = ['LHIPp','RHIPp','LHIPy','RHIPy',
           'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
           'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']

step = 20
action_scale = cfg["action_scale"]
print(f"\n  action_scale = {action_scale}")
print(f"\n  {'Isaac16关节':10s} | {'action':>8s} | {'expected_target':>15s} | {'actual_target':>15s} | {'match':>6s}")
print(f"  {'-'*70}")

i16_to_mj = []
for jn in isaac16:
    i16_to_mj.append(mj_names.index(jn))

for i16 in range(16):
    mj_idx = i16_to_mj[i16]
    action = a16_fwd[step][i16]
    expected = action * action_scale + default_angles[mj_idx]
    actual = tgt_fwd[step+1][mj_idx]  # target在下一步生效
    match = "✓" if abs(expected - actual) < 1e-5 else "✗"
    print(f"  {isaac16[i16]:10s} | {action:+8.4f} | {expected:+15.5f} | {actual:+15.5f} | {match:>6s}")
