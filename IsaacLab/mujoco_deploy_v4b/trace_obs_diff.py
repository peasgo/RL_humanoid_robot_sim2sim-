#!/usr/bin/env python3
"""深度对比：FWD vs BWD 的obs时间序列，找出策略为什么不区分方向。
记录前100个policy step的完整obs，逐分量对比。
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=5, suppress=True, linewidth=200)

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


def run_and_record(cmd_vec, num_steps=100):
    """运行仿真并记录每步的obs和action"""
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
    d.qpos[2] = 0.22  # 使用默认高度
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target = default_angles.copy()
    action16 = np.zeros(16, dtype=np.float32)
    obs = np.zeros(62, dtype=np.float32)
    cmd = np.array(cmd_vec, dtype=np.float32)
    
    obs_history = []
    action_history = []
    pos_history = []
    
    counter = 0
    
    for step in range(num_steps * control_dec):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_dec == 0:
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
            
            obs_history.append(obs.copy())
            pos_history.append(d.qpos[0:3].copy())
            
            with torch.no_grad():
                out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
            action16[:] = np.clip(out, -5.0, 5.0)
            action_history.append(action16.copy())
            
            target[waist_mj] = default_angles[waist_mj]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
    
    return np.array(obs_history), np.array(action_history), np.array(pos_history)


# 运行FWD和BWD
print("运行 FWD cmd=[0.5, 0, 0]...")
obs_fwd, act_fwd, pos_fwd = run_and_record([0.5, 0, 0])
print("运行 BWD cmd=[-0.5, 0, 0]...")
obs_bwd, act_bwd, pos_bwd = run_and_record([-0.5, 0, 0])
print("运行 STAND cmd=[0, 0, 0]...")
obs_stand, act_stand, pos_stand = run_and_record([0, 0, 0])

obs_names = (
    ['lin_vel_fwd', 'lin_vel_lat', 'lin_vel_up'] +
    ['ang_vel_roll', 'ang_vel_pitch', 'ang_vel_yaw'] +
    ['grav_fwd', 'grav_lat', 'grav_up'] +
    ['cmd_fwd', 'cmd_lat', 'cmd_yaw'] +
    [f'qj_{i}' for i in range(17)] +
    [f'dqj_{i}' for i in range(17)] +
    [f'act_{i}' for i in range(16)]
)

print("\n" + "="*120)
print("  逐步对比 FWD vs BWD 的obs差异 (前20步)")
print("="*120)

for step in range(min(20, len(obs_fwd))):
    diff = obs_fwd[step] - obs_bwd[step]
    # 找出差异最大的分量
    top_idx = np.argsort(np.abs(diff))[-5:][::-1]
    
    print(f"\n--- Step {step} ---")
    print(f"  FWD pos: ({pos_fwd[step][0]:+.4f}, {pos_fwd[step][1]:+.4f}, {pos_fwd[step][2]:.4f})")
    print(f"  BWD pos: ({pos_bwd[step][0]:+.4f}, {pos_bwd[step][1]:+.4f}, {pos_bwd[step][2]:.4f})")
    print(f"  obs差异最大的5个分量:")
    for idx in top_idx:
        print(f"    [{idx:2d}] {obs_names[idx]:15s}: FWD={obs_fwd[step][idx]:+.5f}  BWD={obs_bwd[step][idx]:+.5f}  diff={diff[idx]:+.5f}")
    
    act_diff = act_fwd[step] - act_bwd[step]
    act_top = np.argsort(np.abs(act_diff))[-3:][::-1]
    print(f"  action差异最大的3个:")
    for idx in act_top:
        print(f"    [{idx:2d}]: FWD={act_fwd[step][idx]:+.5f}  BWD={act_bwd[step][idx]:+.5f}  diff={act_diff[idx]:+.5f}")

print("\n" + "="*120)
print("  obs分量的时间平均差异 (FWD - BWD, 步骤10-100)")
print("="*120)

mean_diff = np.mean(obs_fwd[10:] - obs_bwd[10:], axis=0)
mean_abs_diff = np.mean(np.abs(obs_fwd[10:] - obs_bwd[10:]), axis=0)

# 按绝对差异排序
sorted_idx = np.argsort(mean_abs_diff)[::-1]
print(f"\n  {'idx':>4s} {'名称':15s} {'平均差异':>10s} {'|平均差异|':>10s} {'FWD均值':>10s} {'BWD均值':>10s}")
print(f"  {'-'*70}")
for i, idx in enumerate(sorted_idx[:20]):
    fwd_mean = np.mean(obs_fwd[10:, idx])
    bwd_mean = np.mean(obs_bwd[10:, idx])
    print(f"  {idx:4d} {obs_names[idx]:15s} {mean_diff[idx]:+10.5f} {mean_abs_diff[idx]:10.5f} {fwd_mean:+10.5f} {bwd_mean:+10.5f}")

print("\n" + "="*120)
print("  action分量的时间平均差异 (FWD - BWD, 步骤10-100)")
print("="*120)

isaac16 = ['LHIPp','RHIPp','LHIPy','RHIPy',
           'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
           'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']

act_mean_diff = np.mean(act_fwd[10:] - act_bwd[10:], axis=0)
act_mean_abs = np.mean(np.abs(act_fwd[10:] - act_bwd[10:]), axis=0)
act_sorted = np.argsort(act_mean_abs)[::-1]

print(f"\n  {'idx':>4s} {'关节':10s} {'平均差异':>10s} {'|平均差异|':>10s} {'FWD均值':>10s} {'BWD均值':>10s}")
print(f"  {'-'*60}")
for idx in act_sorted:
    fwd_mean = np.mean(act_fwd[10:, idx])
    bwd_mean = np.mean(act_bwd[10:, idx])
    print(f"  {idx:4d} {isaac16[idx]:10s} {act_mean_diff[idx]:+10.5f} {act_mean_abs[idx]:10.5f} {fwd_mean:+10.5f} {bwd_mean:+10.5f}")

# 关键：检查cmd是否真的在obs中
print("\n" + "="*120)
print("  验证cmd在obs中的值 (步骤0, 5, 10, 50)")
print("="*120)
for step in [0, 5, 10, 50]:
    if step < len(obs_fwd):
        print(f"  Step {step}: FWD cmd_obs={obs_fwd[step][9:12]}  BWD cmd_obs={obs_bwd[step][9:12]}  STAND cmd_obs={obs_stand[step][9:12]}")

# 关键：检查FWD和BWD的lin_vel obs是否有差异
print("\n" + "="*120)
print("  lin_vel obs对比 (步骤0-20)")
print("="*120)
for step in range(min(20, len(obs_fwd))):
    print(f"  Step {step:2d}: FWD lin_vel={obs_fwd[step][0:3]}  BWD lin_vel={obs_bwd[step][0:3]}  STAND lin_vel={obs_stand[step][0:3]}")

# 最终位置对比
print("\n" + "="*120)
print("  最终位置 (步骤100)")
print("="*120)
fwd_disp = -(pos_fwd[-1][1] - pos_fwd[0][1])
bwd_disp = -(pos_bwd[-1][1] - pos_bwd[0][1])
stand_disp = -(pos_stand[-1][1] - pos_stand[0][1])
print(f"  FWD:   前进={fwd_disp:+.4f}m")
print(f"  BWD:   前进={bwd_disp:+.4f}m")
print(f"  STAND: 前进={stand_disp:+.4f}m")
