#!/usr/bin/env python3
"""系统性消融测试：逐个关闭部署中添加的额外处理，找出哪个导致前进偏差。

部署代码中相比训练多了以下处理：
1. obs_filter_alpha=0.3 (vel_only) — 训练时没有
2. action_ramp_steps=50 — 训练时没有
3. angular velocity double rotation — 训练时用IsaacLab的root_ang_vel_b
4. 初始穿透 — 训练时IsaacLab可能没有穿透

测试矩阵：
- BASELINE: 当前代码 (obs_filter=0.3, ramp=50, double_rot, h=0.22)
- NO_OBS_FILTER: obs_filter=0
- NO_RAMP: action_ramp=0
- NO_BOTH: obs_filter=0, ramp=0
- HIGHER_INIT: h=0.26 (无穿透)
- ALL_CLEAN: obs_filter=0, ramp=0, h=0.26
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=4, suppress=True, linewidth=150)

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

def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def run_sim(cmd_vec, obs_filter_alpha=0.3, action_ramp_steps=50, 
            init_height=0.22, action_filter_alpha=0.0, num_steps=500):
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
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target = default_angles.copy()
    action16 = np.zeros(16, dtype=np.float32)
    action16_prev = np.zeros(16, dtype=np.float32)
    obs = np.zeros(62, dtype=np.float32)
    prev_obs = np.zeros(62, dtype=np.float32)
    cmd = np.array(cmd_vec, dtype=np.float32)
    
    init_pos = d.qpos[0:3].copy()
    init_yaw = quat_to_yaw(d.qpos[3:7])
    counter = 0
    policy_step = 0
    
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
            
            if obs_filter_alpha > 0 and policy_step > 0:
                obs[0:6] = obs_filter_alpha * prev_obs[0:6] + (1-obs_filter_alpha) * obs[0:6]
                obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1-obs_filter_alpha) * obs[29:46]
            prev_obs[:] = obs
            
            with torch.no_grad():
                out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
            action16[:] = out
            action16 = np.clip(action16, -5.0, 5.0)
            
            if action_ramp_steps > 0 and policy_step < action_ramp_steps:
                action16 *= float(policy_step) / float(action_ramp_steps)
            
            if action_filter_alpha > 0:
                action16 = action_filter_alpha * action16_prev + (1-action_filter_alpha) * action16
            action16_prev[:] = action16
            
            policy_step += 1
            
            target[waist_mj] = default_angles[waist_mj]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
    
    final_pos = d.qpos[0:3]
    fwd = -(final_pos[1] - init_pos[1])
    lat = final_pos[0] - init_pos[0]
    yaw = np.degrees(quat_to_yaw(d.qpos[3:7]) - init_yaw)
    height = final_pos[2]
    return fwd, lat, yaw, height


cmds = [
    ("FWD 0.5",  [0.5, 0, 0]),
    ("FWD 0.3",  [0.3, 0, 0]),
    ("STAND",    [0, 0, 0]),
    ("BWD 0.3",  [-0.3, 0, 0]),
    ("LEFT 0.3", [0, 0.3, 0]),
    ("TURN_L",   [0, 0, 0.5]),
]

configs = [
    # name, obs_filter, ramp, height, action_filter
    ("BASELINE (filter=0.3,ramp=50,h=0.22)",     0.3, 50, 0.22, 0.0),
    ("NO_OBS_FILTER (filter=0,ramp=50,h=0.22)",  0.0, 50, 0.22, 0.0),
    ("NO_RAMP (filter=0.3,ramp=0,h=0.22)",       0.3,  0, 0.22, 0.0),
    ("NO_FILTER_NO_RAMP (filter=0,ramp=0,h=0.22)", 0.0, 0, 0.22, 0.0),
    ("HIGH_INIT (filter=0.3,ramp=50,h=0.26)",    0.3, 50, 0.26, 0.0),
    ("ALL_CLEAN (filter=0,ramp=0,h=0.26)",        0.0,  0, 0.26, 0.0),
    ("CLEAN+AF (filter=0,ramp=0,h=0.26,af=0.3)", 0.0,  0, 0.26, 0.3),
]

for cname, of_alpha, ramp, h, af_alpha in configs:
    print(f"\n{'='*90}")
    print(f"  {cname}")
    print(f"{'='*90}")
    print(f"  {'命令':12s} | {'前进(m)':>8s} | {'横向(m)':>8s} | {'转向(°)':>8s} | {'高度(m)':>7s}")
    print(f"  {'-'*60}")
    
    for cmd_name, cmd_vec in cmds:
        fwd, lat, yaw, height = run_sim(
            cmd_vec, 
            obs_filter_alpha=of_alpha, 
            action_ramp_steps=ramp,
            init_height=h,
            action_filter_alpha=af_alpha,
        )
        print(f"  {cmd_name:12s} | {fwd:+8.3f} | {lat:+8.3f} | {yaw:+8.1f} | {height:7.3f}")

# 计算方向性指标
print("\n" + "="*90)
print("方向性指标总结")
print("="*90)
print(f"  {'配置':45s} | {'FWD-BWD差':>10s} | {'STAND偏差':>10s} | {'LEFT横向':>10s} | {'TURN转向':>10s}")
print(f"  {'-'*95}")

for cname, of_alpha, ramp, h, af_alpha in configs:
    results = {}
    for cmd_name, cmd_vec in cmds:
        fwd, lat, yaw, height = run_sim(
            cmd_vec, 
            obs_filter_alpha=of_alpha, 
            action_ramp_steps=ramp,
            init_height=h,
            action_filter_alpha=af_alpha,
        )
        results[cmd_name] = (fwd, lat, yaw)
    
    fwd_bwd_diff = results["FWD 0.3"][0] - results["BWD 0.3"][0]
    stand_bias = abs(results["STAND"][0])
    left_lat = results["LEFT 0.3"][1]
    turn_yaw = results["TURN_L"][2]
    
    print(f"  {cname:45s} | {fwd_bwd_diff:+10.3f} | {stand_bias:10.3f} | {left_lat:+10.3f} | {turn_yaw:+10.1f}")
