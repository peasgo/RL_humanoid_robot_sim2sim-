#!/usr/bin/env python3
"""第三轮诊断：用最佳配置(无action_filter)测试所有方向，记录XY位移"""
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


def run_sim(cmd_vec, num_steps=500):
    """无action_filter配置，返回(fwd, lat, yaw_deg)"""
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
    prev_obs = np.zeros(62, dtype=np.float32)
    cmd = np.array(cmd_vec, dtype=np.float32)
    
    init_pos = d.qpos[0:3].copy()
    init_quat = d.qpos[3:7].copy()
    counter = 0
    policy_step = 0
    
    # 无action filter, 有obs filter, 有ramp
    obs_filter_alpha = 0.3
    action_ramp_steps = 50
    
    for step in range(num_steps * control_dec):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_dec == 0:
            quat = d.qpos[3:7]
            lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
            omega = world_to_body(d.qvel[3:6].copy(), quat)  # 保持double_rot（当前代码）
            
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
            obs[46:62] = action16.astype(np.float32)  # 无filter时等于raw
            
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
            
            policy_step += 1
            
            target[waist_mj] = default_angles[waist_mj]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
    
    final_pos = d.qpos[0:3]
    final_quat = d.qpos[3:7]
    
    # V4: forward = -Y, lateral = +X
    fwd = -(final_pos[1] - init_pos[1])
    lat = final_pos[0] - init_pos[0]
    
    # 计算yaw变化（绕世界Z轴）
    # 从四元数提取yaw
    def quat_to_yaw(q):
        w, x, y, z = q
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    yaw_change = np.degrees(quat_to_yaw(final_quat) - quat_to_yaw(init_quat))
    
    return fwd, lat, yaw_change


print("="*100)
print("无action_filter配置 - 全方向测试")
print("V4坐标系: forward=-Y, lateral=+X, yaw=绕Z轴")
print("="*100)

cmds = [
    ("FWD 0.5",  [0.5, 0, 0]),
    ("FWD 0.3",  [0.3, 0, 0]),
    ("FWD 0.1",  [0.1, 0, 0]),
    ("STAND",    [0, 0, 0]),
    ("BWD 0.1",  [-0.1, 0, 0]),
    ("BWD 0.3",  [-0.3, 0, 0]),
    ("LEFT 0.3", [0, 0.3, 0]),
    ("RIGHT 0.3",[0, -0.3, 0]),
    ("TURN_L",   [0, 0, 0.5]),
    ("TURN_R",   [0, 0, -0.5]),
    ("FWD+LEFT", [0.3, 0.3, 0]),
    ("FWD+TURN", [0.3, 0, 0.5]),
]

print(f"\n{'命令':15s} | {'cmd':20s} | {'前进(m)':>8s} | {'横向(m)':>8s} | {'转向(°)':>8s}")
print("-"*80)

for cname, cvec in cmds:
    fwd, lat, yaw = run_sim(cvec, num_steps=500)
    print(f"{cname:15s} | {str(cvec):20s} | {fwd:+8.3f} | {lat:+8.3f} | {yaw:+8.1f}")

print("\n" + "="*100)
print("期望行为:")
print("  FWD: 前进>0, 横向≈0")
print("  BWD: 前进<0, 横向≈0")
print("  LEFT: 横向>0, 前进≈0")
print("  TURN_L: 转向>0, 前进≈0")
print("  STAND: 全部≈0")
print("="*100)
