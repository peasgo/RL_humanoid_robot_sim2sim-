#!/usr/bin/env python3
"""确认角速度bug并测试修复效果。

关键发现：MuJoCo qvel[3:6] 对于free joint是一种特殊的角速度表示，
不是简单的世界坐标系角速度。world_to_body(qvel[3:6]) 给出错误的body frame角速度。
应该用 mj_objectVelocity(flg_local=1) 获取正确的body frame速度。

但线速度 qvel[0:3] 是世界坐标系的，world_to_body(qvel[0:3]) 是正确的。
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

def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def run_sim(cmd_vec, use_objvel=False, num_steps=500):
    """运行仿真，可选择使用mj_objectVelocity"""
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
    
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    
    target = default_angles.copy()
    action16 = np.zeros(16, dtype=np.float32)
    obs = np.zeros(62, dtype=np.float32)
    cmd = np.array(cmd_vec, dtype=np.float32)
    
    init_pos = d.qpos[0:3].copy()
    init_yaw = quat_to_yaw(d.qpos[3:7])
    counter = 0
    
    for step in range(num_steps * control_dec):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_dec == 0:
            quat = d.qpos[3:7]
            
            if use_objvel:
                # 使用mj_objectVelocity获取正确的body frame速度
                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 1)  # flg_local=1
                lin_vel_b = vel6[3:6]  # body frame线速度
                omega_b = vel6[0:3]    # body frame角速度
            else:
                # 原始方法：world_to_body(qvel)
                lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
                omega_b = world_to_body(d.qvel[3:6].copy(), quat)
            
            qj_i17 = d.qpos[7:].copy()[i17_to_mj]
            dqj_i17 = d.qvel[6:].copy()[i17_to_mj]
            def_i17 = default_angles[i17_to_mj]
            grav = get_gravity(quat)
            
            obs[0:3] = v4_remap_lin(lin_vel_b)
            obs[3:6] = v4_remap_ang(omega_b)
            obs[6:9] = v4_remap_grav(grav)
            obs[9:12] = cmd
            obs[12:29] = (qj_i17 - def_i17).astype(np.float32)
            obs[29:46] = dqj_i17.astype(np.float32)
            obs[46:62] = action16.astype(np.float32)
            
            with torch.no_grad():
                out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
            action16[:] = np.clip(out, -5.0, 5.0)
            
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
    ("FWD 0.5",   [0.5, 0, 0]),
    ("FWD 0.3",   [0.3, 0, 0]),
    ("STAND",     [0, 0, 0]),
    ("BWD 0.3",   [-0.3, 0, 0]),
    ("BWD 0.5",   [-0.5, 0, 0]),
    ("LEFT 0.3",  [0, 0.3, 0]),
    ("RIGHT 0.3", [0, -0.3, 0]),
    ("TURN_L",    [0, 0, 0.5]),
    ("TURN_R",    [0, 0, -0.5]),
]

for method_name, use_objvel in [("原始(world_to_body)", False), ("修复(mj_objectVelocity)", True)]:
    print(f"\n{'='*90}")
    print(f"  {method_name}")
    print(f"{'='*90}")
    print(f"  {'命令':12s} | {'前进(m)':>8s} | {'横向(m)':>8s} | {'转向(°)':>8s} | {'高度(m)':>7s}")
    print(f"  {'-'*60}")
    
    for cmd_name, cmd_vec in cmds:
        fwd, lat, yaw, height = run_sim(cmd_vec, use_objvel=use_objvel)
        print(f"  {cmd_name:12s} | {fwd:+8.3f} | {lat:+8.3f} | {yaw:+8.1f} | {height:7.3f}")

# 方向性指标
print(f"\n{'='*90}")
print(f"  方向性指标对比")
print(f"{'='*90}")

for method_name, use_objvel in [("原始", False), ("修复", True)]:
    results = {}
    for cmd_name, cmd_vec in cmds:
        fwd, lat, yaw, height = run_sim(cmd_vec, use_objvel=use_objvel)
        results[cmd_name] = (fwd, lat, yaw, height)
    
    fwd_bwd_diff = results["FWD 0.3"][0] - results["BWD 0.3"][0]
    stand_bias = results["STAND"][0]
    left_lat = results["LEFT 0.3"][1]
    turn_yaw = results["TURN_L"][2]
    bwd_actual = results["BWD 0.3"][0]
    
    print(f"  {method_name:8s}: FWD-BWD差={fwd_bwd_diff:+.3f}m, STAND偏差={stand_bias:+.3f}m, "
          f"LEFT横向={left_lat:+.3f}m, TURN转向={turn_yaw:+.1f}°, BWD实际={bwd_actual:+.3f}m")
