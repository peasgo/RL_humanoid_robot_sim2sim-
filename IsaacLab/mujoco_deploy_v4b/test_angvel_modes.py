#!/usr/bin/env python3
"""精确测试：分别修复线速度和角速度，找出哪个是关键。

MuJoCo free joint qvel:
- qvel[0:3]: 世界坐标系线速度 → world_to_body()正确
- qvel[3:6]: 文档说是"local frame angular velocity" → 已经是body frame！
  world_to_body()会double rotate，是BUG

但verify_coords.py显示：
- qvel[3]=1.0时，world_to_body=[1,0,0], mj_objvel_body=[1,0,0] → X轴一致
- qvel[4]=1.0时，world_to_body=[0,0,-1], mj_objvel_body=[0,1,0] → 不一致！
- qvel[5]=1.0时，world_to_body=[0,1,0], mj_objvel_body=[0,0,1] → 不一致！

对于V4的初始quat=[0.7071,0.7071,0,0]（绕X轴90°），
R^T的效果是交换Y和Z并翻转一个符号。
所以world_to_body(qvel[3:6])实际上是把body frame的Y/Z分量交换了。

这意味着V4 remap后：
- 正确的ang_vel_body = [wx, wy, wz]
- V4 remap: [wx, wz, wy]
- 错误的(double rot): world_to_body([wx,wy,wz]) = [wx, wz, -wy]
- V4 remap错误: [wx, -wy, wz]

所以错误的结果是wy被取反了！wy对应的是V4的pitch（绕前进轴的俯仰）。
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


def run_sim(cmd_vec, ang_vel_mode="double_rot", num_steps=500):
    """
    ang_vel_mode:
    - "double_rot": 原始bug，world_to_body(qvel[3:6])
    - "direct": 直接用qvel[3:6]作为body frame角速度
    - "objvel": 用mj_objectVelocity获取body frame角速度
    """
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
            
            # 线速度始终用world_to_body（已验证正确）
            lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
            
            # 角速度根据模式选择
            if ang_vel_mode == "double_rot":
                omega_b = world_to_body(d.qvel[3:6].copy(), quat)
            elif ang_vel_mode == "direct":
                omega_b = d.qvel[3:6].copy()
            elif ang_vel_mode == "objvel":
                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 1)
                omega_b = vel6[0:3]
            
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
    ("TURN_L",    [0, 0, 0.5]),
    ("TURN_R",    [0, 0, -0.5]),
]

modes = [
    ("A: double_rot(原始bug)", "double_rot"),
    ("B: direct(qvel直接用)", "direct"),
    ("C: objvel(mj_objectVelocity)", "objvel"),
]

for mode_name, mode in modes:
    print(f"\n{'='*90}")
    print(f"  {mode_name}")
    print(f"{'='*90}")
    print(f"  {'命令':12s} | {'前进(m)':>8s} | {'横向(m)':>8s} | {'转向(°)':>8s} | {'高度(m)':>7s}")
    print(f"  {'-'*60}")
    
    for cmd_name, cmd_vec in cmds:
        fwd, lat, yaw, height = run_sim(cmd_vec, ang_vel_mode=mode)
        print(f"  {cmd_name:12s} | {fwd:+8.3f} | {lat:+8.3f} | {yaw:+8.1f} | {height:7.3f}")

# 方向性指标
print(f"\n{'='*90}")
print(f"  方向性指标对比")
print(f"{'='*90}")

for mode_name, mode in modes:
    results = {}
    for cmd_name, cmd_vec in cmds:
        results[cmd_name] = run_sim(cmd_vec, ang_vel_mode=mode)
    
    fwd_bwd = results["FWD 0.3"][0] - results["BWD 0.3"][0]
    stand = results["STAND"][0]
    left = results["LEFT 0.3"][1]
    turn = results["TURN_L"][2]
    bwd = results["BWD 0.3"][0]
    
    print(f"  {mode_name:35s}: FWD-BWD={fwd_bwd:+.3f}m, STAND={stand:+.3f}m, LEFT_lat={left:+.3f}m, TURN={turn:+.1f}°, BWD={bwd:+.3f}m")
