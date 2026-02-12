#!/usr/bin/env python3
"""对比 mj_objectVelocity vs 手动 world_to_body(qvel) 的差异，
然后测试用 mj_objectVelocity 获取速度后的 sim2sim 效果。

Go2 的部署代码用 mj_objectVelocity(flg_local=1) 直接获取 body frame 速度，
而 V4b 用 world_to_body(qvel[0:3]) + world_to_body(qvel[3:6])。
后者对角速度是 double rotation bug。
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


# ============================================================
# Part 1: 对比两种速度获取方式
# ============================================================
print("="*80)
print("Part 1: 对比 mj_objectVelocity vs world_to_body(qvel)")
print("="*80)

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)

base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
print(f"base_link body id = {base_body_id}")

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

# 给一些初始速度
d.qvel[0] = 0.1   # world X
d.qvel[1] = -0.2  # world Y (forward for V4)
d.qvel[2] = 0.05  # world Z
d.qvel[3] = 0.3   # body ang X
d.qvel[4] = 0.5   # body ang Y
d.qvel[5] = -0.2  # body ang Z

mujoco.mj_forward(m, d)

quat = d.qpos[3:7]

# 方法1: 手动 world_to_body
lin_vel_manual = world_to_body(d.qvel[0:3], quat)
ang_vel_manual = world_to_body(d.qvel[3:6], quat)  # double rotation!
ang_vel_direct = d.qvel[3:6].copy()  # 直接用（已是body frame）

# 方法2: mj_objectVelocity (flg_local=1)
vel6 = np.zeros(6)
mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_body_id, vel6, 1)
ang_vel_mj = vel6[0:3]
lin_vel_mj = vel6[3:6]

print(f"\n给定 qvel[0:3] (world lin) = {d.qvel[0:3]}")
print(f"给定 qvel[3:6] (body ang)  = {d.qvel[3:6]}")
print(f"\n线速度:")
print(f"  world_to_body(qvel[0:3])  = {lin_vel_manual}")
print(f"  mj_objectVelocity lin     = {lin_vel_mj}")
print(f"  差异                      = {lin_vel_manual - lin_vel_mj}")
print(f"\n角速度:")
print(f"  world_to_body(qvel[3:6])  = {ang_vel_manual}  (DOUBLE ROTATION!)")
print(f"  qvel[3:6] 直接用          = {ang_vel_direct}")
print(f"  mj_objectVelocity ang     = {ang_vel_mj}")
print(f"  差异(direct vs mj)        = {ang_vel_direct - ang_vel_mj}")
print(f"  差异(double_rot vs mj)    = {ang_vel_manual - ang_vel_mj}")


# ============================================================
# Part 2: 用 mj_objectVelocity 运行仿真测试
# ============================================================
print("\n" + "="*80)
print("Part 2: 用 mj_objectVelocity 运行仿真")
print("="*80)

def run_sim(cmd_vec, vel_mode="qvel_double_rot", action_filter_alpha=0.0, num_steps=500):
    """
    vel_mode:
      "qvel_double_rot" - 当前代码: world_to_body(qvel[0:3]) + world_to_body(qvel[3:6])
      "qvel_fixed"      - 修复角速度: world_to_body(qvel[0:3]) + qvel[3:6]直接用
      "mj_objvel"       - Go2方式: mj_objectVelocity(flg_local=1)
    """
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    m.opt.timestep = cfg["simulation_dt"]
    
    default_angles = np.array(cfg["default_angles"], dtype=np.float64)
    action_scale = cfg["action_scale"]
    control_dec = cfg["control_decimation"]
    obs_filter_alpha = cfg["obs_filter_alpha"]
    action_ramp_steps = cfg["action_ramp_steps"]
    
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    
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
            
            if vel_mode == "qvel_double_rot":
                lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
                omega = world_to_body(d.qvel[3:6].copy(), quat)
            elif vel_mode == "qvel_fixed":
                lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
                omega = d.qvel[3:6].copy()
            elif vel_mode == "mj_objvel":
                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base_body_id, vel6, 1)
                omega = vel6[0:3]
                lin_vel_b = vel6[3:6]
            
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
    ("当前代码(double_rot+filter=0)",  "qvel_double_rot", 0.0),
    ("修复角速度(qvel_fixed+filter=0)", "qvel_fixed",      0.0),
    ("Go2方式(mj_objvel+filter=0)",    "mj_objvel",       0.0),
    ("当前代码(double_rot+filter=0.3)", "qvel_double_rot", 0.3),
    ("Go2方式(mj_objvel+filter=0.3)",  "mj_objvel",       0.3),
]

for cname, vel_mode, af_alpha in configs:
    print(f"\n{'='*90}")
    print(f"  {cname}")
    print(f"{'='*90}")
    print(f"  {'命令':12s} | {'前进(m)':>8s} | {'横向(m)':>8s} | {'转向(°)':>8s} | {'高度(m)':>7s}")
    print(f"  {'-'*60}")
    
    for cmd_name, cmd_vec in cmds:
        fwd, lat, yaw, h = run_sim(cmd_vec, vel_mode=vel_mode, action_filter_alpha=af_alpha, num_steps=500)
        print(f"  {cmd_name:12s} | {fwd:+8.3f} | {lat:+8.3f} | {yaw:+8.1f} | {h:7.3f}")
