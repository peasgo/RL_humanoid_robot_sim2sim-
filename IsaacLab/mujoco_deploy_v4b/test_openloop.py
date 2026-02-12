#!/usr/bin/env python3
"""开环测试：用一个cmd的action序列驱动另一个cmd的仿真。
如果用BWD的action序列驱动仿真，机器人应该后退。
如果仍然前进，说明是物理执行的问题。
如果后退，说明是obs反馈导致策略输出错误。
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


def run_closedloop(cmd_vec, num_steps=500):
    """正常闭环运行，记录action序列"""
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
    
    init_pos = d.qpos[0:3].copy()
    init_yaw = quat_to_yaw(d.qpos[3:7])
    counter = 0
    
    action_seq = []
    
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
            
            with torch.no_grad():
                out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
            action16[:] = np.clip(out, -5.0, 5.0)
            action_seq.append(action16.copy())
            
            target[waist_mj] = default_angles[waist_mj]
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
    
    final_pos = d.qpos[0:3]
    fwd = -(final_pos[1] - init_pos[1])
    lat = final_pos[0] - init_pos[0]
    yaw = np.degrees(quat_to_yaw(d.qpos[3:7]) - init_yaw)
    return fwd, lat, yaw, np.array(action_seq)


def run_openloop(action_seq, num_steps=500):
    """开环运行：用预录的action序列驱动仿真"""
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
    
    isaac16 = ['LHIPp','RHIPp','LHIPy','RHIPy',
               'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
               'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']
    
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
    init_pos = d.qpos[0:3].copy()
    init_yaw = quat_to_yaw(d.qpos[3:7])
    counter = 0
    policy_step = 0
    
    actual_steps = min(num_steps, len(action_seq))
    
    for step in range(actual_steps * control_dec):
        d.ctrl[:] = target[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_dec == 0:
            if policy_step < len(action_seq):
                action16 = action_seq[policy_step]
                target[waist_mj] = default_angles[waist_mj]
                for i16 in range(16):
                    mj_idx = i16_to_mj[i16]
                    target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
                policy_step += 1
    
    final_pos = d.qpos[0:3]
    fwd = -(final_pos[1] - init_pos[1])
    lat = final_pos[0] - init_pos[0]
    yaw = np.degrees(quat_to_yaw(d.qpos[3:7]) - init_yaw)
    return fwd, lat, yaw


# 1. 闭环运行，记录action序列
print("="*80)
print("  第1步：闭环运行，记录action序列")
print("="*80)

fwd_f, lat_f, yaw_f, act_seq_fwd = run_closedloop([0.5, 0, 0])
print(f"  FWD 闭环: 前进={fwd_f:+.4f}m, 横向={lat_f:+.4f}m, 转向={yaw_f:+.1f}°")

fwd_b, lat_b, yaw_b, act_seq_bwd = run_closedloop([-0.5, 0, 0])
print(f"  BWD 闭环: 前进={fwd_b:+.4f}m, 横向={lat_b:+.4f}m, 转向={yaw_b:+.1f}°")

fwd_s, lat_s, yaw_s, act_seq_stand = run_closedloop([0, 0, 0])
print(f"  STAND闭环: 前进={fwd_s:+.4f}m, 横向={lat_s:+.4f}m, 转向={yaw_s:+.1f}°")

# 2. 开环运行：用FWD的action驱动
print(f"\n{'='*80}")
print("  第2步：开环运行 - 用不同cmd的action序列驱动仿真")
print("="*80)

fwd_ol, lat_ol, yaw_ol = run_openloop(act_seq_fwd)
print(f"  FWD action开环: 前进={fwd_ol:+.4f}m, 横向={lat_ol:+.4f}m, 转向={yaw_ol:+.1f}°")

bwd_ol, lat_ol2, yaw_ol2 = run_openloop(act_seq_bwd)
print(f"  BWD action开环: 前进={bwd_ol:+.4f}m, 横向={lat_ol2:+.4f}m, 转向={yaw_ol2:+.1f}°")

stand_ol, lat_ol3, yaw_ol3 = run_openloop(act_seq_stand)
print(f"  STAND action开环: 前进={stand_ol:+.4f}m, 横向={lat_ol3:+.4f}m, 转向={yaw_ol3:+.1f}°")

# 3. 分析action序列的差异
print(f"\n{'='*80}")
print("  第3步：分析action序列统计")
print("="*80)

isaac16 = ['LHIPp','RHIPp','LHIPy','RHIPy',
           'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
           'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']

print(f"\n  action均值对比 (步骤10-500):")
print(f"  {'关节':10s} | {'FWD':>8s} | {'BWD':>8s} | {'STAND':>8s} | {'FWD-BWD':>8s}")
print(f"  {'-'*55}")
for i in range(16):
    fwd_mean = np.mean(act_seq_fwd[10:, i])
    bwd_mean = np.mean(act_seq_bwd[10:, i])
    stand_mean = np.mean(act_seq_stand[10:, i])
    diff = fwd_mean - bwd_mean
    print(f"  {isaac16[i]:10s} | {fwd_mean:+8.4f} | {bwd_mean:+8.4f} | {stand_mean:+8.4f} | {diff:+8.4f}")

# 4. 检查action的对称性
print(f"\n{'='*80}")
print("  第4步：检查FWD和BWD的action对称性")
print("  如果策略正确，FWD和BWD的腿部action应该有明显的前后差异")
print("="*80)

# 腿部关节对：(左前, 右前) vs (左后, 右后)
# Isaac16: LHIPp(0), RHIPp(1), LHIPy(2), RHIPy(3), LSDp(4), RSDp(5), 
#          LKNEEp(6), RKNEEP(7), LSDy(8), RSDy(9), LANKLEp(10), RANKLEp(11)
# 前腿 = SD/ARM (4,5,8,9,12,13,14,15)
# 后腿 = HIP/KNEE/ANKLE (0,1,2,3,6,7,10,11)

front_legs = [4, 5, 8, 9, 12, 13, 14, 15]  # SD, ARM
rear_legs = [0, 1, 2, 3, 6, 7, 10, 11]     # HIP, KNEE, ANKLE

fwd_front_mean = np.mean(np.abs(act_seq_fwd[10:, front_legs]))
fwd_rear_mean = np.mean(np.abs(act_seq_fwd[10:, rear_legs]))
bwd_front_mean = np.mean(np.abs(act_seq_bwd[10:, front_legs]))
bwd_rear_mean = np.mean(np.abs(act_seq_bwd[10:, rear_legs]))

print(f"\n  FWD: 前腿|action|均值={fwd_front_mean:.4f}, 后腿|action|均值={fwd_rear_mean:.4f}")
print(f"  BWD: 前腿|action|均值={bwd_front_mean:.4f}, 后腿|action|均值={bwd_rear_mean:.4f}")

# 5. 关键测试：action L2 norm随时间的变化
print(f"\n{'='*80}")
print("  第5步：action L2 norm随时间变化")
print("="*80)

for step in [0, 5, 10, 20, 50, 100, 200, 499]:
    if step < len(act_seq_fwd):
        fwd_l2 = np.linalg.norm(act_seq_fwd[step])
        bwd_l2 = np.linalg.norm(act_seq_bwd[step])
        diff_l2 = np.linalg.norm(act_seq_fwd[step] - act_seq_bwd[step])
        print(f"  Step {step:3d}: FWD L2={fwd_l2:.4f}, BWD L2={bwd_l2:.4f}, FWD-BWD L2={diff_l2:.4f}")
