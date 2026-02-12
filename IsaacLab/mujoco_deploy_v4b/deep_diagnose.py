#!/usr/bin/env python3
"""深度诊断：逐步对比MuJoCo部署代码的obs构建与预期值。

关键问题：所有速度获取方式都有前进偏差，说明问题可能不在速度获取，
而在obs的其他部分（joint_pos, joint_vel, gravity等）。

这个脚本会：
1. 在初始姿态（零速度）下构建obs，检查每个分量是否合理
2. 运行几步后，打印完整obs，检查是否有异常值
3. 对比不同cmd下的obs差异，看cmd是否正确传入
4. 检查action到target_dof_pos的映射是否正确
"""
import numpy as np
import mujoco
import torch
import yaml
import os

np.set_printoptions(precision=6, suppress=True, linewidth=150)

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

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
m.opt.timestep = cfg["simulation_dt"]
default_angles = np.array(cfg["default_angles"], dtype=np.float64)

# 获取关节名
mj_names = []
for jid in range(m.njnt):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_names.append(jn)

print(f"MuJoCo关节顺序 ({len(mj_names)}): {mj_names}")

isaac17 = ['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2',
           'LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy',
           'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']
isaac16 = [j for j in isaac17 if j != 'Waist_2']

i17_to_mj = np.array([mj_names.index(j) for j in isaac17])
i16_to_mj = np.array([mj_names.index(j) for j in isaac16])
waist_mj = mj_names.index('Waist_2')

# actuator mapping
act_to_jnt = []
for i in range(m.nu):
    jid = m.actuator_trnid[i, 0]
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    act_to_jnt.append(mj_names.index(jn))
act_to_jnt = np.array(act_to_jnt)

print(f"\nIsaac17顺序: {isaac17}")
print(f"Isaac16顺序: {isaac16}")
print(f"i17_to_mj: {i17_to_mj}")
print(f"i16_to_mj: {i16_to_mj}")

# ============================================================
# Test 1: 初始姿态下的obs
# ============================================================
print("\n" + "="*80)
print("Test 1: 初始姿态下的obs（零速度，默认关节角）")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0
mujoco.mj_forward(m, d)

quat = d.qpos[3:7]
lin_vel_b = world_to_body(d.qvel[0:3], quat)
omega = world_to_body(d.qvel[3:6], quat)
grav = get_gravity(quat)

print(f"\nquat = {quat}")
print(f"body frame lin_vel = {lin_vel_b}")
print(f"body frame ang_vel = {omega}")
print(f"gravity (body) = {grav}")
print(f"  -> 预期: grav_x≈0, grav_y≈-1, grav_z≈0 (V4站立时重力沿body -Y)")

print(f"\nV4 remap:")
print(f"  lin_vel_obs = {v4_remap_lin(lin_vel_b)}")
print(f"  ang_vel_obs = {v4_remap_ang(omega)}")
print(f"  gravity_obs = {v4_remap_grav(grav)}")
print(f"  -> 预期: gravity_obs ≈ [0, 0, -1] (remap后: [gz, gx, gy] = [0, 0, -1])")

qj_mj = d.qpos[7:]
qj_i17 = qj_mj[i17_to_mj]
def_i17 = default_angles[i17_to_mj]
qj_rel = qj_i17 - def_i17

print(f"\n关节位置 (MuJoCo顺序):")
for i, jn in enumerate(mj_names):
    print(f"  MJ[{i:2d}] {jn:12s}: pos={qj_mj[i]:+.4f}  default={default_angles[i]:+.4f}  rel={qj_mj[i]-default_angles[i]:+.6f}")

print(f"\n关节位置 (Isaac17顺序):")
for i, jn in enumerate(isaac17):
    mj_idx = i17_to_mj[i]
    print(f"  I17[{i:2d}] {jn:12s} (MJ[{mj_idx:2d}]): pos={qj_i17[i]:+.4f}  default={def_i17[i]:+.4f}  rel={qj_rel[i]:+.6f}")

print(f"\njoint_pos_rel 应该全为0: max_abs = {np.max(np.abs(qj_rel)):.8f}")

# ============================================================
# Test 2: 运行10步后的obs
# ============================================================
print("\n" + "="*80)
print("Test 2: 运行10个policy步后的obs（cmd=[0,0,0]）")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

target = default_angles.copy()
action16 = np.zeros(16, dtype=np.float32)
obs = np.zeros(62, dtype=np.float32)
prev_obs = np.zeros(62, dtype=np.float32)
cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
action_scale = cfg["action_scale"]
control_dec = cfg["control_decimation"]
obs_filter_alpha = cfg["obs_filter_alpha"]
action_ramp_steps = cfg["action_ramp_steps"]

counter = 0
policy_step = 0

for step in range(10 * control_dec):
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
        
        if policy_step < 3 or policy_step == 9:
            print(f"\n--- Policy step {policy_step} ---")
            print(f"  pos = {d.qpos[0:3]}")
            print(f"  quat = {quat}")
            print(f"  obs[0:3]  lin_vel  = {obs[0:3]}")
            print(f"  obs[3:6]  ang_vel  = {obs[3:6]}")
            print(f"  obs[6:9]  gravity  = {obs[6:9]}")
            print(f"  obs[9:12] cmd      = {obs[9:12]}")
            print(f"  obs[12:29] qj_rel  = {obs[12:29]}")
            print(f"  obs[29:46] dqj     = {obs[29:46]}")
            print(f"  obs[46:62] last_act = {obs[46:62]}")
        
        with torch.no_grad():
            out = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
        
        action16[:] = out
        action16 = np.clip(action16, -5.0, 5.0)
        
        if action_ramp_steps > 0 and policy_step < action_ramp_steps:
            action16 *= float(policy_step) / float(action_ramp_steps)
        
        if policy_step < 3 or policy_step == 9:
            print(f"  raw action = {out}")
            print(f"  ramped action = {action16}")
            
            # 显示action对应的关节目标
            print(f"  action -> target:")
            for i16 in range(16):
                mj_idx = i16_to_mj[i16]
                tgt = action16[i16] * action_scale + default_angles[mj_idx]
                print(f"    I16[{i16:2d}] {isaac16[i16]:12s} (MJ[{mj_idx:2d}]): "
                      f"act={action16[i16]:+.4f} * {action_scale} + {default_angles[mj_idx]:+.4f} = {tgt:+.4f}")
        
        policy_step += 1
        
        target[waist_mj] = default_angles[waist_mj]
        for i16 in range(16):
            mj_idx = i16_to_mj[i16]
            target[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]


# ============================================================
# Test 3: 检查action对称性
# ============================================================
print("\n" + "="*80)
print("Test 3: 零obs下的action对称性检查")
print("="*80)

# 完全零obs
zero_obs = np.zeros(62, dtype=np.float32)
# 设置正确的gravity
zero_obs[6:9] = v4_remap_grav(get_gravity([0.70710678, 0.70710678, 0.0, 0.0]))

with torch.no_grad():
    zero_action = policy(torch.from_numpy(zero_obs).unsqueeze(0)).numpy().squeeze()

print(f"gravity_obs = {zero_obs[6:9]}")
print(f"action (16) = {zero_action}")
print(f"\n对称性检查 (左右对应关节的action差异):")
# Isaac16: LHIPp(0), RHIPp(1), LHIPy(2), RHIPy(3), LSDp(4), RSDp(5), 
#          LKNEEp(6), RKNEEP(7), LSDy(8), RSDy(9), LANKLEp(10), RANKLEp(11),
#          LARMp(12), RARMp(13), LARMAp(14), RARMAP(15)
pairs = [(0,1,"HIPp"), (2,3,"HIPy"), (4,5,"SDp"), (6,7,"KNEEp"), 
         (8,9,"SDy"), (10,11,"ANKLEp"), (12,13,"ARMp"), (14,15,"ARMAp")]
for l, r, name in pairs:
    print(f"  {isaac16[l]:12s}[{l:2d}]={zero_action[l]:+.4f}  vs  "
          f"{isaac16[r]:12s}[{r:2d}]={zero_action[r]:+.4f}  diff={zero_action[l]-zero_action[r]:+.6f}")


# ============================================================
# Test 4: 不同cmd下的action差异
# ============================================================
print("\n" + "="*80)
print("Test 4: 不同cmd下的action差异（零状态obs）")
print("="*80)

cmds_test = [
    ("FWD 0.5",  [0.5, 0, 0]),
    ("BWD 0.3",  [-0.3, 0, 0]),
    ("LEFT 0.3", [0, 0.3, 0]),
    ("TURN_L",   [0, 0, 0.5]),
]

base_obs = np.zeros(62, dtype=np.float32)
base_obs[6:9] = v4_remap_grav(get_gravity([0.70710678, 0.70710678, 0.0, 0.0]))

for cmd_name, cmd_vec in cmds_test:
    test_obs = base_obs.copy()
    test_obs[9:12] = cmd_vec
    with torch.no_grad():
        act = policy(torch.from_numpy(test_obs).unsqueeze(0)).numpy().squeeze()
    diff = act - zero_action
    print(f"\n{cmd_name} (cmd={cmd_vec}):")
    print(f"  action = {act}")
    print(f"  diff from STAND:")
    for i16 in range(16):
        if abs(diff[i16]) > 0.01:
            print(f"    I16[{i16:2d}] {isaac16[i16]:12s}: {diff[i16]:+.4f}")


# ============================================================
# Test 5: 检查MuJoCo的actuator配置
# ============================================================
print("\n" + "="*80)
print("Test 5: MuJoCo actuator配置")
print("="*80)

for i in range(m.nu):
    jid = m.actuator_trnid[i, 0]
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    kp = m.actuator_gainprm[i, 0]
    # For position actuators, biasprm[1] = -kp, biasprm[2] = -kd
    kp_bias = -m.actuator_biasprm[i, 1]
    kd_bias = -m.actuator_biasprm[i, 2]
    print(f"  Act[{i:2d}] {an:20s} -> Joint {jn:12s}  kp={kp_bias:.0f}  kd={kd_bias:.0f}  "
          f"ctrlrange=[{m.actuator_ctrlrange[i,0]:+.4f}, {m.actuator_ctrlrange[i,1]:+.4f}]")


# ============================================================
# Test 6: 检查初始ctrl值是否正确
# ============================================================
print("\n" + "="*80)
print("Test 6: 初始ctrl值（应该等于default_angles对应的actuator）")
print("="*80)

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.22
d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
d.qpos[7:] = default_angles
d.qvel[:] = 0

target = default_angles.copy()
d.ctrl[:] = target[act_to_jnt]

print(f"default_angles (MJ order): {default_angles}")
print(f"ctrl values: {d.ctrl}")
print(f"act_to_jnt: {act_to_jnt}")
for i in range(m.nu):
    jidx = act_to_jnt[i]
    an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  Act[{i:2d}] {an:20s}: ctrl={d.ctrl[i]:+.4f}  default[{jidx}]={default_angles[jidx]:+.4f}  "
          f"qpos[{7+jidx}]={d.qpos[7+jidx]:+.4f}")
