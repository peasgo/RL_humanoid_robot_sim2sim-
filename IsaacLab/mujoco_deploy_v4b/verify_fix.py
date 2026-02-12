"""验证修复后的v4b run_v4_robot.py关节映射是否正确
直接复用diagnose_joint_order_v2.py的逻辑，但用修复后的Isaac17顺序
"""
import numpy as np
import torch
import mujoco
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "v4_scene.xml")
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = 0.005

mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

# 修复后的Isaac17/16顺序（与v4一致）
isaac17_order = [
    'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy', 'Waist_2',
    'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
    'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
]
isaac16_action_order = [
    'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
    'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
    'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
]

isaac17_to_mj17 = np.array([mj_joint_names.index(j) for j in isaac17_order])
isaac16_to_mj17 = np.array([mj_joint_names.index(j) for j in isaac16_action_order])

default_angles_mj = np.array([
    3.14159, 0.7854, 0.0, -1.5708, 0.7854,
    0.7854, 0.0, 1.5708, 0.7854,
    0.7854, 0.0, 1.5708, -0.7854,
    -0.7854, 0.0, -1.5708, -0.7854,
], dtype=np.float32)

policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/exported/policy.pt"
policy = torch.jit.load(policy_path)

def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

def quat_to_rotmat_wxyz(quat_wxyz):
    w, x, y, z = quat_wxyz
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0: w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def world_to_body(v_world, quat_wxyz):
    R = quat_to_rotmat_wxyz(quat_wxyz)
    return R.T @ v_world

def run_test(cmd, num_steps=200):
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles_mj
    
    actuator_to_joint = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_to_joint.append(mj_joint_names.index(jn))
    actuator_to_joint = np.array(actuator_to_joint)
    
    target_dof_pos = default_angles_mj.copy()
    action_16 = np.zeros(16, dtype=np.float32)
    waist_mj_idx = mj_joint_names.index('Waist_2')
    
    # Warmup
    for _ in range(1000):
        d.ctrl[:] = target_dof_pos[actuator_to_joint]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    
    for step in range(num_steps):
        d.ctrl[:] = target_dof_pos[actuator_to_joint]
        for _ in range(4):
            mujoco.mj_step(m, d)
        
        quat = d.qpos[3:7]
        lin_vel_b = world_to_body(d.qvel[0:3].copy(), quat)
        ang_vel_b = world_to_body(d.qvel[3:6].copy(), quat)
        gravity = get_gravity_orientation(quat)
        
        qj_mj = d.qpos[7:].copy()
        dqj_mj = d.qvel[6:].copy()
        
        # Isaac17 reorder
        qj_i17 = qj_mj[isaac17_to_mj17]
        dqj_i17 = dqj_mj[isaac17_to_mj17]
        default_i17 = default_angles_mj[isaac17_to_mj17]
        
        obs = np.zeros(62, dtype=np.float32)
        obs[0:3] = [lin_vel_b[2], lin_vel_b[0], lin_vel_b[1]]
        obs[3:6] = [ang_vel_b[0], ang_vel_b[2], ang_vel_b[1]]
        obs[6:9] = [gravity[2], gravity[0], gravity[1]]
        obs[9:12] = cmd
        obs[12:29] = (qj_i17 - default_i17)
        obs[29:46] = dqj_i17
        obs[46:62] = action_16
        
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action_16 = policy(obs_t).detach().numpy().squeeze()
        action_16 = np.clip(action_16, -5.0, 5.0)
        
        target_dof_pos[waist_mj_idx] = default_angles_mj[waist_mj_idx]
        for i16 in range(16):
            mj_idx = isaac16_to_mj17[i16]
            target_dof_pos[mj_idx] = action_16[i16] * 0.25 + default_angles_mj[mj_idx]
        
        if step % 50 == 0 or step == num_steps - 1:
            pos = d.qpos[0:3]
            print(f"  step={step:3d} pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) act_max={np.max(np.abs(action_16)):.3f}")
    
    return d.qpos[0:3].copy()

print("=" * 60)
print("验证修复后的Isaac17关节映射")
print("=" * 60)

print("\n--- 前进 cmd=[0.3, 0.0, 0.0] ---")
pos = run_test(np.array([0.3, 0.0, 0.0], dtype=np.float32), 200)
print(f"  最终: x={pos[0]:+.4f}, y={pos[1]:+.4f}, z={pos[2]:.4f}")
print(f"  前进距离(-Y): {-pos[1]:.4f}m")

print("\n--- 左移 cmd=[0.0, 0.3, 0.0] ---")
pos = run_test(np.array([0.0, 0.3, 0.0], dtype=np.float32), 200)
print(f"  最终: x={pos[0]:+.4f}, y={pos[1]:+.4f}, z={pos[2]:.4f}")

print("\n--- 左转 cmd=[0.0, 0.0, 0.5] ---")
pos = run_test(np.array([0.0, 0.0, 0.5], dtype=np.float32), 200)
print(f"  最终: x={pos[0]:+.4f}, y={pos[1]:+.4f}, z={pos[2]:.4f}")

print("\n--- 站立 cmd=[0.0, 0.0, 0.0] ---")
pos = run_test(np.array([0.0, 0.0, 0.0], dtype=np.float32), 200)
print(f"  最终: x={pos[0]:+.4f}, y={pos[1]:+.4f}, z={pos[2]:.4f}")
print(f"  高度保持: {pos[2]:.4f}m (目标0.22m)")

print("\n✓ 如果前进距离>0.1m且高度>0.15m，说明映射正确")
