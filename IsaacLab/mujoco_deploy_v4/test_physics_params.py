#!/usr/bin/env python3
"""
测试不同物理参数组合对V4四足机器人sim2sim的影响。
无头模式运行，记录零命令下的漂移距离。

用法:
    cd IsaacLab/mujoco_deploy_v4
    python test_physics_params.py v4_robot.yaml
"""

import time
import mujoco
import numpy as np
import torch
import yaml
import os
import argparse
import copy

try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ImportError:
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def quat_to_rotmat_wxyz(quat_wxyz):
    w, x, y, z = quat_wxyz
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n > 0:
        w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R


def world_to_body(v_world, quat_wxyz):
    R_wb = quat_to_rotmat_wxyz(quat_wxyz)
    return R_wb.T @ v_world


def v4_remap_lin_vel(lin_vel_body):
    return np.array([lin_vel_body[2], lin_vel_body[0], lin_vel_body[1]])


def v4_remap_ang_vel(ang_vel_body):
    return np.array([ang_vel_body[0], ang_vel_body[2], ang_vel_body[1]])


def v4_remap_gravity(gravity_body):
    return np.array([gravity_body[2], gravity_body[0], gravity_body[1]])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_test(m, policy, config, kps, kds, default_angles, 
             mj_joint_names, actuator_to_joint_indices, effort_limit_pd,
             isaac17_to_mujoco17, isaac16_action_to_mj17, 
             isaac17_joint_order, isaac16_action_order,
             waist_mj_idx, waist_default, dq_sign,
             num_actions, num_obs, action_scale, cmd_scale,
             lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
             action_clip, action_filter_alpha,
             test_duration=10.0, label=""):
    """运行一次测试，返回漂移距离和最终高度"""
    
    d = mujoco.MjData(m)
    num_mj_joints = len(mj_joint_names)
    
    # Damping compensation
    mj_joint_damping = np.zeros(num_mj_joints, dtype=np.float32)
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof_adr = m.jnt_dofadr[jid]
        mj_joint_damping[i] = float(m.dof_damping[dof_adr])
    kds_compensated = np.maximum(kds - mj_joint_damping, 0.0)
    
    # Initial pose
    d.qpos[2] = 0.22
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    
    target_dof_pos = default_angles.copy()
    
    # Warmup
    for ws in range(1000):
        current_q = d.qpos[7:]
        current_dq = d.qvel[6:]
        tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds_compensated), current_dq, kds_compensated)
        tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)
        d.ctrl[:] = tau[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
    
    # Record initial position
    init_pos = d.qpos[0:3].copy()
    d.qvel[:] = 0  # Clean start
    
    # Run with policy
    action_isaac16 = np.zeros(num_actions, dtype=np.float32)
    action_isaac16_prev = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    counter = 0
    
    num_steps = int(test_duration / m.opt.timestep)
    control_decimation = 4
    
    joint_limits = {}
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if m.jnt_limited[jid]:
            joint_limits[jname] = (float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1]))
    
    fell = False
    for step in range(num_steps):
        current_q = d.qpos[7:]
        current_dq = d.qvel[6:]
        tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds_compensated), current_dq, kds_compensated)
        tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)
        d.ctrl[:] = tau[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        counter += 1
        
        # Check if fallen
        if d.qpos[2] < 0.05:
            fell = True
            break
        
        if counter % control_decimation == 0 and policy is not None:
            quat = d.qpos[3:7]
            base_lin_vel_world = d.qvel[0:3].copy()
            base_ang_vel_world = d.qvel[3:6].copy()
            base_lin_vel = world_to_body(base_lin_vel_world, quat)
            omega = world_to_body(base_ang_vel_world, quat)
            
            qj_mujoco = d.qpos[7:].copy()
            dqj_mujoco = d.qvel[6:].copy()
            qj_isaac17 = qj_mujoco[isaac17_to_mujoco17]
            dqj_isaac17 = dqj_mujoco[isaac17_to_mujoco17] * dq_sign
            default_angles_isaac17 = default_angles[isaac17_to_mujoco17]
            
            gravity_orientation = get_gravity_orientation(quat)
            
            base_lin_vel_obs = v4_remap_lin_vel(base_lin_vel) * lin_vel_scale
            omega_obs = v4_remap_ang_vel(omega) * ang_vel_scale
            gravity_obs = v4_remap_gravity(gravity_orientation)
            
            qj = (qj_isaac17 - default_angles_isaac17) * dof_pos_scale
            dqj = dqj_isaac17 * dof_vel_scale
            
            obs[0:3] = base_lin_vel_obs
            obs[3:6] = omega_obs
            obs[6:9] = gravity_obs
            obs[9:12] = cmd * cmd_scale
            obs[12:29] = qj
            obs[29:46] = dqj
            obs[46:62] = action_isaac16.astype(np.float32)
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_isaac16 = policy(obs_tensor).detach().numpy().squeeze()
            
            if action_clip is not None:
                action_isaac16 = np.clip(action_isaac16, -action_clip, action_clip)
            
            if action_filter_alpha > 0:
                action_isaac16 = action_filter_alpha * action_isaac16_prev + (1.0 - action_filter_alpha) * action_isaac16
            action_isaac16_prev[:] = action_isaac16
            
            target_dof_pos[waist_mj_idx] = waist_default
            for i16 in range(num_actions):
                mj_idx = isaac16_action_to_mj17[i16]
                target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]
            
            for i, jname in enumerate(mj_joint_names):
                if jname in joint_limits:
                    low, high = joint_limits[jname]
                    target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)
    
    final_pos = d.qpos[0:3].copy()
    drift = np.sqrt((final_pos[0] - init_pos[0])**2 + (final_pos[1] - init_pos[1])**2)
    height = final_pos[2]
    
    status = "FELL" if fell else "OK"
    print(f"  [{label:30s}] drift={drift:.3f}m  h={height:.3f}m  status={status}")
    return drift, height, fell


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, args.config_file)
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    policy_path = config["policy_path"]
    xml_path = os.path.join(current_dir, config["xml_path"])
    
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    # Load policy
    policy = torch.jit.load(policy_path)
    print("Policy loaded.")
    
    # Load base model
    m_base = mujoco.MjModel.from_xml_path(xml_path)
    m_base.opt.timestep = 0.005
    
    # Get joint info from base model
    mj_joint_names = []
    for jid in range(m_base.njnt):
        jname = mujoco.mj_id2name(m_base, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m_base.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)
    
    num_mj_joints = len(mj_joint_names)
    
    actuator_to_joint_indices = []
    for i in range(m_base.nu):
        joint_id = m_base.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m_base, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)
    
    effort_limit_pd = np.full((num_mj_joints,), np.inf, dtype=np.float32)
    for act_i in range(m_base.nu):
        j_pd = int(actuator_to_joint_indices[act_i])
        fr = m_base.actuator_forcerange[act_i]
        effort_limit_pd[j_pd] = float(max(abs(fr[0]), abs(fr[1])))
    
    # Isaac joint orders
    isaac17_joint_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy', 'Waist_2',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]
    isaac16_action_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]
    
    isaac17_to_mujoco17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac17_joint_order], dtype=np.int32)
    isaac16_action_to_mj17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac16_action_order], dtype=np.int32)
    
    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]
    dq_sign = np.ones((len(isaac17_joint_order),), dtype=np.float32)
    
    print(f"\n{'='*70}")
    print(f"V4 Quadruped Physics Parameter Sweep (10s zero-command test)")
    print(f"{'='*70}")
    
    # ============================================================
    # Test different parameter combinations
    # We modify the model's damping and armature directly
    # ============================================================
    
    test_configs = [
        # (label, damping, armature, robot_friction, floor_friction, action_clip, filter_alpha)
        ("CURRENT: d=3 a=0.3 f=2/3",       3.0, 0.3, 2.0, 3.0, 3.0, 0.2),
        ("REF-like: d=1 a=0.1 f=1/1",       1.0, 0.1, 1.0, 1.0, None, 0.0),
        ("d=1 a=0.1 f=0.6/1.0",             1.0, 0.1, 0.6, 1.0, None, 0.0),
        ("d=0.5 a=0.05 f=0.6/1.0",          0.5, 0.05, 0.6, 1.0, None, 0.0),
        ("d=1 a=0.1 f=0.6/1.0 clip3",       1.0, 0.1, 0.6, 1.0, 3.0, 0.0),
        ("d=1 a=0.1 f=0.6/1.0 clip3 filt",  1.0, 0.1, 0.6, 1.0, 3.0, 0.2),
        ("d=0 a=0.01 f=0.6/1.0",            0.0, 0.01, 0.6, 1.0, None, 0.0),
        ("d=0 a=0.01 f=0.6/1.0 clip3",      0.0, 0.01, 0.6, 1.0, 3.0, 0.0),
        ("d=2 a=0.2 f=1/1",                 2.0, 0.2, 1.0, 1.0, None, 0.0),
        ("d=1 a=0.1 f=0.8/1.0",             1.0, 0.1, 0.8, 1.0, None, 0.0),
        ("d=1 a=0.1 f=0.8/1.0 clip3 filt",  1.0, 0.1, 0.8, 1.0, 3.0, 0.2),
    ]
    
    results = []
    for label, damping, armature, robot_fric, floor_fric, ac, fa in test_configs:
        # Create fresh model copy
        m = mujoco.MjModel.from_xml_path(xml_path)
        m.opt.timestep = 0.005
        
        # Modify damping and armature for all DOFs
        for jid in range(m.njnt):
            if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                dof_adr = m.jnt_dofadr[jid]
                m.dof_damping[dof_adr] = damping
                m.dof_armature[dof_adr] = armature
        
        # Modify friction for all geoms
        for gid in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname == "floor":
                m.geom_friction[gid, 0] = floor_fric
            elif m.geom_contype[gid] == 1:  # robot collision geoms
                m.geom_friction[gid, 0] = robot_fric
        
        try:
            drift, height, fell = run_test(
                m, policy, config, kps, kds, default_angles,
                mj_joint_names, actuator_to_joint_indices, effort_limit_pd,
                isaac17_to_mujoco17, isaac16_action_to_mj17,
                isaac17_joint_order, isaac16_action_order,
                waist_mj_idx, waist_default, dq_sign,
                num_actions, num_obs, action_scale, cmd_scale,
                lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
                ac, fa if fa else 0.0,
                test_duration=10.0, label=label
            )
            results.append((label, drift, height, fell))
        except Exception as e:
            print(f"  [{label:30s}] ERROR: {e}")
            results.append((label, -1, -1, True))
    
    print(f"\n{'='*70}")
    print(f"{'Label':35s} {'Drift(m)':>10s} {'Height(m)':>10s} {'Status':>8s}")
    print(f"{'-'*70}")
    for label, drift, height, fell in results:
        status = "FELL" if fell else "OK"
        print(f"{label:35s} {drift:10.3f} {height:10.3f} {status:>8s}")
    print(f"{'='*70}")
    print("\nTarget: drift < 0.1m, height ≈ 0.22m, status = OK")
