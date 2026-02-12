#!/usr/bin/env python3
"""Final verification: test the fixed run_v4_robot.py logic with different commands.
Headless MuJoCo simulation, no viewer needed."""

import numpy as np
import mujoco
import torch
import yaml
import os
import sys

def world_to_body(vec_world, quat_wxyz):
    w, x, y, z = quat_wxyz
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])
    return R.T @ vec_world

def get_gravity_orientation(quat_wxyz):
    w, x, y, z = quat_wxyz
    gravity_world = np.array([0.0, 0.0, -1.0])
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])
    return R.T @ gravity_world

def v4_remap_lin_vel(v): return np.array([v[2], v[0], v[1]])
def v4_remap_ang_vel(v): return np.array([v[0], v[2], v[1]])
def v4_remap_gravity(v): return np.array([v[2], v[0], v[1]])

def run_test(cmd, label, config, m_template, num_steps=500):
    """Run a single test with given cmd, return final Y position (forward direction)."""
    m = m_template
    d = mujoco.MjData(m)
    
    # Config
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    action_scale = config["action_scale"]
    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    action_filter_alpha = float(config.get("action_filter_alpha", 0.0))
    action_ramp_steps = int(config.get("action_ramp_steps", 0))
    action_clip = config.get("action_clip", None)
    if action_clip is not None:
        action_clip = float(action_clip)
    init_height = float(config.get("init_height", 0.22))
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    # Joint mappings (same as run_v4_robot.py)
    mj_joint_names = [
        "Waist_2", "RSDp", "RSDy", "RARMp", "RARMAP",
        "LSDp", "LSDy", "LARMp", "LARMAp",
        "RHIPp", "RHIPy", "RKNEEP", "RANKLEp",
        "LHIPp", "LHIPy", "LKNEEp", "LANKLEp"
    ]
    isaac17_names = [
        "LHIPp", "RHIPp", "LHIPy", "RHIPy", "Waist_2",
        "LSDp", "RSDp", "LKNEEp", "RKNEEP", "LSDy", "RSDy",
        "LANKLEp", "RANKLEp", "LARMp", "RARMp", "LARMAp", "RARMAP"
    ]
    isaac16_action_names = [n for n in isaac17_names if n != "Waist_2"]
    
    mj_name_to_idx = {n: i for i, n in enumerate(mj_joint_names)}
    isaac17_to_mujoco17 = np.array([mj_name_to_idx[n] for n in isaac17_names])
    isaac16_action_to_mj17 = np.array([mj_name_to_idx[n] for n in isaac16_action_names])
    
    waist_mj_idx = mj_name_to_idx["Waist_2"]
    waist_default = default_angles[waist_mj_idx]
    
    # Actuator to joint mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, aname)
        joint_qpos_adr = m.jnt_qposadr[jid]
        actuator_to_joint_indices.append(joint_qpos_adr - 7)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices)
    
    dq_sign = np.ones(17, dtype=np.float32)
    
    # Initialize
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles
    mujoco.mj_forward(m, d)
    
    # Warmup
    target_dof_pos = default_angles.copy()
    for _ in range(200):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    
    # State
    action_isaac16_raw = np.zeros(num_actions, dtype=np.float32)
    action_isaac16_exec = np.zeros(num_actions, dtype=np.float32)
    action_isaac16_prev = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    
    cmd_arr = np.array(cmd, dtype=np.float32)
    counter = 0
    policy_step_count = 0
    init_pos = d.qpos[0:3].copy()
    
    # Load policy
    current_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = config["policy_path"]
    policy = torch.jit.load(policy_path)
    
    for step in range(num_steps * control_decimation):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        counter += 1
        
        if counter % control_decimation == 0:
            quat = d.qpos[3:7]
            base_lin_vel = world_to_body(d.qvel[0:3].copy(), quat)
            omega = world_to_body(d.qvel[3:6].copy(), quat)
            
            qj_mujoco = d.qpos[7:].copy()
            dqj_mujoco = d.qvel[6:].copy()
            qj_isaac17 = qj_mujoco[isaac17_to_mujoco17]
            dqj_isaac17 = dqj_mujoco[isaac17_to_mujoco17] * dq_sign
            default_angles_isaac17 = default_angles[isaac17_to_mujoco17]
            gravity_orientation = get_gravity_orientation(quat)
            
            base_lin_vel_obs = v4_remap_lin_vel(base_lin_vel)
            omega_obs = v4_remap_ang_vel(omega)
            gravity_obs = v4_remap_gravity(gravity_orientation)
            
            base_lin_vel_obs = base_lin_vel_obs * lin_vel_scale
            omega_obs = omega_obs * ang_vel_scale
            qj = (qj_isaac17 - default_angles_isaac17) * dof_pos_scale
            dqj = dqj_isaac17 * dof_vel_scale
            
            obs[0:3] = base_lin_vel_obs
            obs[3:6] = omega_obs
            obs[6:9] = gravity_obs
            obs[9:12] = cmd_arr * cmd_scale
            obs[12:29] = qj
            obs[29:46] = dqj
            obs[46:62] = action_isaac16_raw.astype(np.float32)  # RAW policy output
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_isaac16_raw = policy(obs_tensor).detach().numpy().squeeze()
            
            if action_clip is not None:
                action_isaac16_raw = np.clip(action_isaac16_raw, -action_clip, action_clip)
            
            # Separate execution action
            action_isaac16_exec = action_isaac16_raw.copy()
            
            if action_ramp_steps > 0 and policy_step_count < action_ramp_steps:
                action_isaac16_exec *= float(policy_step_count) / float(action_ramp_steps)
            
            if action_filter_alpha > 0:
                action_isaac16_exec = action_filter_alpha * action_isaac16_prev + (1.0 - action_filter_alpha) * action_isaac16_exec
            action_isaac16_prev[:] = action_isaac16_exec
            
            policy_step_count += 1
            
            target_dof_pos[waist_mj_idx] = waist_default
            for i16 in range(num_actions):
                mj_idx = isaac16_action_to_mj17[i16]
                target_dof_pos[mj_idx] = action_isaac16_exec[i16] * action_scale + default_angles[mj_idx]
    
    final_pos = d.qpos[0:3].copy()
    delta = final_pos - init_pos
    
    # V4 forward = -Y in world
    fwd_dist = -delta[1]
    lat_dist = delta[0]
    
    # Compute yaw from quaternion
    quat = d.qpos[3:7]
    w, x, y, z = quat
    # For V4 with X+90° base rotation, extract yaw around Z
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    yaw_deg = np.degrees(yaw)
    
    print(f"  {label:12s}: fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m  "
          f"yaw={yaw_deg:+.1f}°  h={final_pos[2]:.3f}m  "
          f"(dx={delta[0]:+.3f} dy={delta[1]:+.3f})")
    
    return fwd_dist, lat_dist, yaw_deg

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "v4_robot.yaml")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.opt.timestep = config["simulation_dt"]
    
    print("=" * 70)
    print("FINAL VERIFICATION: Fixed run_v4_robot.py configuration")
    print(f"  action_filter_alpha: {config.get('action_filter_alpha', 0)}")
    print(f"  obs_filter_alpha: {config.get('obs_filter_alpha', 0)}")
    print(f"  action_ramp_steps: {config.get('action_ramp_steps', 0)}")
    print(f"  init_height: {config.get('init_height', 0.22)}")
    print(f"  last_action in obs: RAW policy output (not filtered)")
    print("=" * 70)
    
    tests = [
        ([0.3, 0, 0],   "FWD 0.3"),
        ([-0.3, 0, 0],  "BWD 0.3"),
        ([0.0, 0, 0],   "STAND"),
        ([0.0, 0.3, 0], "LEFT 0.3"),
        ([0.0, 0, 0.5], "TURN_L"),
        ([0.0, 0, -0.5],"TURN_R"),
        ([0.5, 0, 0],   "FWD 0.5"),
        ([-0.5, 0, 0],  "BWD 0.5"),
    ]
    
    results = {}
    for cmd, label in tests:
        fwd, lat, yaw = run_test(cmd, label, config, m, num_steps=500)
        results[label] = (fwd, lat, yaw)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    fwd03 = results["FWD 0.3"][0]
    bwd03 = results["BWD 0.3"][0]
    stand = results["STAND"][0]
    
    print(f"  FWD-BWD diff (0.3): {fwd03 - bwd03:+.3f}m (should be > 0.3)")
    print(f"  FWD 0.3:  {fwd03:+.3f}m (should be positive)")
    print(f"  BWD 0.3:  {bwd03:+.3f}m (should be negative or near zero)")
    print(f"  STAND:    {stand:+.3f}m (should be near zero)")
    
    if "FWD 0.5" in results and "BWD 0.5" in results:
        fwd05 = results["FWD 0.5"][0]
        bwd05 = results["BWD 0.5"][0]
        print(f"  FWD-BWD diff (0.5): {fwd05 - bwd05:+.3f}m")
    
    turn_l = results.get("TURN_L", (0,0,0))[2]
    turn_r = results.get("TURN_R", (0,0,0))[2]
    print(f"  TURN_L yaw: {turn_l:+.1f}° (should be positive)")
    print(f"  TURN_R yaw: {turn_r:+.1f}° (should be negative)")
    
    # Pass/fail
    ok = True
    if fwd03 - bwd03 < 0.3:
        print("\n  ❌ FAIL: FWD-BWD differentiation too small")
        ok = False
    if bwd03 > 0.3:
        print(f"\n  ❌ FAIL: BWD still goes forward significantly ({bwd03:+.3f}m)")
        ok = False
    if ok:
        print("\n  ✅ PASS: Robot responds to different commands!")

if __name__ == "__main__":
    main()
