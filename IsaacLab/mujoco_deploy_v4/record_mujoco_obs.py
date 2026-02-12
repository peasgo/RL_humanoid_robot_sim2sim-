"""
Record MuJoCo observations for comparison with IsaacLab reference.
Runs headless, saves first 100 policy steps, then prints comparison.
"""
import numpy as np
import mujoco
import torch
import yaml
import os
from scipy.spatial.transform import Rotation as R

def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector v by inverse of quaternion q (w,x,y,z format)."""
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # scipy uses x,y,z,w
    return r.inv().apply(v)

def v4_remap_lin_vel(v):
    return np.array([v[2], v[0], v[1]])

def v4_remap_ang_vel(v):
    return np.array([v[0], v[2], v[1]])

def v4_remap_gravity(v):
    return np.array([v[2], v[0], v[1]])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def main():
    deploy_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load config
    with open(os.path.join(deploy_dir, "v4_robot.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(os.path.join(deploy_dir, "v4_scene.xml"))
    d = mujoco.MjData(m)
    
    # Load policy
    policy_path = os.path.join(deploy_dir, cfg["policy_path"])
    policy = torch.jit.load(policy_path, map_location="cpu")
    
    num_actions = cfg["num_actions"]  # 16
    num_obs = cfg["num_obs"]  # 62
    action_scale = cfg["action_scale"]  # 0.25
    
    # Hardcoded joint orders (same as run_v4_robot.py)
    ISAAC17_JOINT_ORDER = [
        'LHIPp','RHIPp','LHIPy','RHIPy','Waist_2',
        'LSDp','RSDp','LKNEEp','RKNEEp','LSDy','RSDy',
        'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAp'
    ]
    ISAAC16_ACTION_ORDER = [
        'LHIPp','RHIPp','LHIPy','RHIPy',
        'LSDp','RSDp','LKNEEp','RKNEEp','LSDy','RSDy',
        'LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAp'
    ]
    MUJOCO17_JOINT_ORDER = [
        'LHIPy','LHIPp','LKNEEp','LANKLEp',
        'RHIPy','RHIPp','RKNEEp','RANKLEp',
        'Waist_2',
        'LSDy','LSDp','LARMp','LARMAp',
        'RSDy','RSDp','RARMp','RARMAp'
    ]
    
    # Build mapping arrays
    mj_joint_ids = []
    for name in MUJOCO17_JOINT_ORDER:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        mj_joint_ids.append(jid)
    
    # isaac17 -> mujoco17 mapping
    isaac17_to_mj17 = []
    for iname in ISAAC17_JOINT_ORDER:
        mj_idx = MUJOCO17_JOINT_ORDER.index(iname)
        isaac17_to_mj17.append(mj_idx)
    isaac17_to_mj17 = np.array(isaac17_to_mj17)
    
    # isaac16 action -> mujoco17 mapping
    isaac16_to_mj17 = []
    for aname in ISAAC16_ACTION_ORDER:
        mj_idx = MUJOCO17_JOINT_ORDER.index(aname)
        isaac16_to_mj17.append(mj_idx)
    isaac16_to_mj17 = np.array(isaac16_to_mj17)
    
    # Waist_2 lock
    waist2_mj_idx = MUJOCO17_JOINT_ORDER.index("Waist_2")
    waist2_lock_value = np.pi  # 3.14159...
    
    # Default angles (mujoco order)
    default_angles_mj = np.array(cfg["default_angles"], dtype=np.float64)
    
    # PD gains (mujoco order)
    kps_mj = np.array(cfg["kps"], dtype=np.float64)
    kds_mj = np.array(cfg["kds"], dtype=np.float64)
    
    # Effort limits (from run_v4_robot.py)
    effort_limits_mj = np.array([
        150, 150, 150, 120,   # L leg
        150, 150, 150, 120,   # R leg
        1000,                  # Waist_2
        150, 150, 150, 150,   # L arm
        150, 150, 150, 150,   # R arm
    ], dtype=np.float64)
    
    # Damping compensation
    mj_joint_damping = np.zeros(17)
    for i, jid in enumerate(mj_joint_ids):
        dof_adr = m.jnt_dofadr[jid]
        mj_joint_damping[i] = m.dof_damping[dof_adr]
    kds_compensated = np.maximum(kds_mj - mj_joint_damping, 0.0)
    
    print(f"MuJoCo joint damping: {mj_joint_damping[:5]}...")
    print(f"kds_mj: {kds_mj[:5]}...")
    print(f"kds_compensated: {kds_compensated[:5]}...")
    
    # Actuator mapping
    actuator_to_joint = []
    for i in range(m.nu):
        act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        found = False
        for j, jname in enumerate(MUJOCO17_JOINT_ORDER):
            if jname.lower() == act_name.lower():
                actuator_to_joint.append(j)
                found = True
                break
        if not found:
            print(f"WARNING: actuator '{act_name}' not matched to any joint!")
    actuator_to_joint = np.array(actuator_to_joint)
    
    # Initialize robot
    mujoco.mj_resetData(m, d)
    init_height = 0.22
    d.qpos[2] = init_height
    
    # V4 orientation: rotated +90° around X
    d.qpos[3] = 0.70710678  # w
    d.qpos[4] = 0.70710678  # x
    d.qpos[5] = 0.0  # y
    d.qpos[6] = 0.0  # z
    
    # Set default joint positions
    for i, jid in enumerate(mj_joint_ids):
        qadr = m.jnt_qposadr[jid]
        d.qpos[qadr] = default_angles_mj[i]
    
    mujoco.mj_forward(m, d)
    
    # Warmup: PD to default pose
    warmup_steps = 1000
    print(f"Warmup: {warmup_steps} steps...")
    for step in range(warmup_steps):
        current_q = np.array([d.qpos[m.jnt_qposadr[jid]] for jid in mj_joint_ids])
        current_dq = np.array([d.qvel[m.jnt_dofadr[jid]] for jid in mj_joint_ids])
        
        tau = pd_control(default_angles_mj, current_q, kps_mj,
                        np.zeros(17), current_dq, kds_compensated)
        tau = np.clip(tau, -effort_limits_mj, effort_limits_mj)
        d.ctrl[:] = tau[actuator_to_joint]
        mujoco.mj_step(m, d)
    
    print(f"After warmup: pos=({d.qpos[0]:.4f},{d.qpos[1]:.4f},{d.qpos[2]:.4f})")
    print(f"  quat(wxyz)=({d.qpos[3]:.4f},{d.qpos[4]:.4f},{d.qpos[5]:.4f},{d.qpos[6]:.4f})")
    
    # Policy loop
    decimation = 4
    last_action = np.zeros(num_actions)
    cmd = np.array([0.0, 0.0, 0.0])
    
    all_obs = []
    all_actions = []
    all_pos = []
    all_quat = []
    
    num_policy_steps = 100
    
    for step in range(num_policy_steps):
        # Get current state
        quat_wxyz = d.qpos[3:7].copy()
        
        # World frame velocities
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        lin_vel_world = d.cvel[body_id][3:6].copy()
        ang_vel_world = d.cvel[body_id][0:3].copy()
        
        # Convert to body frame
        lin_vel_body = quat_rotate_inverse(quat_wxyz, lin_vel_world)
        ang_vel_body = quat_rotate_inverse(quat_wxyz, ang_vel_world)
        
        # Gravity in body frame
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = quat_rotate_inverse(quat_wxyz, gravity_world)
        
        # V4 coordinate remap
        lin_vel_obs = v4_remap_lin_vel(lin_vel_body)
        ang_vel_obs = v4_remap_ang_vel(ang_vel_body)
        gravity_obs = v4_remap_gravity(gravity_body)
        
        # Joint positions and velocities in Isaac order
        current_q_mj = np.array([d.qpos[m.jnt_qposadr[jid]] for jid in mj_joint_ids])
        current_dq_mj = np.array([d.qvel[m.jnt_dofadr[jid]] for jid in mj_joint_ids])
        
        joint_pos_isaac = current_q_mj[isaac17_to_mj17]
        joint_vel_isaac = current_dq_mj[isaac17_to_mj17]
        default_isaac = default_angles_mj[isaac17_to_mj17]
        
        joint_pos_rel = joint_pos_isaac - default_isaac
        
        # Build observation
        obs = np.zeros(num_obs, dtype=np.float32)
        obs[0:3] = lin_vel_obs
        obs[3:6] = ang_vel_obs
        obs[6:9] = gravity_obs
        obs[9:12] = cmd
        obs[12:29] = joint_pos_rel
        obs[29:46] = joint_vel_isaac
        obs[46:62] = last_action
        
        all_obs.append(obs.copy())
        all_pos.append(d.qpos[:3].copy())
        all_quat.append(quat_wxyz.copy())
        
        # Policy inference
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action = policy(obs_tensor).detach().numpy().squeeze()
        all_actions.append(action.copy())
        last_action = action.copy()
        
        # Apply action
        target_q_mj = default_angles_mj.copy()
        for i, mj_idx in enumerate(isaac16_to_mj17):
            target_q_mj[mj_idx] += action[i] * action_scale
        target_q_mj[waist2_mj_idx] = waist2_lock_value
        
        # Simulate decimation steps
        for _ in range(decimation):
            current_q = np.array([d.qpos[m.jnt_qposadr[jid]] for jid in mj_joint_ids])
            current_dq = np.array([d.qvel[m.jnt_dofadr[jid]] for jid in mj_joint_ids])
            
            tau = pd_control(target_q_mj, current_q, kps_mj,
                           np.zeros(17), current_dq, kds_compensated)
            tau = np.clip(tau, -effort_limits_mj, effort_limits_mj)
            d.ctrl[:] = tau[actuator_to_joint]
            mujoco.mj_step(m, d)
        
        if step < 10 or step % 10 == 0:
            print(f"  Step {step:3d}: pos=({d.qpos[0]:+.4f},{d.qpos[1]:+.4f},{d.qpos[2]:.4f}) "
                  f"obs[:3]=[{obs[0]:+.4f},{obs[1]:+.4f},{obs[2]:+.4f}] "
                  f"obs[6:9]=[{obs[6]:+.4f},{obs[7]:+.4f},{obs[8]:+.4f}] "
                  f"act_max={np.max(np.abs(action)):.3f}")
    
    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)
    all_pos = np.array(all_pos)
    all_quat = np.array(all_quat)
    
    save_path = os.path.join(deploy_dir, "mujoco_obs.npz")
    np.savez(save_path, obs=all_obs, actions=all_actions, pos=all_pos, quat=all_quat)
    print(f"\nSaved {num_policy_steps} steps to {save_path}")
    
    # Load IsaacLab reference and compare
    ref_path = os.path.join(deploy_dir, "isaaclab_reference_obs.npz")
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        ref_obs = ref["obs"]
        ref_actions = ref["actions"]
        ref_pos = ref["pos"]
        
        n = min(len(all_obs), len(ref_obs))
        
        print(f"\n{'='*80}")
        print(f"COMPARISON: MuJoCo vs IsaacLab (first {n} steps)")
        print(f"{'='*80}")
        
        # Compare observation components
        labels = [
            ("lin_vel", 0, 3),
            ("ang_vel", 3, 6),
            ("gravity", 6, 9),
            ("cmd", 9, 12),
            ("joint_pos_rel", 12, 29),
            ("joint_vel", 29, 46),
            ("last_action", 46, 62),
        ]
        
        for label, start, end in labels:
            mj_vals = all_obs[:n, start:end]
            isaac_vals = ref_obs[:n, start:end]
            diff = mj_vals - isaac_vals
            
            # Mean absolute difference
            mad = np.mean(np.abs(diff))
            max_diff = np.max(np.abs(diff))
            
            # Per-step comparison for first 5 steps
            print(f"\n--- {label} (indices {start}:{end}) ---")
            print(f"  Mean abs diff: {mad:.6f}, Max abs diff: {max_diff:.6f}")
            
            dim = min(3, end - start)
            for s in range(min(5, n)):
                print(f"  Step {s}: MJ={mj_vals[s,:dim]}  "
                      f"Isaac={isaac_vals[s,:dim]}  "
                      f"Diff={diff[s,:dim]}")
        
        # Position comparison
        print(f"\n--- Position drift ---")
        for s in [0, 5, 10, 20, 50, min(n-1, 99)]:
            if s < n:
                mj_p = all_pos[s]
                isaac_p = ref_pos[s]
                print(f"  Step {s:3d}: MJ=({mj_p[0]:+.4f},{mj_p[1]:+.4f},{mj_p[2]:.4f})  "
                      f"Isaac=({isaac_p[0]:+.4f},{isaac_p[1]:+.4f},{isaac_p[2]:.4f})")
        
        # Action comparison
        print(f"\n--- Action comparison ---")
        for s in range(min(5, n)):
            mj_a = all_actions[s]
            isaac_a = ref_actions[s]
            diff_a = mj_a - isaac_a
            print(f"  Step {s}: MJ_max={np.max(np.abs(mj_a)):.3f} Isaac_max={np.max(np.abs(isaac_a)):.3f} "
                  f"diff_max={np.max(np.abs(diff_a)):.3f}")
    else:
        print(f"\nNo IsaacLab reference found at {ref_path}")

if __name__ == "__main__":
    main()
