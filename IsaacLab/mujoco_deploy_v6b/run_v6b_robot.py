import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os
import argparse


def get_gravity_orientation(quaternion):
    """Convert quaternion [w,x,y,z] to gravity vector in body frame."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def quat_to_rotmat_wxyz(quat_wxyz):
    """Quaternion (w,x,y,z) -> rotation matrix."""
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
    """Rotate world-frame vector into body frame."""
    R_wb = quat_to_rotmat_wxyz(quat_wxyz)
    return R_wb.T @ v_world


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control: tau = kp*(target_q - q) + kd*(target_dq - dq)"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V6b Humanoid Sim2Sim Deploy")
    parser.add_argument("config_file", type=str, help="config file name (e.g., v6b_robot.yaml)")
    args = parser.parse_args()

    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_paths = [
        os.path.join(current_dir, args.config_file),
        args.config_file
    ]

    config_path = None
    for p in potential_paths:
        if os.path.exists(p):
            config_path = p
            break

    if config_path is None:
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    print(f"Loading config: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        if not os.path.isabs(xml_path):
            xml_path = os.path.join(current_dir, xml_path)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        use_tanh_action = bool(config.get("use_tanh_action", False))
        action_clip = config.get("action_clip", None)
        if action_clip is not None:
            action_clip = float(action_clip)
        clip_actions = float(config.get("clip_actions", 100.0))

        dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Load MuJoCo model
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Build joint name list (exclude freejoint)
    joint_names_in_qpos_order = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            joint_names_in_qpos_order.append(jname)

    print(f"MuJoCo joint order ({len(joint_names_in_qpos_order)} joints): {joint_names_in_qpos_order}")

    # Build actuator -> joint mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = joint_names_in_qpos_order.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # Effort limits
    effort_limit_pd = np.full((num_actions,), np.inf, dtype=np.float32)
    for act_i in range(m.nu):
        j_pd = int(actuator_to_joint_indices[act_i])
        fr = m.actuator_forcerange[act_i]
        effort_limit_pd[j_pd] = float(max(abs(fr[0]), abs(fr[1])))

    # Isaac joint order from YAML (PhysX BFS order)
    isaac_joint_order = config.get("isaac_joint_order", None)
    if isaac_joint_order is None:
        isaac_joint_order = [
            'pelvis_link',
            'LHIPp', 'RHIPp',
            'LHIPy', 'RHIPy',
            'LHIPr', 'RHIPr',
            'LKNEEp', 'RKNEEp',
            'LANKLEp', 'RANKLEp',
            'LANKLEy', 'RANKLEy',
        ]

    print(f"Isaac joint order ({len(isaac_joint_order)} joints): {isaac_joint_order}")

    # Validate all joints exist
    for jname in isaac_joint_order:
        if jname not in joint_names_in_qpos_order:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model. Available: {joint_names_in_qpos_order}")

    # mujoco_to_isaac[mj_idx] = isaac_idx
    mujoco_to_isaac = []
    for mj_jname in joint_names_in_qpos_order:
        if mj_jname in isaac_joint_order:
            mujoco_to_isaac.append(isaac_joint_order.index(mj_jname))
        else:
            raise ValueError(f"Joint {mj_jname} not in Isaac order")
    mujoco_to_isaac = np.array(mujoco_to_isaac, dtype=np.int32)

    # isaac_to_mujoco[isaac_idx] = mj_idx
    isaac_to_mujoco = np.full((len(isaac_joint_order),), -1, dtype=np.int32)
    for mj_idx, isaac_idx in enumerate(mujoco_to_isaac):
        if isaac_idx >= 0:
            isaac_to_mujoco[isaac_idx] = mj_idx

    if np.any(isaac_to_mujoco < 0):
        raise ValueError("Missing joints in MuJoCo model")

    # Print mapping
    print(f"\nMapping - Isaac <-> MuJoCo:")
    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        print(f"  Isaac[{i_isaac:2d}] {jname:14s} <-> MJ[{mj_idx:2d}] {joint_names_in_qpos_order[mj_idx]}"
              f"  def={default_angles[mj_idx]:+.4f}")

    # dq sign correction
    dq_sign = np.ones((len(isaac_joint_order),), dtype=np.float32)
    if isinstance(dq_sign_fixes_cfg, dict):
        for jn, s in dq_sign_fixes_cfg.items():
            if jn in isaac_joint_order:
                dq_sign[isaac_joint_order.index(jn)] = float(s)

    # Joint limits (from v6b_robot.xml joint range attributes)
    joint_limits = {
        'pelvis_link': (-1.6, 1.6),
        'RHIPp': (-3.1, 0.79), 'RHIPy': (-1.6, 0.7), 'RHIPr': (-1.7, 0.52),
        'RKNEEp': (-0.79, 2.6), 'RANKLEp': (-1.4, 1.4), 'RANKLEy': (-0.52, 0.52),
        'LHIPp': (-3.1, 0.79), 'LHIPy': (-0.7, 1.6), 'LHIPr': (-0.52, 1.7),
        'LKNEEp': (-2.6, 0.79), 'LANKLEp': (-1.4, 1.4), 'LANKLEy': (-0.523, 0.523),
    }

    # Precompute default angles in Isaac order
    default_angles_isaac = default_angles[isaac_to_mujoco].copy()

    # Initialize state
    action_isaac = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Set initial pose
    init_height = float(config.get("init_height", 0.55))
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles

    # Load policy
    policy = None
    if os.path.exists(policy_path) and os.path.isfile(policy_path):
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        print("Policy loaded successfully!")
    else:
        print(f"\nWARNING: No policy found at {policy_path}. Running in DEBUG MODE (holding default stance).")

    print(f"\nStarting simulation...")
    print(f"  num_actions: {num_actions}, num_obs: {num_obs}")
    print(f"  action_scale: {action_scale}, ang_vel_scale: {ang_vel_scale}")
    print(f"  dof_pos_scale: {dof_pos_scale}, dof_vel_scale: {dof_vel_scale}")
    print(f"  init_height: {init_height}")
    print(f"  cmd_init: {cmd}")
    print(f"  clip_actions: {clip_actions}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # ============================================================
        # Warmup phase: settle robot under gravity with PD holding default pose
        # ============================================================
        warmup_seconds = float(config.get("warmup_seconds", 5.0))
        warmup_steps = int(warmup_seconds / simulation_dt)
        print(f"\nWarmup: settling for {warmup_seconds:.1f}s ({warmup_steps} steps)...")

        for ws in range(warmup_steps):
            current_q = d.qpos[7:]
            current_dq = d.qvel[6:]
            tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds), current_dq, kds)
            tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)
            d.ctrl[:] = tau[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if ws % 100 == 0:
                viewer.sync()

        viewer.sync()

        # Print warmup end state
        warmup_quat = d.qpos[3:7].copy()
        warmup_gravity = get_gravity_orientation(warmup_quat)
        print(f"\n--- Warmup End State ---")
        print(f"  height: {d.qpos[2]:.4f}m  (init: {init_height})")
        print(f"  quat:   {warmup_quat}  (w,x,y,z)")
        print(f"  gravity_b: {warmup_gravity}  (expect ~[0,0,-1])")
        print(f"  ncon: {d.ncon}")
        print(f"--- End Warmup ---\n")

        # Reset velocities after warmup (clean start for policy)
        d.qvel[:] = 0
        counter = 0

        start = time.time()
        substep = 0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            current_q = d.qpos[7:]
            current_dq = d.qvel[6:]

            # Compute torque via PD control
            tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds), current_dq, kds)
            tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)

            # Apply torque
            d.ctrl[:] = tau[actuator_to_joint_indices]

            # Step simulation
            mujoco.mj_step(m, d)
            counter += 1
            substep += 1

            # Render (not every step to save performance)
            if counter % 4 == 0:
                viewer.sync()

            # Policy inference at decimation boundary
            if substep >= control_decimation and policy is not None:
                substep = 0

                quat = d.qpos[3:7].copy()
                # MuJoCo freejoint qvel[3:6] is angular velocity in body frame
                omega_body = d.qvel[3:6].copy()

                qj_mujoco = d.qpos[7:].copy()
                dqj_mujoco = d.qvel[6:].copy()
                qj_isaac = qj_mujoco[isaac_to_mujoco]
                dqj_isaac = dqj_mujoco[isaac_to_mujoco] * dq_sign

                # Scale observations
                omega = omega_body * ang_vel_scale
                qj = (qj_isaac - default_angles_isaac) * dof_pos_scale
                dqj = dqj_isaac * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)

                # Build observation:
                # [ang_vel(3), gravity(3), cmd(3), joint_pos_rel(13), joint_vel(13), last_action(13)] = 48
                obs[0:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9+num_actions] = qj
                obs[9+num_actions:9+2*num_actions] = dqj
                obs[9+2*num_actions:9+3*num_actions] = action_isaac.astype(np.float32)

                # Clip observations (rsl_rl does this)
                obs = np.clip(obs, -100.0, 100.0)

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_isaac = policy(obs_tensor).detach().numpy().squeeze()

                # Clip actions (rsl_rl does this)
                action_isaac = np.clip(action_isaac, -clip_actions, clip_actions)

                # Action post-processing
                if use_tanh_action:
                    action_isaac = np.tanh(action_isaac)
                if action_clip is not None:
                    action_isaac = np.clip(action_isaac, -action_clip, action_clip)

                # Map action from Isaac order to MuJoCo order
                for i_isaac in range(num_actions):
                    mj_idx = isaac_to_mujoco[i_isaac]
                    target_dof_pos[mj_idx] = action_isaac[i_isaac] * action_scale + default_angles[mj_idx]

                # Apply joint limits
                for i, jname in enumerate(joint_names_in_qpos_order):
                    if jname in joint_limits:
                        low, high = joint_limits[jname]
                        target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)

            elif substep >= control_decimation:
                substep = 0

            # Timing
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Simulation completed.")
