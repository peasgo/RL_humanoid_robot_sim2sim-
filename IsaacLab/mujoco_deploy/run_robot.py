import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os
import argparse

try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ImportError:
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()

    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_paths = [
        os.path.join(current_dir, "configs", args.config_file),
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
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # Fix xml_path if needed
        if not os.path.exists(xml_path):
            for candidate in [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), xml_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), xml_path),
                os.path.basename(xml_path)
            ]:
                if os.path.exists(candidate):
                    xml_path = candidate
                    break

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        use_tanh_action = bool(config.get("use_tanh_action", False))
        action_clip = config.get("action_clip", None)
        if action_clip is not None:
            action_clip = float(action_clip)

        dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Initialize state
    action_isaac = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    fixed_base_height = 0.3

    # Load MuJoCo model
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Build joint name list (exclude freejoint)
    joint_names_in_qpos_order = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            joint_names_in_qpos_order.append(jname)

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

    # Set initial pose
    d.qpos[7:] = default_angles
    d.qpos[2] = fixed_base_height

    # Load policy
    policy = None
    if os.path.exists(policy_path) and os.path.isfile(policy_path):
        print(f"Loading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
    else:
        print("WARNING: No policy found. Running in DEBUG MODE (holding default stance).")

    # Joint order mapping (MuJoCo <-> Isaac/training)
    isaac_joint_order = [
        'LHipP', 'RHipP', 'LHipY', 'RHipY', 'LHipR', 'RHipR',
        'LKneeP', 'RKneeP', 'LAnkleP', 'RAankleP', 'LAnkleR', 'RAnkleR'
    ]

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
    # dq sign correction
    dq_sign = np.ones((len(isaac_joint_order),), dtype=np.float32)
    if isinstance(dq_sign_fixes_cfg, dict):
        for jn, s in dq_sign_fixes_cfg.items():
            if jn in isaac_joint_order:
                dq_sign[isaac_joint_order.index(jn)] = float(s)

    # Joint limits
    joint_limits = {
        'RHipP': (-1.0, 1.0), 'RHipY': (-0.5, 0.5), 'RHipR': (-0.5, 0.5),
        'RKneeP': (-2.0, 0.2), 'RAankleP': (-0.5, 0.5), 'RAnkleR': (-0.3, 0.3),
        'LHipP': (-1.0, 1.0), 'LHipY': (-0.5, 0.5), 'LHipR': (-0.5, 0.5),
        'LKneeP': (-2.0, 0.5), 'LAnkleP': (-0.5, 0.5), 'LAnkleR': (-0.3, 0.3)
    }

    print("Starting simulation...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            current_q = d.qpos[7:]
            current_dq = d.qvel[6:]

            # Compute torque
            tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds), current_dq, kds)
            tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)

            # Apply torque
            d.ctrl[:] = tau[actuator_to_joint_indices]

            # Step simulation
            mujoco.mj_step(m, d)
            viewer.sync()
            counter += 1

            # Policy inference
            if counter % control_decimation == 0 and policy is not None:
                quat = d.qpos[3:7]
                base_lin_vel_world = d.qvel[0:3].copy()
                base_ang_vel_world = d.qvel[3:6].copy()
                base_lin_vel = world_to_body(base_lin_vel_world, quat)
                omega = world_to_body(base_ang_vel_world, quat)

                qj_mujoco = d.qpos[7:].copy()
                dqj_mujoco = d.qvel[6:].copy()
                qj_isaac = qj_mujoco[isaac_to_mujoco]
                dqj_isaac = dqj_mujoco[isaac_to_mujoco] * dq_sign

                # Scale observations
                base_lin_vel = base_lin_vel * lin_vel_scale
                omega = omega * ang_vel_scale
                default_angles_isaac = default_angles[isaac_to_mujoco]
                qj = (qj_isaac - default_angles_isaac) * dof_pos_scale
                dqj = dqj_isaac * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)

                # Build observation
                obs[0:3] = base_lin_vel
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = qj
                obs[24:36] = dqj
                obs[36:48] = action_isaac.astype(np.float32)

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_isaac = policy(obs_tensor).detach().numpy().squeeze()

                # Action post-processing
                if use_tanh_action:
                    action_isaac = np.tanh(action_isaac)
                if action_clip is not None:
                    action_isaac = np.clip(action_isaac, -action_clip, action_clip)

                # Map to MuJoCo order
                action = action_isaac[mujoco_to_isaac]
                target_dof_pos = action * action_scale + default_angles

                # Apply joint limits
                for i, jname in enumerate(joint_names_in_qpos_order):
                    if jname in joint_limits:
                        low, high = joint_limits[jname]
                        target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)

            # Timing
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Simulation completed.")
