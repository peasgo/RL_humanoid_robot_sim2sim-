"""Headless MuJoCo obs dump for V6 humanoid — compare with IsaacLab dump.

Runs MuJoCo with the same policy, same initial state, cmd=[0,0,0],
and prints the first 2 policy steps' observations for comparison.

Usage:
    cd IsaacLab/mujoco_deploy_v6_humanoid
    python3 headless_obs_dump.py v6_robot.yaml
"""
import mujoco
import numpy as np
import torch
import yaml
import os
import sys


def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "v6_robot.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    clip_obs = float(config.get("clip_obs", 100.0))
    clip_actions = float(config.get("clip_actions", 100.0))
    init_height = float(config.get("init_height", 0.55))

    isaac_joint_order = config.get("isaac_joint_order")
    action_joint_order = config.get("action_joint_order")

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Get MuJoCo joint names
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    # Actuator -> joint mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # Joint order mappings
    isaac_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac_joint_order], dtype=np.int32
    )
    action_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in action_joint_order], dtype=np.int32
    )
    default_angles_isaac = default_angles[isaac_to_mujoco]
    default_angles_action = default_angles[action_to_mujoco]

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"Policy loaded: {policy_path}")

    # Initialize state — same as IsaacLab
    # IsaacLab Step0: joints at 0, NOT at default. No warmup.
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = 0.0  # joints at zero (matching IsaacLab Step0 with reset disabled)
    d.qvel[:] = 0.0

    action_raw = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    print(f"\nMuJoCo joint order: {mj_joint_names}")
    print(f"Isaac obs order:    {isaac_joint_order}")
    print(f"Action order:       {action_joint_order}")

    # Run policy steps
    for policy_step in range(2):
        # Read state BEFORE sim steps (for Step 0) or AFTER sim steps (for Step 1+)
        quat = d.qpos[3:7].copy()
        omega_body = d.qvel[3:6].copy()
        qj_mujoco = d.qpos[7:].copy()
        dqj_mujoco = d.qvel[6:].copy()

        # Remap to Isaac order
        qj_isaac = qj_mujoco[isaac_to_mujoco]
        dqj_isaac = dqj_mujoco[isaac_to_mujoco]

        # Compute observation terms
        omega_obs = omega_body * ang_vel_scale
        gravity_obs = get_gravity_orientation(quat)
        cmd_obs = cmd * cmd_scale
        qj_rel = (qj_isaac - default_angles_isaac) * dof_pos_scale
        dqj_obs = dqj_isaac * dof_vel_scale
        last_act = action_raw.copy()

        # Assemble obs
        obs = np.concatenate([omega_obs, gravity_obs, cmd_obs, qj_rel, dqj_obs, last_act])
        obs = np.clip(obs, -clip_obs, clip_obs)

        # Print
        print(f"\n{'='*70}")
        print(f"[MuJoCo Policy Step {policy_step}]")
        print(f"  qpos[0:3] (pos):     {d.qpos[0:3]}")
        print(f"  qpos[3:7] (quat):    {quat}  (w,x,y,z)")
        print(f"  qvel[3:6] (ang_vel): {omega_body}  (body frame)")
        print(f"  height:              {d.qpos[2]:.6f}m")

        labels = (
            ["ang_vel_x*0.2", "ang_vel_y*0.2", "ang_vel_z*0.2",
             "grav_x", "grav_y", "grav_z",
             "cmd_vx", "cmd_vy", "cmd_wz"]
            + [f"pos_rel_{jn}" for jn in isaac_joint_order]
            + [f"vel_{jn}*0.05" for jn in isaac_joint_order]
            + [f"last_act_{jn}" for jn in isaac_joint_order]
        )

        print(f"\n  --- Observation (48 dims) ---")
        for i, label in enumerate(labels):
            print(f"    obs[{i:2d}] {label:25s} = {obs[i]:+.8f}")

        # Run policy
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action_tensor = policy(obs_tensor)
        action_raw = action_tensor[0].numpy()
        action_raw = np.clip(action_raw, -clip_actions, clip_actions)

        print(f"\n  --- Action ({num_actions} dims) ---")
        for i, jn in enumerate(isaac_joint_order):
            if i < len(action_raw):
                print(f"    [{i:2d}] {jn:14s}  act={action_raw[i]:+.8f}")

        # Apply action — EXACTLY as run_v6_humanoid.py lines 608-613:
        # action_raw[i_isaac] is in Isaac BFS order (same as obs order)
        # Maps to MuJoCo via isaac_to_mujoco
        for i_isaac in range(num_actions):
            mj_idx = isaac_to_mujoco[i_isaac]
            target_dof_pos[mj_idx] = (
                action_raw[i_isaac] * action_scale + default_angles[mj_idx]
            )

        # Step simulation (decimation steps)
        for _ in range(control_decimation):
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)

    print(f"\n{'='*70}")
    print("Done. Compare these values with IsaacLab dump_v6_isaac_state_step0.py output.")


if __name__ == "__main__":
    main()
