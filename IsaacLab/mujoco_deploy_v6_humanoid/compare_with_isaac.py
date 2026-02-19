"""
MuJoCo observation dump for V6 humanoid — element-by-element comparison with Isaac.

Usage:
  cd IsaacLab/mujoco_deploy_v6_humanoid
  python3 compare_with_isaac.py [--warmup] [--warmup-secs 5.0] [--steps 5]

This script prints the full 48-dim observation vector with per-element labels,
matching the format of dump_v6_isaac_obs.py for easy side-by-side comparison.

Two modes:
  Default:   Start from exact initial pose (no warmup), zero velocities
  --warmup:  Run warmup to let robot settle, then reset velocities to zero
"""

import mujoco
import numpy as np
import torch
import yaml
import os
import argparse


def get_gravity_orientation(quaternion):
    """Compute projected gravity in body frame from quaternion [w,x,y,z]."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def main():
    parser = argparse.ArgumentParser(description="MuJoCo obs dump for Isaac comparison")
    parser.add_argument("--warmup", action="store_true", help="Run warmup before dumping")
    parser.add_argument("--warmup-secs", type=float, default=5.0, help="Warmup duration (seconds)")
    parser.add_argument("--steps", type=int, default=5, help="Number of policy steps to dump")
    parser.add_argument("--no-policy", action="store_true", help="Skip policy inference, just dump obs")
    args = parser.parse_args()

    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "v6_robot.yaml")

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
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    init_height = float(config.get("init_height", 0.55))
    clip_obs = float(config.get("clip_obs", 100.0))
    clip_actions = float(config.get("clip_actions", 100.0))

    isaac_joint_order = config.get("isaac_joint_order", [
        'pelvis_link',
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'LHIPr', 'RHIPr', 'LKNEEp', 'RKNEEp',
        'LANKLEp', 'RANKLEp', 'LANKLEy', 'RANKLEy',
    ])

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Get joint names
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    isaac_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac_joint_order], dtype=np.int32
    )
    default_angles_isaac = default_angles[isaac_to_mujoco]

    # Actuator mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # Set initial pose
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0

    target_dof_pos = default_angles.copy()

    print(f"MuJoCo joint order: {mj_joint_names}")
    print(f"Isaac joint order:  {isaac_joint_order}")
    print(f"isaac_to_mujoco:    {isaac_to_mujoco}")
    print(f"default_angles (MJ): {default_angles}")
    print(f"default_angles (Isaac): {default_angles_isaac}")

    # Optional warmup
    if args.warmup:
        warmup_steps = int(args.warmup_secs / simulation_dt)
        print(f"\nRunning warmup for {args.warmup_secs}s ({warmup_steps} steps)...")
        for _ in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
        print(f"  Post-warmup height: {d.qpos[2]:.4f}m")
        print(f"  Post-warmup ncon: {d.ncon}")
        # Reset velocities (like run_v6_humanoid.py does)
        d.qvel[:] = 0
        print(f"  Velocities reset to zero.")
    else:
        # Just run mj_forward to initialize derived quantities
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_forward(m, d)
        print(f"\nNo warmup. Starting from exact initial pose.")
        print(f"  height: {d.qpos[2]:.4f}m, ncon: {d.ncon}")

    # Load policy
    policy = None
    if not args.no_policy and os.path.exists(policy_path):
        policy = torch.jit.load(policy_path)
        print(f"Policy loaded: {policy_path}")
    else:
        print(f"No policy loaded (--no-policy or file not found)")

    # Run policy steps
    action_raw = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    obs_labels = []
    obs_labels.append(("obs[0]", "ang_vel_x * 0.2"))
    obs_labels.append(("obs[1]", "ang_vel_y * 0.2"))
    obs_labels.append(("obs[2]", "ang_vel_z * 0.2"))
    obs_labels.append(("obs[3]", "gravity_x"))
    obs_labels.append(("obs[4]", "gravity_y"))
    obs_labels.append(("obs[5]", "gravity_z"))
    obs_labels.append(("obs[6]", "cmd_vx"))
    obs_labels.append(("obs[7]", "cmd_vy"))
    obs_labels.append(("obs[8]", "cmd_wz"))
    for i, jname in enumerate(isaac_joint_order):
        obs_labels.append((f"obs[{9+i}]", f"pos_rel_{jname}"))
    for i, jname in enumerate(isaac_joint_order):
        obs_labels.append((f"obs[{22+i}]", f"vel_{jname}*0.05"))
    for i, jname in enumerate(isaac_joint_order):
        obs_labels.append((f"obs[{35+i}]", f"last_act_{jname}"))

    for step in range(args.steps):
        # Simulate decimation steps
        for _ in range(control_decimation):
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)

        # Read state
        quat = d.qpos[3:7].copy()
        omega_body = d.qvel[3:6].copy()
        qj_mujoco = d.qpos[7:].copy()
        dqj_mujoco = d.qvel[6:].copy()

        qj_isaac = qj_mujoco[isaac_to_mujoco]
        dqj_isaac = dqj_mujoco[isaac_to_mujoco]

        # Build observation
        obs[0:3] = omega_body * ang_vel_scale
        obs[3:6] = get_gravity_orientation(quat)
        obs[6:9] = cmd * cmd_scale
        obs[9:22] = (qj_isaac - default_angles_isaac) * dof_pos_scale
        obs[22:35] = dqj_isaac * dof_vel_scale
        obs[35:48] = action_raw.copy()

        obs = np.clip(obs, -clip_obs, clip_obs)

        # Print
        print(f"\n{'='*70}")
        print(f"[MuJoCo Step {step}]")
        print(f"  root_pos:    {d.qpos[0:3]}")
        print(f"  root_quat:   {quat}  (w,x,y,z)")
        print(f"  ang_vel_b:   {omega_body}")
        print(f"  gravity_b:   {get_gravity_orientation(quat)}")
        print(f"  height:      {d.qpos[2]:.4f}m")
        print(f"  ncon:        {d.ncon}")

        print(f"\n  --- Joint state (Isaac order) ---")
        for i, jname in enumerate(isaac_joint_order):
            mj_idx = isaac_to_mujoco[i]
            print(f"    [{i:2d}] {jname:14s}  pos={qj_isaac[i]:+.6f}  def={default_angles_isaac[i]:+.4f}"
                  f"  rel={qj_isaac[i]-default_angles_isaac[i]:+.6f}  vel={dqj_isaac[i]:+.6f}")

        print(f"\n  --- Full observation (48 dims, element by element) ---")
        for idx, (label, desc) in enumerate(obs_labels):
            print(f"    {label:8s} {desc:25s} = {obs[idx]:+.8f}")

        # Run policy
        if policy is not None:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_raw = policy(obs_tensor).detach().numpy().squeeze()
            action_raw = np.clip(action_raw, -clip_actions, clip_actions)

            print(f"\n  --- Action (13 dims) ---")
            for i, jname in enumerate(isaac_joint_order):
                mj_idx = isaac_to_mujoco[i]
                target = action_raw[i] * action_scale + default_angles[mj_idx]
                print(f"    [{i:2d}] {jname:14s}  act={action_raw[i]:+.6f}"
                      f"  target={target:+.6f}  (def={default_angles[mj_idx]:+.4f})")

            # Apply actions
            for i_isaac in range(num_actions):
                mj_idx = isaac_to_mujoco[i_isaac]
                target_dof_pos[mj_idx] = action_raw[i_isaac] * action_scale + default_angles[mj_idx]

    print(f"\n{'='*70}")
    print("Done. Compare these values with dump_v6_isaac_obs.py output.")
    print("Key things to check:")
    print("  1. obs[3:6] gravity should be ~[0, 0, -1] for upright robot")
    print("  2. obs[9:22] joint_pos_rel should be near zero at initial pose")
    print("  3. obs[22:35] joint_vel should be near zero at start")
    print("  4. Action values should be similar between Isaac and MuJoCo")


if __name__ == "__main__":
    main()
