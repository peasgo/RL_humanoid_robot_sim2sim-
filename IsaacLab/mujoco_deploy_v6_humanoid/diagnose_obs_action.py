#!/usr/bin/env python3
"""Diagnose V6 humanoid sim2sim: build MuJoCo step-0 obs, run policy, compare with IsaacLab.

Usage:
  # 1) First dump IsaacLab state (in IsaacLab/ dir):
  #    python scripts/dump_v6_isaac_state_step0.py --num_envs 1 --headless
  #
  # 2) Then run this script (in mujoco_deploy_v6_humanoid/ dir):
  #    python diagnose_obs_action.py [--isaac-npz ../isaac_step0_state.npz]
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import torch
import yaml


def get_gravity_orientation(quaternion):
    """Compute gravity vector in body frame from world-frame quaternion (w,x,y,z)."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v6_robot.yaml")
    parser.add_argument("--isaac-npz", default=None,
                        help="Path to isaac_step0_state.npz for comparison")
    args = parser.parse_args()

    # ---- Load config ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    xml_path = os.path.join(script_dir, cfg["xml_path"])
    policy_path = cfg["policy_path"]

    isaac_joint_order = cfg["isaac_joint_order"]
    action_joint_order = cfg["action_joint_order"]
    default_angles = np.array(cfg["default_angles"], dtype=np.float64)
    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]
    action_scale = cfg["action_scale"]
    ang_vel_scale = cfg["ang_vel_scale"]
    dof_pos_scale = cfg["dof_pos_scale"]
    dof_vel_scale = cfg["dof_vel_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float64)
    init_height = cfg["init_height"]

    # ---- Load MuJoCo model ----
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Get MuJoCo joint names (skip freejoint)
    mj_joint_names = []
    for i in range(m.njnt):
        jname = m.joint(i).name
        if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    print(f"MuJoCo joint names ({len(mj_joint_names)}): {mj_joint_names}")
    print(f"Isaac joint order  ({len(isaac_joint_order)}): {isaac_joint_order}")
    print(f"Action joint order ({len(action_joint_order)}): {action_joint_order}")

    # ---- Build index mappings ----
    isaac_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac_joint_order], dtype=np.int32
    )
    action_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in action_joint_order], dtype=np.int32
    )

    print(f"\nisaac_to_mujoco:  {isaac_to_mujoco}")
    print(f"action_to_mujoco: {action_to_mujoco}")
    print(f"Same mapping? {np.array_equal(isaac_to_mujoco, action_to_mujoco)}")

    # ---- Default angles in Isaac order ----
    default_angles_isaac = default_angles[isaac_to_mujoco].copy()

    print(f"\nDefault angles (MuJoCo order):")
    for i, jname in enumerate(mj_joint_names):
        print(f"  [{i:2d}] {jname:14s} = {default_angles[i]:+.4f}")

    print(f"\nDefault angles (Isaac order):")
    for i, jname in enumerate(isaac_joint_order):
        print(f"  [{i:2d}] {jname:14s} = {default_angles_isaac[i]:+.4f}")

    # ---- Set default pose (no warmup) ----
    mujoco.mj_resetData(m, d)
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]

    # Set default joint angles
    nq_free = 7
    for i, angle in enumerate(default_angles):
        d.qpos[nq_free + i] = angle

    # Zero velocities
    d.qvel[:] = 0.0

    mujoco.mj_forward(m, d)

    # ---- Read state ----
    quat = d.qpos[3:7].copy()
    omega_body = d.qvel[3:6].copy()
    gravity_b = get_gravity_orientation(quat)

    # Joint positions and velocities in MuJoCo order
    qpos_mj = np.array([d.qpos[nq_free + i] for i in range(len(mj_joint_names))])
    qvel_mj = np.array([d.qvel[6 + i] for i in range(len(mj_joint_names))])

    # Convert to Isaac order
    qpos_isaac = qpos_mj[isaac_to_mujoco]
    qvel_isaac = qvel_mj[isaac_to_mujoco]

    # Relative joint positions
    qpos_rel_isaac = qpos_isaac - default_angles_isaac

    # Command (zero for diagnosis)
    cmd = np.array([0.0, 0.0, 0.0])

    # Last action (zeros at step 0)
    last_act = np.zeros(num_actions)

    # ---- Build observation (same as run_v6_humanoid.py) ----
    obs = np.zeros(num_obs, dtype=np.float32)
    obs[0:3] = omega_body * ang_vel_scale
    obs[3:6] = gravity_b
    obs[6:9] = cmd * cmd_scale
    obs[9:22] = qpos_rel_isaac * dof_pos_scale
    obs[22:35] = qvel_isaac * dof_vel_scale
    obs[35:48] = last_act

    # ---- Print MuJoCo step 0 state ----
    print(f"\n{'='*70}")
    print(f"MuJoCo Step0 State")
    print(f"{'='*70}")
    print(f"  root_pos:           [{d.qpos[0]:.6f}, {d.qpos[1]:.6f}, {d.qpos[2]:.6f}]")
    print(f"  root_quat (wxyz):   {quat}")
    print(f"  root_lin_vel:       {d.qvel[0:3]}")
    print(f"  root_ang_vel_body:  {omega_body}")
    print(f"  projected_gravity:  {gravity_b}")
    print(f"  cmd:                {cmd}")

    print(f"\n  --- Joint state (Isaac order) ---")
    for i, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i]
        print(f"    [{i:2d}] {jname:14s}  pos={qpos_isaac[i]:+.8f}  "
              f"def={default_angles_isaac[i]:+.4f}  "
              f"rel={qpos_rel_isaac[i]:+.8f}  "
              f"vel={qvel_isaac[i]:+.8f}")

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

    # ---- Load policy and run ----
    print(f"\n{'='*70}")
    print(f"Loading policy: {policy_path}")
    print(f"{'='*70}")

    policy = torch.jit.load(policy_path, map_location="cpu")
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()

    with torch.inference_mode():
        action_raw = policy(obs_tensor).squeeze(0).numpy()

    print(f"\n  --- Policy output (action_raw, {len(action_raw)} dims) ---")
    for i, jname in enumerate(isaac_joint_order):
        if i < len(action_raw):
            target = action_raw[i] * action_scale + default_angles_isaac[i]
            print(f"    [{i:2d}] {jname:14s}  raw={action_raw[i]:+.8f}  "
                  f"target={target:+.8f}  "
                  f"delta_from_default={action_raw[i]*action_scale:+.8f}")

    # ---- Apply action to MuJoCo (using action_to_mujoco) ----
    target_dof_pos = default_angles.copy()
    for i_isaac in range(num_actions):
        mj_idx = action_to_mujoco[i_isaac]
        target_dof_pos[mj_idx] = action_raw[i_isaac] * action_scale + default_angles[mj_idx]

    print(f"\n  --- Target positions (MuJoCo order) ---")
    for i, jname in enumerate(mj_joint_names):
        print(f"    [{i:2d}] {jname:14s}  target={target_dof_pos[i]:+.8f}  "
              f"default={default_angles[i]:+.4f}  "
              f"delta={target_dof_pos[i]-default_angles[i]:+.8f}")

    # ---- Compare with IsaacLab dump if available ----
    if args.isaac_npz:
        print(f"\n{'='*70}")
        print(f"Comparing with IsaacLab dump: {args.isaac_npz}")
        print(f"{'='*70}")

        data = np.load(args.isaac_npz, allow_pickle=True)

        # Print available keys
        print(f"  Available keys: {list(data.keys())}")

        isaac_obs = data.get("step0_obs", None)
        isaac_joint_names = data.get("step0_joint_names", None)
        isaac_gravity = data.get("step0_projected_gravity", None)
        isaac_quat = data.get("step0_root_quat_w", None)
        isaac_ang_vel = data.get("step0_root_ang_vel_b", None)
        isaac_joint_pos = data.get("step0_joint_pos_isaac", None)
        isaac_default_pos = data.get("step0_default_joint_pos", None)

        if isaac_joint_names is not None:
            print(f"\n  Isaac joint names: {list(isaac_joint_names)}")
            print(f"  YAML isaac order:  {isaac_joint_order}")
            if list(isaac_joint_names) != isaac_joint_order:
                print(f"  *** MISMATCH! Joint order in .npz differs from YAML! ***")

        if isaac_quat is not None:
            print(f"\n  Isaac root_quat_w: {isaac_quat}")
            print(f"  MuJoCo root_quat:  {quat}")
            print(f"  Diff:              {isaac_quat - quat}")

        if isaac_gravity is not None:
            print(f"\n  Isaac gravity_b:   {isaac_gravity}")
            print(f"  MuJoCo gravity_b:  {gravity_b}")
            print(f"  Diff:              {isaac_gravity - gravity_b}")

        if isaac_ang_vel is not None:
            print(f"\n  Isaac ang_vel_b:   {isaac_ang_vel}")
            print(f"  MuJoCo ang_vel_b:  {omega_body}")
            print(f"  Diff:              {isaac_ang_vel - omega_body}")

        if isaac_joint_pos is not None and isaac_default_pos is not None:
            print(f"\n  --- Joint position comparison (Isaac order) ---")
            for i, jname in enumerate(isaac_joint_order):
                if i < len(isaac_joint_pos):
                    isaac_rel = isaac_joint_pos[i] - isaac_default_pos[i]
                    mj_rel = qpos_rel_isaac[i]
                    print(f"    [{i:2d}] {jname:14s}  "
                          f"isaac_pos={isaac_joint_pos[i]:+.8f}  "
                          f"mj_pos={qpos_isaac[i]:+.8f}  "
                          f"diff={isaac_joint_pos[i]-qpos_isaac[i]:+.8f}  "
                          f"isaac_rel={isaac_rel:+.8f}  "
                          f"mj_rel={mj_rel:+.8f}")

        if isaac_obs is not None:
            print(f"\n  --- Observation comparison (48 dims) ---")
            max_diff = 0.0
            max_diff_idx = -1
            for i in range(min(len(obs), len(isaac_obs))):
                diff = isaac_obs[i] - obs[i]
                flag = " ***" if abs(diff) > 0.01 else ""
                label = labels[i] if i < len(labels) else f"obs[{i}]"
                print(f"    [{i:2d}] {label:25s}  "
                      f"isaac={isaac_obs[i]:+.8f}  "
                      f"mujoco={obs[i]:+.8f}  "
                      f"diff={diff:+.8f}{flag}")
                if abs(diff) > max_diff:
                    max_diff = abs(diff)
                    max_diff_idx = i

            print(f"\n  Max obs diff: {max_diff:.8f} at index {max_diff_idx}")
            if max_diff < 0.01:
                print(f"  ✓ Observations match well!")
            else:
                print(f"  ✗ Significant observation mismatch detected!")

    # ---- Sanity checks ----
    print(f"\n{'='*70}")
    print(f"Sanity Checks")
    print(f"{'='*70}")

    # Check action magnitude
    act_max = np.max(np.abs(action_raw))
    act_mean = np.mean(np.abs(action_raw))
    print(f"  Action max abs:  {act_max:.6f}")
    print(f"  Action mean abs: {act_mean:.6f}")
    if act_max > 10.0:
        print(f"  ✗ Actions seem too large! Policy may be getting bad observations.")
    elif act_max > 3.0:
        print(f"  ⚠ Actions are moderately large.")
    else:
        print(f"  ✓ Action magnitudes look reasonable.")

    # Check gravity vector
    grav_norm = np.linalg.norm(gravity_b)
    print(f"\n  Gravity norm: {grav_norm:.6f} (should be ~1.0)")
    if abs(grav_norm - 1.0) > 0.01:
        print(f"  ✗ Gravity vector not unit length!")

    # Check if gravity points roughly in expected direction
    # For 90° rotation around X: gravity should be [0, -1, 0] in body frame
    expected_grav = np.array([0.0, -1.0, 0.0])
    grav_diff = np.linalg.norm(gravity_b - expected_grav)
    print(f"  Gravity body: {gravity_b}")
    print(f"  Expected:     {expected_grav}")
    print(f"  Diff norm:    {grav_diff:.6f}")
    if grav_diff > 0.1:
        print(f"  ⚠ Gravity direction differs from expected (robot may not be upright)")

    # Check joint position relative values at step 0
    rel_max = np.max(np.abs(qpos_rel_isaac))
    print(f"\n  Max |joint_pos_rel| at step 0: {rel_max:.8f}")
    if rel_max > 0.01:
        print(f"  ⚠ Joint positions differ from defaults at step 0!")
    else:
        print(f"  ✓ Joint positions at defaults.")

    print(f"\n{'='*70}")
    print(f"Done. If robot falls immediately, check:")
    print(f"  1. Observation mismatch (run with --isaac-npz)")
    print(f"  2. Action order (isaac_joint_order vs action_joint_order)")
    print(f"  3. Joint axis sign conventions (MuJoCo vs PhysX)")
    print(f"  4. PD gains and force limits")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
