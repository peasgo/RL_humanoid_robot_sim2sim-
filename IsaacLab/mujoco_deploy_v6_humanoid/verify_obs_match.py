"""Verify MuJoCo obs construction matches Isaac Lab — by injecting Isaac Step0 state.

This script:
1. Loads Isaac Step0 state from .npz (exported by dump_v6_isaac_state_step0.py)
2. Forces that exact state into MuJoCo d.qpos / d.qvel (NO warmup, NO simulation)
3. Computes obs using the SAME obs construction code as run_v6_humanoid.py
4. Compares element-by-element with Isaac's obs

This isolates the obs construction function from any dynamics differences.

Usage:
  python verify_obs_match.py isaac_step0_state.npz [--config v6_robot.yaml]
"""

import argparse
import os
import sys
import numpy as np
import yaml
import mujoco


def get_gravity_orientation(quaternion):
    """Compute projected gravity in body frame from quaternion [w,x,y,z].
    Same function as in run_v6_humanoid.py.
    """
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def main():
    parser = argparse.ArgumentParser(description="Verify MuJoCo obs matches Isaac obs (state injection)")
    parser.add_argument("npz_file", type=str, help="Path to isaac_step0_state.npz")
    parser.add_argument("--config", type=str, default="v6_robot.yaml", help="YAML config file")
    args = parser.parse_args()

    # ================================================================
    # Load Isaac state
    # ================================================================
    data = np.load(args.npz_file, allow_pickle=True)
    isaac_root_pos = data["root_pos_w"]           # (3,)
    isaac_root_quat = data["root_quat_w"]         # (4,) wxyz
    isaac_root_lin_vel = data["root_lin_vel_w"]    # (3,) world frame
    isaac_root_ang_vel_b = data["root_ang_vel_b"]  # (3,) body frame
    isaac_projected_gravity = data["projected_gravity"]  # (3,)
    isaac_joint_pos = data["joint_pos_isaac"]      # (13,) Isaac order
    isaac_joint_vel = data["joint_vel_isaac"]      # (13,) Isaac order
    isaac_default_pos = data["default_joint_pos"]  # (13,) Isaac order
    isaac_cmd = data["cmd"]                        # (3,)
    isaac_last_action = data["last_action"]        # (13,)
    isaac_obs = data["obs"]                        # (48,)
    isaac_joint_names = list(data["joint_names"])   # list of str

    print(f"Loaded Isaac state from: {args.npz_file}")
    print(f"  Isaac joint order: {isaac_joint_names}")
    print(f"  root_pos:  {isaac_root_pos}")
    print(f"  root_quat: {isaac_root_quat}")
    print(f"  root_ang_vel_b: {isaac_root_ang_vel_b}")
    print(f"  cmd: {isaac_cmd}")

    # ================================================================
    # Load config
    # ================================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float64)
    default_angles_mj = np.array(config["default_angles"], dtype=np.float64)
    clip_obs = float(config.get("clip_obs", 100.0))

    isaac_joint_order = config.get("isaac_joint_order", isaac_joint_names)

    # ================================================================
    # Load MuJoCo model (only for joint name mapping, no simulation)
    # ================================================================
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    print(f"\n  MuJoCo joint order: {mj_joint_names}")

    # Isaac index -> MuJoCo joint index
    isaac_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac_joint_order],
        dtype=np.int32
    )

    # Default angles in Isaac order (from MuJoCo config)
    default_angles_isaac = default_angles_mj[isaac_to_mujoco]

    # ================================================================
    # Verify default joint positions match
    # ================================================================
    print(f"\n{'='*70}")
    print(f"Step 1: Verify default joint positions")
    print(f"{'='*70}")
    max_def_diff = 0.0
    for i, jname in enumerate(isaac_joint_order):
        diff = abs(default_angles_isaac[i] - isaac_default_pos[i])
        max_def_diff = max(max_def_diff, diff)
        marker = " *** MISMATCH ***" if diff > 1e-6 else ""
        print(f"  [{i:2d}] {jname:14s}  mj_def={default_angles_isaac[i]:+.6f}  "
              f"isaac_def={isaac_default_pos[i]:+.6f}  diff={diff:.2e}{marker}")
    print(f"  Max default pos diff: {max_def_diff:.2e}")
    if max_def_diff > 1e-4:
        print(f"  *** WARNING: Default positions differ significantly! ***")

    # ================================================================
    # Inject Isaac state into MuJoCo (NO simulation, just set state)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"Step 2: Inject Isaac state into MuJoCo qpos/qvel")
    print(f"{'='*70}")

    # qpos[0:3] = root position
    d.qpos[0:3] = isaac_root_pos

    # qpos[3:7] = root quaternion (wxyz)
    d.qpos[3:7] = isaac_root_quat

    # qpos[7:] = joint positions (need to convert Isaac order -> MuJoCo order)
    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        d.qpos[7 + mj_idx] = isaac_joint_pos[i_isaac]

    # qvel[0:3] = root linear velocity (world frame)
    d.qvel[0:3] = isaac_root_lin_vel

    # qvel[3:6] = root angular velocity (body frame in MuJoCo)
    d.qvel[3:6] = isaac_root_ang_vel_b

    # qvel[6:] = joint velocities (Isaac order -> MuJoCo order)
    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        d.qvel[6 + mj_idx] = isaac_joint_vel[i_isaac]

    print(f"  Injected qpos[0:7]: {d.qpos[0:7]}")
    print(f"  Injected qvel[0:6]: {d.qvel[0:6]}")

    # ================================================================
    # Compute obs using MuJoCo obs construction (same as run_v6_humanoid.py)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"Step 3: Compute MuJoCo obs from injected state")
    print(f"{'='*70}")

    quat = d.qpos[3:7].copy()
    omega_body = d.qvel[3:6].copy()
    qj_mujoco = d.qpos[7:].copy()
    dqj_mujoco = d.qvel[6:].copy()

    # Remap to Isaac order
    qj_isaac = qj_mujoco[isaac_to_mujoco]
    dqj_isaac = dqj_mujoco[isaac_to_mujoco]

    # obs[0:3]: base_ang_vel * ang_vel_scale
    omega_obs = omega_body * ang_vel_scale

    # obs[3:6]: projected_gravity_b
    gravity_obs = get_gravity_orientation(quat)

    # obs[6:9]: velocity_commands
    cmd_obs = isaac_cmd * cmd_scale

    # obs[9:22]: joint_pos_rel
    qj_rel = (qj_isaac - default_angles_isaac) * dof_pos_scale

    # obs[22:35]: joint_vel * dof_vel_scale
    dqj_obs = dqj_isaac * dof_vel_scale

    # obs[35:48]: last_action
    last_act = isaac_last_action.copy()

    # Assemble
    mj_obs = np.zeros(48, dtype=np.float64)
    mj_obs[0:3] = omega_obs
    mj_obs[3:6] = gravity_obs
    mj_obs[6:9] = cmd_obs
    mj_obs[9:22] = qj_rel
    mj_obs[22:35] = dqj_obs
    mj_obs[35:48] = last_act

    # Clip
    mj_obs = np.clip(mj_obs, -clip_obs, clip_obs)

    # ================================================================
    # Also compute projected gravity directly from Isaac data for comparison
    # ================================================================
    gravity_from_isaac_quat = get_gravity_orientation(isaac_root_quat)

    # ================================================================
    # Element-by-element comparison
    # ================================================================
    print(f"\n{'='*70}")
    print(f"Step 4: Element-by-element comparison")
    print(f"{'='*70}")

    labels = (
        ["ang_vel_x*0.2", "ang_vel_y*0.2", "ang_vel_z*0.2",
         "grav_x", "grav_y", "grav_z",
         "cmd_vx", "cmd_vy", "cmd_wz"]
        + [f"pos_rel_{jn}" for jn in isaac_joint_order]
        + [f"vel_{jn}*0.05" for jn in isaac_joint_order]
        + [f"last_act_{jn}" for jn in isaac_joint_order]
    )

    max_abs_diff = 0.0
    diffs = []
    print(f"  {'idx':>3s}  {'label':25s}  {'isaac':>12s}  {'mujoco':>12s}  {'diff':>12s}  {'status'}")
    print(f"  {'---':>3s}  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*12}  {'------'}")

    for i in range(48):
        diff = abs(isaac_obs[i] - mj_obs[i])
        diffs.append(diff)
        max_abs_diff = max(max_abs_diff, diff)
        status = "OK" if diff < 1e-4 else ("WARN" if diff < 1e-2 else "*** FAIL ***")
        print(f"  [{i:2d}]  {labels[i]:25s}  {isaac_obs[i]:+12.8f}  {mj_obs[i]:+12.8f}  {diff:12.2e}  {status}")

    diffs = np.array(diffs)

    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Max absolute diff:  {max_abs_diff:.2e}")
    print(f"  Mean absolute diff: {np.mean(diffs):.2e}")
    print(f"  Num elements > 1e-4: {np.sum(diffs > 1e-4)}")
    print(f"  Num elements > 1e-2: {np.sum(diffs > 1e-2)}")
    print(f"  Num elements > 1e-1: {np.sum(diffs > 1e-1)}")

    # Breakdown by group
    print(f"\n  Per-group max diff:")
    print(f"    ang_vel  [0:3]:   {np.max(diffs[0:3]):.2e}")
    print(f"    gravity  [3:6]:   {np.max(diffs[3:6]):.2e}")
    print(f"    cmd      [6:9]:   {np.max(diffs[6:9]):.2e}")
    print(f"    pos_rel  [9:22]:  {np.max(diffs[9:22]):.2e}")
    print(f"    vel      [22:35]: {np.max(diffs[22:35]):.2e}")
    print(f"    last_act [35:48]: {np.max(diffs[35:48]):.2e}")

    # Extra: compare gravity computation
    print(f"\n  Gravity verification:")
    print(f"    Isaac projected_gravity:     {isaac_projected_gravity}")
    print(f"    MuJoCo get_gravity_orient(): {gravity_obs}")
    print(f"    From Isaac quat directly:    {gravity_from_isaac_quat}")
    print(f"    Diff (isaac vs mj_func):     {np.max(np.abs(isaac_projected_gravity - gravity_obs)):.2e}")
    print(f"    Diff (isaac vs isaac_quat):  {np.max(np.abs(isaac_projected_gravity - gravity_from_isaac_quat)):.2e}")

    if max_abs_diff < 1e-4:
        print(f"\n  ✅ PASS: Obs construction is consistent (max diff < 1e-4)")
    elif max_abs_diff < 1e-2:
        print(f"\n  ⚠️  WARN: Small differences detected (max diff < 1e-2)")
    else:
        print(f"\n  ❌ FAIL: Significant differences detected (max diff = {max_abs_diff:.2e})")
        print(f"     Check the elements marked *** FAIL *** above.")


if __name__ == "__main__":
    main()
