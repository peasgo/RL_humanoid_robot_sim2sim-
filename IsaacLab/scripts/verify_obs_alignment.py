"""Verify V6 observation alignment: IsaacLab vs MuJoCo deploy formulas.

Takes the IsaacLab Step0 state dump (isaac_step0_state.npz) and recomputes
observations using the EXACT same formulas as run_v6_humanoid.py (MuJoCo deploy).
Then compares element-by-element.

This proves the observation coordinate systems are aligned WITHOUT needing
to run MuJoCo — it's a pure math verification.

Usage:
    python scripts/verify_obs_alignment.py [--npz path/to/isaac_step0_state.npz]
"""
import numpy as np
import argparse
import os


def get_gravity_orientation_mujoco(quaternion):
    """MuJoCo deploy formula: project gravity [0,0,-1] into body frame.
    From run_v6_humanoid.py line 48-54."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def quat_apply_inverse_isaac(quat_wxyz, vec):
    """IsaacLab formula: quat_apply_inverse(q, v) = R^T * v
    From rigid_object_data.py: projected_gravity_b = quat_apply_inverse(root_link_quat_w, GRAVITY_VEC_W)
    
    quat_apply_inverse rotates vec from world frame to body frame using inverse of quat.
    Equivalent to: conjugate(q) * v * q  (quaternion sandwich with conjugate)
    """
    w, x, y, z = quat_wxyz
    # Rotation matrix from quaternion
    # R^T * v = rotate v from world to body
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y + w*z)
    r02 = 2*(x*z - w*y)
    r10 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z + w*x)
    r20 = 2*(x*z + w*y)
    r21 = 2*(y*z - w*x)
    r22 = 1 - 2*(x*x + y*y)
    # R^T * v
    result = np.array([
        r00*vec[0] + r10*vec[1] + r20*vec[2],
        r01*vec[0] + r11*vec[1] + r21*vec[2],
        r02*vec[0] + r12*vec[1] + r22*vec[2],
    ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="isaac_step0_state.npz")
    parser.add_argument("--step", type=int, default=1, help="Which step to verify (0 or 1)")
    args = parser.parse_args()

    # Find npz file
    npz_path = args.npz
    if not os.path.exists(npz_path):
        npz_path = os.path.join(os.path.dirname(__file__), "..", args.npz)
    if not os.path.exists(npz_path):
        npz_path = os.path.join(os.path.dirname(__file__), "..", "IsaacLab", args.npz)
    
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Auto-detect: new format (step0_*/step1_*) or old format
    step = args.step
    prefix = f"step{step}_" if f"step{step}_obs" in data else ""
    if prefix:
        print(f"Using {prefix}* keys (step {step})")
    else:
        print(f"Using legacy keys (no step prefix)")

    root_quat_w = data[f"{prefix}root_quat_w"]       # (4,) wxyz
    root_ang_vel_b = data[f"{prefix}root_ang_vel_b"]  # (3,) body frame
    projected_gravity = data[f"{prefix}projected_gravity"]  # (3,)
    joint_pos = data[f"{prefix}joint_pos_isaac"]      # (13,) Isaac order
    joint_vel = data[f"{prefix}joint_vel_isaac"]      # (13,) Isaac order
    default_joint_pos = data[f"{prefix}default_joint_pos"]  # (13,) Isaac order
    cmd = data[f"{prefix}cmd"]                        # (3,)
    last_action = data[f"{prefix}last_action"]        # (13,)
    obs_isaac = data[f"{prefix}obs"]                  # (48,)
    joint_names = list(data[f"{prefix}joint_names"])

    # Scales from flat_env_cfg.py ObservationsCfg
    ang_vel_scale = 0.2
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05

    print(f"\n{'='*70}")
    print(f"State from IsaacLab Step0:")
    print(f"  quat (wxyz): {root_quat_w}")
    print(f"  ang_vel_b:   {root_ang_vel_b}")
    print(f"  gravity_b:   {projected_gravity}")
    print(f"  cmd:         {cmd}")
    print(f"  joint_names: {joint_names}")

    # ================================================================
    # 1. Verify projected_gravity_b
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 1: projected_gravity_b")
    print(f"{'='*70}")

    # IsaacLab formula: quat_apply_inverse(root_link_quat_w, [0, 0, -1])
    gravity_isaac_recomputed = quat_apply_inverse_isaac(root_quat_w, np.array([0.0, 0.0, -1.0]))
    
    # MuJoCo formula: get_gravity_orientation(quat)
    gravity_mujoco = get_gravity_orientation_mujoco(root_quat_w)

    print(f"  IsaacLab obs[3:6]:           {projected_gravity}")
    print(f"  Isaac formula recomputed:    {gravity_isaac_recomputed}")
    print(f"  MuJoCo formula:              {gravity_mujoco}")
    print(f"  Isaac vs MuJoCo diff:        {np.abs(gravity_isaac_recomputed - gravity_mujoco)}")
    print(f"  Match: {np.allclose(gravity_isaac_recomputed, gravity_mujoco, atol=1e-7)}")

    # ================================================================
    # 2. Verify angular velocity (body frame)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 2: base_ang_vel (body frame)")
    print(f"{'='*70}")
    print(f"  IsaacLab root_ang_vel_b:     {root_ang_vel_b}")
    print(f"  IsaacLab obs[0:3] / 0.2:     {obs_isaac[0:3] / ang_vel_scale}")
    print(f"  IsaacLab obs[0:3]:           {obs_isaac[0:3]}")
    print(f"  MuJoCo uses d.qvel[3:6] directly (body frame) — same source")
    print(f"  MuJoCo would compute:        {root_ang_vel_b * ang_vel_scale}")
    print(f"  Match: {np.allclose(obs_isaac[0:3], root_ang_vel_b * ang_vel_scale, atol=1e-7)}")

    # ================================================================
    # 3. Verify joint_pos_rel
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 3: joint_pos_rel (Isaac order)")
    print(f"{'='*70}")
    
    jpos_rel_isaac = joint_pos - default_joint_pos
    jpos_rel_mujoco = jpos_rel_isaac * dof_pos_scale  # same formula

    print(f"  Isaac obs[9:22]:  {obs_isaac[9:22]}")
    print(f"  Recomputed:       {jpos_rel_mujoco}")
    print(f"  Diff:             {np.abs(obs_isaac[9:22] - jpos_rel_mujoco)}")
    print(f"  Match: {np.allclose(obs_isaac[9:22], jpos_rel_mujoco, atol=1e-6)}")

    # Show per-joint
    print(f"\n  Per-joint detail:")
    for i, jn in enumerate(joint_names):
        print(f"    [{i:2d}] {jn:14s}  pos={joint_pos[i]:+.6f}  def={default_joint_pos[i]:+.4f}"
              f"  rel={jpos_rel_isaac[i]:+.6f}  obs={obs_isaac[9+i]:+.6f}")

    # ================================================================
    # 4. Verify joint_vel_rel
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 4: joint_vel * dof_vel_scale")
    print(f"{'='*70}")
    
    jvel_scaled = joint_vel * dof_vel_scale
    print(f"  Isaac obs[22:35]: {obs_isaac[22:35]}")
    print(f"  Recomputed:       {jvel_scaled}")
    print(f"  Match: {np.allclose(obs_isaac[22:35], jvel_scaled, atol=1e-6)}")

    # ================================================================
    # 5. Verify last_action
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 5: last_action")
    print(f"{'='*70}")
    print(f"  Isaac obs[35:48]: {obs_isaac[35:48]}")
    print(f"  Expected (zeros): {last_action}")
    print(f"  Match: {np.allclose(obs_isaac[35:48], last_action, atol=1e-7)}")

    # ================================================================
    # 6. Full 48-dim comparison
    # ================================================================
    print(f"\n{'='*70}")
    print(f"FULL 48-dim observation reconstruction (MuJoCo formula)")
    print(f"{'='*70}")

    obs_reconstructed = np.zeros(48, dtype=np.float64)
    # ang_vel * scale
    obs_reconstructed[0:3] = root_ang_vel_b * ang_vel_scale
    # gravity
    obs_reconstructed[3:6] = get_gravity_orientation_mujoco(root_quat_w)
    # cmd
    obs_reconstructed[6:9] = cmd
    # joint_pos_rel
    obs_reconstructed[9:22] = (joint_pos - default_joint_pos) * dof_pos_scale
    # joint_vel * scale
    obs_reconstructed[22:35] = joint_vel * dof_vel_scale
    # last_action
    obs_reconstructed[35:48] = last_action

    labels = (
        ["ang_vel_x*0.2", "ang_vel_y*0.2", "ang_vel_z*0.2",
         "grav_x", "grav_y", "grav_z",
         "cmd_vx", "cmd_vy", "cmd_wz"]
        + [f"pos_rel_{jn}" for jn in joint_names]
        + [f"vel_{jn}*0.05" for jn in joint_names]
        + [f"last_act_{jn}" for jn in joint_names]
    )

    max_diff = 0.0
    for i in range(48):
        diff = abs(obs_isaac[i] - obs_reconstructed[i])
        max_diff = max(max_diff, diff)
        status = "✓" if diff < 1e-5 else "✗ MISMATCH"
        print(f"  obs[{i:2d}] {labels[i]:25s}  isaac={obs_isaac[i]:+.8f}  "
              f"mujoco={obs_reconstructed[i]:+.8f}  diff={diff:.2e}  {status}")

    print(f"\n{'='*70}")
    print(f"RESULT: max element-wise diff = {max_diff:.2e}")
    if max_diff < 1e-5:
        print(f"✓ PASS — IsaacLab and MuJoCo deploy observation formulas are ALIGNED")
    else:
        print(f"✗ FAIL — observation mismatch detected!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
