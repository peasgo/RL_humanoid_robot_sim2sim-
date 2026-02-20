"""Diagnose obs construction: compare MuJoCo joint angles with URDF/IsaacLab conventions.

Key test: Set MuJoCo to default pose, check if joint axes produce the same
physical motion as IsaacLab when perturbed by +0.1 rad.

Also: verify that the gravity projection and angular velocity are correct
by comparing with known analytical values.
"""
import mujoco
import numpy as np
import yaml
import os


def get_gravity_orientation(quaternion):
    """Projected gravity in body frame from quat [w,x,y,z]."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def quat_to_rotmat(q):
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix (body-to-world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "v6_robot.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    default_angles = np.array(config["default_angles"], dtype=np.float64)
    isaac_joint_order = config["isaac_joint_order"]

    # Get MuJoCo hinge joint names
    mj_joint_names = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            mj_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))

    isaac_to_mujoco = np.array([mj_joint_names.index(n) for n in isaac_joint_order])
    default_angles_isaac = default_angles[isaac_to_mujoco]

    print(f"MuJoCo joints: {mj_joint_names}")
    print(f"Isaac joints:  {isaac_joint_order}")
    print(f"isaac_to_mujoco: {isaac_to_mujoco}")
    print(f"default_angles (MJ order): {default_angles}")
    print(f"default_angles (Isaac order): {default_angles_isaac}")

    # ================================================================
    # Test 1: Gravity at identity and at init quaternion
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 1: Gravity projection")
    print(f"{'='*60}")

    # Identity quaternion (upright)
    q_identity = [1, 0, 0, 0]
    g_identity = get_gravity_orientation(q_identity)
    print(f"  Identity quat {q_identity}: gravity = {g_identity}  (expect [0, 0, -1])")

    # Init quaternion (90° around Z)
    q_init = [0.7071068, 0.0, 0.0, 0.7071068]
    g_init = get_gravity_orientation(q_init)
    print(f"  Init quat {q_init}: gravity = {g_init}  (expect [0, 0, -1])")

    # Tilted 5° around body X
    angle = np.radians(5)
    q_tilt_x = [np.cos(angle/2), np.sin(angle/2), 0, 0]
    g_tilt_x = get_gravity_orientation(q_tilt_x)
    print(f"  Tilt 5° around X: gravity = {g_tilt_x}  (expect [0, +sin5, -cos5] = [0, {np.sin(angle):.4f}, {-np.cos(angle):.4f}])")

    # ================================================================
    # Test 2: Angular velocity frame
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 2: Angular velocity frame verification")
    print(f"{'='*60}")

    d.qpos[2] = 0.55
    d.qpos[3:7] = q_init
    d.qpos[7:] = default_angles
    d.qvel[:] = 0

    # Set angular velocity in body frame
    test_omega = np.array([0.5, 0.3, 0.1])
    d.qvel[3:6] = test_omega
    mujoco.mj_forward(m, d)

    # Get cvel (world frame angular velocity)
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cvel = d.cvel[base_body_id].copy()
    omega_world_from_cvel = cvel[0:3]

    R = quat_to_rotmat(d.qpos[3:7])
    omega_world_expected = R @ test_omega

    print(f"  Set qvel[3:6] = {test_omega}")
    print(f"  cvel omega (world): {omega_world_from_cvel}")
    print(f"  R @ qvel (if body): {omega_world_expected}")
    print(f"  Error (body hyp):   {np.linalg.norm(omega_world_from_cvel - omega_world_expected):.8f}")
    print(f"  Error (world hyp):  {np.linalg.norm(omega_world_from_cvel - test_omega):.8f}")

    # ================================================================
    # Test 3: Joint axis comparison - perturb each joint and check foot positions
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 3: Joint perturbation test (check physical effect)")
    print(f"{'='*60}")

    # Get foot body IDs
    foot_names = ["RANKLEy", "LANKLEy"]
    foot_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names]

    # Reference: default pose foot positions
    d.qpos[2] = 0.55
    d.qpos[3:7] = q_init
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    ref_foot_pos = {}
    for fn, fid in zip(foot_names, foot_ids):
        ref_foot_pos[fn] = d.xpos[fid].copy()
        print(f"  Default {fn} pos: {ref_foot_pos[fn]}")

    print(f"\n  Perturbing each joint by +0.1 rad (MuJoCo convention):")
    print(f"  {'Joint':14s} {'Isaac idx':>9s} {'MJ idx':>6s} {'R_foot_dx':>10s} {'R_foot_dy':>10s} {'R_foot_dz':>10s} {'L_foot_dx':>10s} {'L_foot_dy':>10s} {'L_foot_dz':>10s}")

    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]

        # Reset to default
        d.qpos[7:] = default_angles.copy()
        # Perturb this joint by +0.1
        d.qpos[7 + mj_idx] = default_angles[mj_idx] + 0.1
        mujoco.mj_forward(m, d)

        r_delta = d.xpos[foot_ids[0]] - ref_foot_pos["RANKLEy"]
        l_delta = d.xpos[foot_ids[1]] - ref_foot_pos["LANKLEy"]

        print(f"  {jname:14s} {i_isaac:9d} {mj_idx:6d} {r_delta[0]:+10.4f} {r_delta[1]:+10.4f} {r_delta[2]:+10.4f} {l_delta[0]:+10.4f} {l_delta[1]:+10.4f} {l_delta[2]:+10.4f}")

    # ================================================================
    # Test 4: Check MuJoCo joint ranges vs URDF
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 4: Joint ranges and axes")
    print(f"{'='*60}")

    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        axis = m.jnt_axis[jid].copy()
        jrange = m.jnt_range[jid].copy() if m.jnt_limited[jid] else [float('nan'), float('nan')]
        print(f"  [{i_isaac:2d}] {jname:14s}  MJ[{mj_idx:2d}]  axis={axis}  range=[{jrange[0]:+.3f}, {jrange[1]:+.3f}]  default={default_angles[mj_idx]:+.3f}")

    # ================================================================
    # Test 5: Simulate 1 second with default pose (no policy) - check stability
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 5: Stability test - hold default pose for 2 seconds")
    print(f"{'='*60}")

    d.qpos[0:3] = [0, 0, 0.55]
    d.qpos[3:7] = q_init
    d.qpos[7:] = default_angles
    d.qvel[:] = 0

    actuator_to_joint = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_to_joint.append(mj_joint_names.index(jn))
    actuator_to_joint = np.array(actuator_to_joint)

    target = default_angles.copy()

    for step in range(int(2.0 / m.opt.timestep)):
        d.ctrl[:] = target[actuator_to_joint]
        mujoco.mj_step(m, d)

        if step % 200 == 0:
            pos = d.qpos[0:3]
            quat = d.qpos[3:7]
            grav = get_gravity_orientation(quat)
            print(f"  t={step*m.opt.timestep:5.2f}s  h={pos[2]:.4f}m  pos=({pos[0]:+.4f},{pos[1]:+.4f})  "
                  f"grav=({grav[0]:+.4f},{grav[1]:+.4f},{grav[2]:+.4f})  ncon={d.ncon}")

    pos = d.qpos[0:3]
    print(f"\n  Final: h={pos[2]:.4f}m  pos=({pos[0]:+.4f},{pos[1]:+.4f})")
    print(f"  Drift: dx={pos[0]:+.6f}  dy={pos[1]:+.6f}")


if __name__ == "__main__":
    main()
