"""Headless test: diagnose LEFT / RIGHT / TURN commands in V4b sim2sim.

For each test case we:
  1. Reset the robot to the default pose
  2. Apply a fixed cmd for N seconds
  3. Record world-frame X/Y displacement and yaw change
  4. Also log the obs ang_vel and lin_vel fed to the policy

Expected V4 world-frame mapping (robot rotated +90° about X):
  - Forward  = world -Y
  - Left     = world +X
  - Yaw (turn-left) = world -Z rotation  (right-hand rule about world +Z => CCW viewed from above)

Usage:
  python test_lateral_turn.py v4_robot.yaml
"""

import time, os, sys, argparse, yaml
import numpy as np, torch, mujoco

# ── helpers (copied from run_v4_robot.py) ──────────────────────────
def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])

def quat_to_rotmat_wxyz(q):
    w, x, y, z = q
    n = np.sqrt(w*w+x*x+y*y+z*z)
    if n > 0: w,x,y,z = w/n,x/n,y/n,z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def world_to_body(v, q):
    return quat_to_rotmat_wxyz(q).T @ v

def v4_remap_lin_vel(v):
    return np.array([v[2], v[0], v[1]])

def v4_remap_ang_vel(v):
    return np.array([v[0], v[2], v[1]])

def v4_remap_gravity(v):
    return np.array([v[2], v[0], v[1]])

def quat_to_yaw_world(q_wxyz):
    """Extract yaw (rotation about world Z) from quaternion [w,x,y,z].
    For V4 the 'heading' in the ground plane is atan2 of the forward
    direction projected onto world XY.  Forward = local +Z.
    """
    R = quat_to_rotmat_wxyz(q_wxyz)
    fwd_world = R @ np.array([0, 0, 1])  # local +Z in world
    return np.arctan2(fwd_world[0], -fwd_world[1])  # heading: +X=left, -Y=fwd

# ── main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(current_dir, args.config_file) if not os.path.isabs(args.config_file) else args.config_file
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    policy_path   = config["policy_path"]
    sim_dt        = config["simulation_dt"]
    decimation    = config["control_decimation"]
    kps           = np.array(config["kps"], dtype=np.float32)
    kds           = np.array(config["kds"], dtype=np.float32)
    default_angles= np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    lin_vel_scale = config["lin_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale  = config["action_scale"]
    cmd_scale     = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions   = config["num_actions"]
    num_obs       = config["num_obs"]
    v4_remap      = config.get("v4_coordinate_remap", False)
    action_clip   = config.get("action_clip", None)
    if action_clip is not None: action_clip = float(action_clip)
    obs_filter_alpha = float(config.get("obs_filter_alpha", 0.0))
    obs_filter_mode  = str(config.get("obs_filter_mode", "all"))
    action_ramp_steps = int(config.get("action_ramp_steps", 0))
    dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)

    # Load MuJoCo
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # Joint mappings (same as run_v4_robot.py)
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    actuator_to_joint = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jn  = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_to_joint.append(mj_joint_names.index(jn))
    actuator_to_joint = np.array(actuator_to_joint, dtype=np.int32)

    isaac17 = [
        'LHIPp','RHIPp','LHIPy','RHIPy','Waist_2',
        'LSDp','RSDp','LKNEEp','RKNEEP',
        'LSDy','RSDy','LANKLEp','RANKLEp',
        'LARMp','RARMp','LARMAp','RARMAP',
    ]
    isaac16 = [
        'LHIPp','RHIPp','LHIPy','RHIPy',
        'LSDp','RSDp','LKNEEp','RKNEEP',
        'LSDy','RSDy','LANKLEp','RANKLEp',
        'LARMp','RARMp','LARMAp','RARMAP',
    ]
    i17_to_mj = np.array([mj_joint_names.index(j) for j in isaac17], dtype=np.int32)
    i16_to_mj = np.array([mj_joint_names.index(j) for j in isaac16], dtype=np.int32)
    waist_mj  = mj_joint_names.index('Waist_2')
    waist_def = default_angles[waist_mj]

    dq_sign = np.ones(17, dtype=np.float32)
    if isinstance(dq_sign_fixes_cfg, dict):
        for jn, s in dq_sign_fixes_cfg.items():
            if jn in isaac17:
                dq_sign[isaac17.index(jn)] = float(s)

    joint_limits = {}
    for i, jn in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if m.jnt_limited[jid]:
            joint_limits[jn] = (float(m.jnt_range[jid,0]), float(m.jnt_range[jid,1]))

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"Policy loaded: {policy_path}")

    init_height = 0.22
    init_quat   = np.array([0.70710678, 0.70710678, 0.0, 0.0])

    # ── Test cases ─────────────────────────────────────────────────
    test_cases = [
        ("FORWARD",    np.array([0.5,  0.0,  0.0], dtype=np.float32)),
        ("BACKWARD",   np.array([-0.3, 0.0,  0.0], dtype=np.float32)),
        ("LEFT",       np.array([0.0,  0.3,  0.0], dtype=np.float32)),
        ("RIGHT",      np.array([0.0, -0.3,  0.0], dtype=np.float32)),
        ("TURN_LEFT",  np.array([0.0,  0.0,  0.5], dtype=np.float32)),
        ("TURN_RIGHT", np.array([0.0,  0.0, -0.5], dtype=np.float32)),
    ]

    test_duration = 5.0   # seconds of policy control per test
    warmup_secs   = 3.0

    for test_name, cmd in test_cases:
        # Reset
        mujoco.mj_resetData(m, d)
        d.qpos[2]   = init_height
        d.qpos[3:7] = init_quat
        d.qpos[7:]  = default_angles
        d.qvel[:]   = 0

        target_dof = default_angles.copy()
        action16   = np.zeros(num_actions, dtype=np.float32)
        action16_prev = np.zeros(num_actions, dtype=np.float32)
        prev_obs   = np.zeros(num_obs, dtype=np.float32)
        obs        = np.zeros(num_obs, dtype=np.float32)
        counter    = 0
        policy_step = 0

        # Warmup (PD hold, no policy)
        warmup_steps = int(warmup_secs / sim_dt)
        for _ in range(warmup_steps):
            d.ctrl[:] = target_dof[actuator_to_joint]
            mujoco.mj_step(m, d)

        # Record start pose
        start_pos  = d.qpos[0:3].copy()
        start_yaw  = quat_to_yaw_world(d.qpos[3:7])
        d.qvel[:]  = 0
        counter    = 0

        total_steps = int(test_duration / sim_dt)
        log_lines = []

        for step in range(total_steps):
            d.ctrl[:] = target_dof[actuator_to_joint]
            mujoco.mj_step(m, d)
            counter += 1

            if counter % decimation == 0:
                quat = d.qpos[3:7]
                lin_vel_w = d.qvel[0:3].copy()
                ang_vel_w = d.qvel[3:6].copy()

                lin_vel_b = world_to_body(lin_vel_w, quat)
                # NOTE: MuJoCo qvel[3:6] is ALREADY body-frame angular velocity
                # but the deploy code does world_to_body (double rotation).
                # We replicate the CURRENT deploy code behaviour here:
                omega_double = world_to_body(ang_vel_w, quat)  # deploy code (double rot)
                omega_correct = ang_vel_w.copy()                # correct (no transform)

                qj_mj  = d.qpos[7:].copy()
                dqj_mj = d.qvel[6:].copy()
                qj17    = qj_mj[i17_to_mj]
                dqj17   = dqj_mj[i17_to_mj] * dq_sign
                def17   = default_angles[i17_to_mj]
                grav    = get_gravity_orientation(quat)

                if v4_remap:
                    lv  = v4_remap_lin_vel(lin_vel_b)
                    av  = v4_remap_ang_vel(omega_double)   # current deploy
                    av_correct = v4_remap_ang_vel(omega_correct)  # what it should be
                    gv  = v4_remap_gravity(grav)
                else:
                    lv  = lin_vel_b
                    av  = omega_double
                    av_correct = omega_correct
                    gv  = grav

                lv_s = lv * lin_vel_scale
                av_s = av * ang_vel_scale
                qj   = (qj17 - def17) * dof_pos_scale
                dqj  = dqj17 * dof_vel_scale

                obs[0:3]   = lv_s
                obs[3:6]   = av_s
                obs[6:9]   = gv
                obs[9:12]  = cmd * cmd_scale
                obs[12:29] = qj
                obs[29:46] = dqj
                obs[46:62] = action16.astype(np.float32)

                if obs_filter_alpha > 0 and policy_step > 0:
                    if obs_filter_mode == "vel_only":
                        obs[0:6]   = obs_filter_alpha * prev_obs[0:6]   + (1-obs_filter_alpha) * obs[0:6]
                        obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1-obs_filter_alpha) * obs[29:46]
                    else:
                        obs[:] = obs_filter_alpha * prev_obs + (1-obs_filter_alpha) * obs
                prev_obs[:] = obs

                # Log every 0.5s
                t = step * sim_dt
                if policy_step % max(1, int(0.5 / (sim_dt * decimation))) == 0:
                    pos = d.qpos[0:3]
                    yaw = quat_to_yaw_world(d.qpos[3:7])
                    log_lines.append(
                        f"  t={t:5.2f}s  world_pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})  "
                        f"dyaw={np.degrees(yaw-start_yaw):+.1f}°  "
                        f"obs_linvel={obs[0:3]}  obs_angvel={obs[3:6]}  "
                        f"angvel_correct={av_correct*ang_vel_scale}"
                    )

                # Policy
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                action16 = policy(obs_t).detach().numpy().squeeze()
                if action_clip is not None:
                    action16 = np.clip(action16, -action_clip, action_clip)
                if action_ramp_steps > 0 and policy_step < action_ramp_steps:
                    action16 *= float(policy_step) / float(action_ramp_steps)
                action16_prev[:] = action16
                policy_step += 1

                target_dof[waist_mj] = waist_def
                for i16 in range(num_actions):
                    mj_idx = i16_to_mj[i16]
                    target_dof[mj_idx] = action16[i16] * action_scale + default_angles[mj_idx]
                for i, jn in enumerate(mj_joint_names):
                    if jn in joint_limits:
                        lo, hi = joint_limits[jn]
                        target_dof[i] = np.clip(target_dof[i], lo, hi)

        # Final result
        end_pos = d.qpos[0:3].copy()
        end_yaw = quat_to_yaw_world(d.qpos[3:7])
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dz = end_pos[2] - start_pos[2]
        dyaw = np.degrees(end_yaw - start_yaw)

        print(f"\n{'='*70}")
        print(f"TEST: {test_name}  cmd={cmd}")
        print(f"  World displacement: dX={dx:+.4f}  dY={dy:+.4f}  dZ={dz:+.4f}")
        print(f"  Yaw change: {dyaw:+.1f}°")
        print(f"  Expected:")
        if test_name == "FORWARD":
            print(f"    dY < 0 (world -Y = forward)")
        elif test_name == "BACKWARD":
            print(f"    dY > 0 (world +Y = backward)")
        elif test_name == "LEFT":
            print(f"    dX > 0 (world +X = left)")
        elif test_name == "RIGHT":
            print(f"    dX < 0 (world -X = right)")
        elif test_name == "TURN_LEFT":
            print(f"    dyaw > 0 (CCW viewed from above)")
        elif test_name == "TURN_RIGHT":
            print(f"    dyaw < 0 (CW viewed from above)")

        ok = False
        if test_name == "FORWARD":   ok = dy < -0.05
        elif test_name == "BACKWARD": ok = dy > 0.05
        elif test_name == "LEFT":     ok = dx > 0.05
        elif test_name == "RIGHT":    ok = dx < -0.05
        elif test_name == "TURN_LEFT":  ok = dyaw > 5
        elif test_name == "TURN_RIGHT": ok = dyaw < -5

        print(f"  Result: {'PASS ✓' if ok else 'FAIL ✗'}")
        print(f"  Timeline:")
        for line in log_lines:
            print(line)

    print(f"\n{'='*70}")
    print("DONE. Check FAIL cases above for diagnosis.")

if __name__ == "__main__":
    main()
