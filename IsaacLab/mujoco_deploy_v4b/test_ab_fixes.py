"""Targeted A/B test: compare current deploy code vs two potential fixes.

Fix A: Correct angular velocity (no double rotation)
Fix B: Correct action_scale (0.5 instead of 0.25)
Fix C: Both fixes combined

Tests LEFT, RIGHT, TURN_LEFT, TURN_RIGHT commands.
"""

import os, sys, argparse, yaml
import numpy as np, torch, mujoco

# ── helpers ────────────────────────────────────────────────────────
def get_gravity_orientation(q):
    w, x, y, z = q
    return np.array([-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))])

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

def quat_to_yaw_world(q):
    R = quat_to_rotmat_wxyz(q)
    fwd = R @ np.array([0, 0, 1])
    return np.arctan2(fwd[0], -fwd[1])


def run_test(m_template, config, policy, test_name, cmd, fix_angvel=False, fix_action_scale=False):
    """Run a single test and return (dx, dy, dyaw)."""
    sim_dt     = config["simulation_dt"]
    decimation = config["control_decimation"]
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale  = config["ang_vel_scale"]
    lin_vel_scale  = config["lin_vel_scale"]
    dof_pos_scale  = config["dof_pos_scale"]
    dof_vel_scale  = config["dof_vel_scale"]
    
    action_scale = 0.5 if fix_action_scale else config["action_scale"]
    
    cmd_scale    = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions  = config["num_actions"]
    num_obs      = config["num_obs"]
    v4_remap     = config.get("v4_coordinate_remap", False)
    action_clip  = config.get("action_clip", None)
    if action_clip is not None: action_clip = float(action_clip)
    obs_filter_alpha = float(config.get("obs_filter_alpha", 0.0))
    obs_filter_mode  = str(config.get("obs_filter_mode", "all"))
    action_ramp_steps = int(config.get("action_ramp_steps", 0))
    dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)

    m = mujoco.MjModel.from_xml_path(config["_xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    mj_joint_names = []
    for jid in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jn)

    act_to_jnt = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jn  = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        act_to_jnt.append(mj_joint_names.index(jn))
    act_to_jnt = np.array(act_to_jnt, dtype=np.int32)

    isaac17 = ['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2',
               'LSDp','RSDp','LKNEEp','RKNEEP',
               'LSDy','RSDy','LANKLEp','RANKLEp',
               'LARMp','RARMp','LARMAp','RARMAP']
    isaac16 = ['LHIPp','RHIPp','LHIPy','RHIPy',
               'LSDp','RSDp','LKNEEp','RKNEEP',
               'LSDy','RSDy','LANKLEp','RANKLEp',
               'LARMp','RARMp','LARMAp','RARMAP']
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

    init_height = 0.22
    init_quat   = np.array([0.70710678, 0.70710678, 0.0, 0.0])

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

    # Warmup
    for _ in range(int(3.0 / sim_dt)):
        d.ctrl[:] = target_dof[act_to_jnt]
        mujoco.mj_step(m, d)

    start_pos = d.qpos[0:3].copy()
    start_yaw = quat_to_yaw_world(d.qpos[3:7])
    d.qvel[:] = 0
    counter   = 0

    test_duration = 5.0
    total_steps = int(test_duration / sim_dt)

    for step in range(total_steps):
        d.ctrl[:] = target_dof[act_to_jnt]
        mujoco.mj_step(m, d)
        counter += 1

        if counter % decimation == 0:
            quat = d.qpos[3:7]
            lin_vel_w = d.qvel[0:3].copy()
            ang_vel_w = d.qvel[3:6].copy()

            lin_vel_b = world_to_body(lin_vel_w, quat)
            
            if fix_angvel:
                # CORRECT: MuJoCo qvel[3:6] is already body-frame
                omega = ang_vel_w.copy()
            else:
                # CURRENT BUG: double rotation
                omega = world_to_body(ang_vel_w, quat)

            qj_mj  = d.qpos[7:].copy()
            dqj_mj = d.qvel[6:].copy()
            qj17    = qj_mj[i17_to_mj]
            dqj17   = dqj_mj[i17_to_mj] * dq_sign
            def17   = default_angles[i17_to_mj]
            grav    = get_gravity_orientation(quat)

            if v4_remap:
                lv = v4_remap_lin_vel(lin_vel_b)
                av = v4_remap_ang_vel(omega)
                gv = v4_remap_gravity(grav)
            else:
                lv, av, gv = lin_vel_b, omega, grav

            obs[0:3]   = lv * lin_vel_scale
            obs[3:6]   = av * ang_vel_scale
            obs[6:9]   = gv
            obs[9:12]  = cmd * cmd_scale
            obs[12:29] = (qj17 - def17) * dof_pos_scale
            obs[29:46] = dqj17 * dof_vel_scale
            obs[46:62] = action16.astype(np.float32)

            if obs_filter_alpha > 0 and policy_step > 0:
                if obs_filter_mode == "vel_only":
                    obs[0:6]   = obs_filter_alpha * prev_obs[0:6]   + (1-obs_filter_alpha) * obs[0:6]
                    obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1-obs_filter_alpha) * obs[29:46]
                else:
                    obs[:] = obs_filter_alpha * prev_obs + (1-obs_filter_alpha) * obs
            prev_obs[:] = obs

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

    end_pos = d.qpos[0:3].copy()
    end_yaw = quat_to_yaw_world(d.qpos[3:7])
    return (end_pos[0]-start_pos[0], end_pos[1]-start_pos[1], np.degrees(end_yaw-start_yaw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(current_dir, args.config_file)
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)
    config["_xml_path"] = xml_path

    policy = torch.jit.load(config["policy_path"])

    test_cases = [
        ("LEFT",       np.array([0.0,  0.3,  0.0], dtype=np.float32)),
        ("RIGHT",      np.array([0.0, -0.3,  0.0], dtype=np.float32)),
        ("TURN_LEFT",  np.array([0.0,  0.0,  0.5], dtype=np.float32)),
        ("TURN_RIGHT", np.array([0.0,  0.0, -0.5], dtype=np.float32)),
        ("FORWARD",    np.array([0.5,  0.0,  0.0], dtype=np.float32)),
    ]

    variants = [
        ("CURRENT (buggy)",       False, False),
        ("FIX_A: angvel only",    True,  False),
        ("FIX_B: action_scale",   False, True),
        ("FIX_C: both",           True,  True),
    ]

    print(f"{'Test':<14} {'Variant':<24} {'dX':>8} {'dY':>8} {'dYaw':>8}  Expected")
    print("-" * 90)

    for test_name, cmd in test_cases:
        for var_name, fix_av, fix_as in variants:
            dx, dy, dyaw = run_test(None, config, policy, test_name, cmd,
                                     fix_angvel=fix_av, fix_action_scale=fix_as)
            
            # Expected direction
            if test_name == "LEFT":
                exp = "dX>0"
                ok = dx > 0.05
            elif test_name == "RIGHT":
                exp = "dX<0"
                ok = dx < -0.05
            elif test_name == "TURN_LEFT":
                exp = "dyaw>0"
                ok = dyaw > 5
            elif test_name == "TURN_RIGHT":
                exp = "dyaw<0"
                ok = dyaw < -5
            elif test_name == "FORWARD":
                exp = "dY<0"
                ok = dy < -0.05
            else:
                exp = "?"
                ok = False

            mark = "✓" if ok else "✗"
            print(f"{test_name:<14} {var_name:<24} {dx:+8.3f} {dy:+8.3f} {dyaw:+8.1f}°  {exp} {mark}")
        print()

if __name__ == "__main__":
    main()
