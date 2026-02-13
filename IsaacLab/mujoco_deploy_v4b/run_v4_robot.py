import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os
import argparse
import threading
import sys


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


def v4_remap_lin_vel(lin_vel_body):
    return np.array([lin_vel_body[2], lin_vel_body[0], lin_vel_body[1]])


def v4_remap_ang_vel(ang_vel_body):
    return np.array([ang_vel_body[0], ang_vel_body[2], ang_vel_body[1]])


def v4_remap_gravity(gravity_body):
    return np.array([gravity_body[2], gravity_body[0], gravity_body[1]])


class KeyboardController:
    """Non-blocking keyboard input for velocity commands."""
    def __init__(self, cmd, step_fwd=0.1, step_lat=0.1, step_yaw=0.1):
        self.cmd = cmd
        self.step_fwd = step_fwd
        self.step_lat = step_lat
        self.step_yaw = step_yaw
        self.reset_requested = False
        self._running = True
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

    def _input_loop(self):
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    ch = sys.stdin.read(1).lower()
                    if ch == 'w':
                        self.cmd[0] = min(self.cmd[0] + self.step_fwd, 1.0)
                    elif ch == 's':
                        self.cmd[0] = max(self.cmd[0] - self.step_fwd, -0.5)
                    elif ch == 'a':
                        self.cmd[1] = min(self.cmd[1] + self.step_lat, 0.5)
                    elif ch == 'd':
                        self.cmd[1] = max(self.cmd[1] - self.step_lat, -0.5)
                    elif ch == 'q':
                        self.cmd[2] = min(self.cmd[2] + self.step_yaw, 1.0)
                    elif ch == 'e':
                        self.cmd[2] = max(self.cmd[2] - self.step_yaw, -1.0)
                    elif ch == ' ':
                        self.cmd[:] = 0.0
                    elif ch == 'r':
                        self.reset_requested = True
                    elif ch == '\x03':
                        self._running = False
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    def stop(self):
        self._running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Quadruped Sim2Sim MuJoCo Deployment (v4b)")
    parser.add_argument("config_file", type=str, help="config file name (e.g., v4_robot.yaml)")
    parser.add_argument("--no-policy", action="store_true", help="Disable policy, only PD hold default stance")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable keyboard control")
    args = parser.parse_args()


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
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        use_tanh_action = bool(config.get("use_tanh_action", False))
        action_clip = config.get("action_clip", None)
        if action_clip is not None:
            action_clip = float(action_clip)


        action_filter_alpha = float(config.get("action_filter_alpha", 0.0))


        obs_filter_alpha = float(config.get("obs_filter_alpha", 0.0))
        obs_filter_mode = str(config.get("obs_filter_mode", "all"))


        action_ramp_steps = int(config.get("action_ramp_steps", 0))

        dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)


        v4_coordinate_remap = config.get("v4_coordinate_remap", False)


        mass_scale = float(config.get("mass_scale", 1.0))


    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt


    if mass_scale != 1.0:
        original_mass = sum(m.body_mass[bi] for bi in range(m.nbody))
        for bi in range(m.nbody):
            m.body_mass[bi] *= mass_scale
            m.body_inertia[bi] *= mass_scale
        scaled_mass = sum(m.body_mass[bi] for bi in range(m.nbody))
        print(f"Mass scaling: {mass_scale}x  ({original_mass:.3f}kg -> {scaled_mass:.3f}kg)")


    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    num_mj_joints = len(mj_joint_names)
    print(f"MuJoCo joint order ({num_mj_joints} joints): {mj_joint_names}")


    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)


    init_height = float(config.get("init_height", 0.22))
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles


    warmup_steps = int(5.0 / simulation_dt)


    isaac17_joint_order = [
        'LHIPp',
        'RHIPp',
        'LHIPy',
        'RHIPy',
        'Waist_2',
        'LSDp',
        'RSDp',
        'LKNEEp',
        'RKNEEP',
        'LSDy',
        'RSDy',
        'LANKLEp',
        'RANKLEp',
        'LARMp',
        'RARMp',
        'LARMAp',
        'RARMAP',
    ]


    isaac16_action_order = [
        'LHIPp',
        'RHIPp',
        'LHIPy',
        'RHIPy',
        'LSDp',
        'RSDp',
        'LKNEEp',
        'RKNEEP',
        'LSDy',
        'RSDy',
        'LANKLEp',
        'RANKLEp',
        'LARMp',
        'RARMp',
        'LARMAp',
        'RARMAP',
    ]

    print(f"Isaac17 joint order ({len(isaac17_joint_order)} joints): {isaac17_joint_order}")
    print(f"Isaac16 action order ({len(isaac16_action_order)} joints): {isaac16_action_order}")


    for jname in isaac17_joint_order:
        if jname not in mj_joint_names:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model. Available: {mj_joint_names}")


    isaac17_to_mujoco17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac17_joint_order],
        dtype=np.int32
    )

    print(f"\nMapping 1 - Isaac17 <-> MuJoCo17 (for obs):")
    print(f"  isaac17_to_mujoco17: {isaac17_to_mujoco17}")
    for i17, jname in enumerate(isaac17_joint_order):
        mj_idx = isaac17_to_mujoco17[i17]
        print(f"  Isaac17[{i17:2d}] {jname:12s} <-> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]}")


    isaac16_action_to_mj17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac16_action_order],
        dtype=np.int32
    )

    print(f"\nMapping 2 - Isaac16 <-> MuJoCo17 (for action):")
    print(f"  isaac16_action_to_mj17: {isaac16_action_to_mj17}")
    for i16, jname in enumerate(isaac16_action_order):
        mj_idx = isaac16_action_to_mj17[i16]
        print(f"  Isaac16[{i16:2d}] {jname:12s} -> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]}")


    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]
    print(f"\nWaist_2: MJ idx={waist_mj_idx}, locked at {waist_default:.4f} rad")


    dq_sign = np.ones(len(isaac17_joint_order), dtype=np.float32)
    if isinstance(dq_sign_fixes_cfg, dict):
        for jn, s in dq_sign_fixes_cfg.items():
            if jn in isaac17_joint_order:
                dq_sign[isaac17_joint_order.index(jn)] = float(s)


    joint_limits = {}
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if m.jnt_limited[jid]:
            joint_limits[jname] = (float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1]))


    policy = None
    if args.no_policy:
        print("\n--no-policy flag set. Running in STANCE MODE (PD hold only, no policy).")
    elif os.path.exists(policy_path) and os.path.isfile(policy_path):
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        print("Policy loaded successfully!")
    else:
        print(f"\nWARNING: No policy found at {policy_path}. Running in DEBUG MODE.")


    action_isaac16_raw = np.zeros(num_actions, dtype=np.float32)
    action_isaac16_exec = np.zeros(num_actions, dtype=np.float32)
    action_isaac16_prev = np.zeros(num_actions, dtype=np.float32)
    prev_obs = np.zeros(num_obs, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    policy_step_count = 0

    print(f"\n{'='*60}")
    print(f"V4 Quadruped Sim2Sim Configuration (v4b):")
    print(f"  num_actions: {num_actions}")
    print(f"  num_obs: {num_obs}")
    print(f"  action_scale: {action_scale}")
    print(f"  action_clip: {action_clip}")
    print(f"  action_filter_alpha: {action_filter_alpha}")
    print(f"  obs_filter_alpha: {obs_filter_alpha} (mode={obs_filter_mode})")
    print(f"  action_ramp_steps: {action_ramp_steps}")
    print(f"  mass_scale: {mass_scale}")
    print(f"  control_decimation: {control_decimation}")
    print(f"  simulation_dt: {simulation_dt}")
    print(f"  v4_coordinate_remap: {v4_coordinate_remap}")
    print(f"  cmd_init: {cmd}")
    print(f"{'='*60}\n")


    kb_controller = None
    if not args.no_keyboard:
        kb_controller = KeyboardController(cmd)
        print("  Keyboard: W/S=fwd/back, A/D=left/right, Q/E=turn, SPACE=stop, R=reset")

    print("\nStarting simulation...")

    with mujoco.viewer.launch_passive(m, d) as viewer:


        print(f"Warmup: settling for {warmup_steps * simulation_dt:.1f}s...")
        for ws in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if ws % 100 == 0:
                viewer.sync()
        viewer.sync()
        print(f"Warmup done. Height: {d.qpos[2]:.4f}m")


        counter = 0
        d.qvel[:] = 0

        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))
        last_print_time = 0.0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()


            if kb_controller and kb_controller.reset_requested:
                kb_controller.reset_requested = False
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action_isaac16_raw[:] = 0
                action_isaac16_exec[:] = 0
                action_isaac16_prev[:] = 0
                prev_obs[:] = 0
                counter = 0
                policy_step_count = 0
                start = time.time()
                print("\n[RESET] Robot pose reset!")
                continue


            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]


            mujoco.mj_step(m, d)
            counter += 1


            if counter % render_interval == 0:
                viewer.sync()


            if counter % control_decimation == 0 and policy is not None:
                quat = d.qpos[3:7]
                base_lin_vel_world = d.qvel[0:3].copy()
                base_ang_vel_world = d.qvel[3:6].copy()


                base_lin_vel = world_to_body(base_lin_vel_world, quat)


                omega = world_to_body(base_ang_vel_world, quat)


                qj_mujoco = d.qpos[7:].copy()
                dqj_mujoco = d.qvel[6:].copy()


                qj_isaac17 = qj_mujoco[isaac17_to_mujoco17]
                dqj_isaac17 = dqj_mujoco[isaac17_to_mujoco17] * dq_sign


                default_angles_isaac17 = default_angles[isaac17_to_mujoco17]


                gravity_orientation = get_gravity_orientation(quat)


                if v4_coordinate_remap:
                    base_lin_vel_obs = v4_remap_lin_vel(base_lin_vel)
                    omega_obs = v4_remap_ang_vel(omega)
                    gravity_obs = v4_remap_gravity(gravity_orientation)
                else:
                    base_lin_vel_obs = base_lin_vel
                    omega_obs = omega
                    gravity_obs = gravity_orientation


                base_lin_vel_obs = base_lin_vel_obs * lin_vel_scale
                omega_obs = omega_obs * ang_vel_scale
                qj = (qj_isaac17 - default_angles_isaac17) * dof_pos_scale
                dqj = dqj_isaac17 * dof_vel_scale


                obs[0:3] = base_lin_vel_obs
                obs[3:6] = omega_obs
                obs[6:9] = gravity_obs
                obs[9:12] = cmd * cmd_scale
                obs[12:29] = qj
                obs[29:46] = dqj
                obs[46:62] = action_isaac16_raw.astype(np.float32)


                if obs_filter_alpha > 0 and policy_step_count > 0:
                    if obs_filter_mode == "vel_only":
                        obs[0:6] = obs_filter_alpha * prev_obs[0:6] + (1.0 - obs_filter_alpha) * obs[0:6]
                        obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1.0 - obs_filter_alpha) * obs[29:46]
                    else:
                        obs[:] = obs_filter_alpha * prev_obs + (1.0 - obs_filter_alpha) * obs
                prev_obs[:] = obs


                t_now = time.time() - start
                if t_now - last_print_time >= 2.0:
                    last_print_time = t_now
                    pos = d.qpos[0:3]
                    print(f"[t={t_now:5.1f}s] h={pos[2]:.3f}m pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                          f"ncon={d.ncon} act_max={np.max(np.abs(action_isaac16_exec)):.2f}")
                    print(f"  obs: lin_vel={obs[0:3]} ang_vel={obs[3:6]}")
                    print(f"  obs: gravity={obs[6:9]} cmd_obs={obs[9:12]}")
                    print(f"  obs: qj_rel[:4]={obs[12:16]} dqj[:4]={obs[29:33]}")
                    print(f"  base_vel_world={base_lin_vel_world} base_vel_body={base_lin_vel}")


                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_isaac16_raw = policy(obs_tensor).detach().numpy().squeeze()


                if use_tanh_action:
                    action_isaac16_raw = np.tanh(action_isaac16_raw)
                if action_clip is not None:
                    action_isaac16_raw = np.clip(action_isaac16_raw, -action_clip, action_clip)


                action_isaac16_exec = action_isaac16_raw.copy()


                if action_ramp_steps > 0 and policy_step_count < action_ramp_steps:
                    ramp_factor = float(policy_step_count) / float(action_ramp_steps)
                    action_isaac16_exec = action_isaac16_exec * ramp_factor


                if action_filter_alpha > 0:
                    action_isaac16_exec = action_filter_alpha * action_isaac16_prev + (1.0 - action_filter_alpha) * action_isaac16_exec
                action_isaac16_prev[:] = action_isaac16_exec

                policy_step_count += 1


                target_dof_pos[waist_mj_idx] = waist_default

                for i16 in range(num_actions):
                    mj_idx = isaac16_action_to_mj17[i16]
                    target_dof_pos[mj_idx] = action_isaac16_exec[i16] * action_scale + default_angles[mj_idx]


                for i, jname in enumerate(mj_joint_names):
                    if jname in joint_limits:
                        low, high = joint_limits[jname]
                        target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)


            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if kb_controller:
        kb_controller.stop()
    print("Simulation completed.")