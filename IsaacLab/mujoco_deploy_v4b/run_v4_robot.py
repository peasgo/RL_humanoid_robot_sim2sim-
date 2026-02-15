"""
V4 Quadruped Sim2Sim MuJoCo Deployment - DOG_V5 (12-action policy)
Adapted for the 2026-02-15 trained policy with 13 joints / 12 actions / 47 obs.
"""
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


# V4 coordinate remap functions (Isaac body frame -> standard frame)
# V4 URDF: forward=body_z, lateral=body_x, vertical=body_y
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
    parser = argparse.ArgumentParser(description="V4 Quadruped Sim2Sim - DOG_V5 (12-action)")
    parser.add_argument("config_file", type=str, help="config file name (e.g., v4_robot.yaml)")
    parser.add_argument("--no-policy", action="store_true", help="Disable policy, only PD hold default stance")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable keyboard control")
    args = parser.parse_args()

    # Load config
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
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        v4_coordinate_remap = config.get("v4_coordinate_remap", False)
        mass_scale = float(config.get("mass_scale", 1.0))

    # Load MuJoCo model
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Mass scaling
    if mass_scale != 1.0:
        original_mass = sum(m.body_mass[bi] for bi in range(m.nbody))
        for bi in range(m.nbody):
            m.body_mass[bi] *= mass_scale
            m.body_inertia[bi] *= mass_scale
        scaled_mass = sum(m.body_mass[bi] for bi in range(m.nbody))
        print(f"Mass scaling: {mass_scale}x  ({original_mass:.3f}kg -> {scaled_mass:.3f}kg)")

    # Get MuJoCo joint names (non-free joints)
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    num_mj_joints = len(mj_joint_names)
    print(f"MuJoCo joint order ({num_mj_joints} joints): {mj_joint_names}")

    # Actuator -> joint index mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # ============================================================
    # Isaac joint ordering (from training env)
    # ============================================================
    # Isaac 13-joint order (all joints including Waist_2, for observations):
    # This is the PhysX articulation DOF order (BFS-like traversal).
    # Verified by running scripts/print_joint_order.py --headless
    isaac13_joint_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'Waist_2',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP',
        'LSDy', 'RSDy', 'LARMp', 'RARMp',
    ]

    # Isaac 12-action order: find_joints() uses preserve_order=False by default,
    # so actions are returned in PhysX DOF order (not ActionsCfg list order).
    # This is isaac13_joint_order minus Waist_2.
    isaac12_action_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP',
        'LSDy', 'RSDy', 'LARMp', 'RARMp',
    ]

    print(f"Isaac13 joint order ({len(isaac13_joint_order)} joints): {isaac13_joint_order}")
    print(f"Isaac12 action order ({len(isaac12_action_order)} joints): {isaac12_action_order}")

    # Validate all joints exist in MuJoCo
    for jname in isaac13_joint_order:
        if jname not in mj_joint_names:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model. Available: {mj_joint_names}")

    # Mapping: Isaac13 index -> MuJoCo joint index (for observations)
    isaac13_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac13_joint_order],
        dtype=np.int32
    )

    print(f"\nMapping - Isaac13 <-> MuJoCo (for obs):")
    for i13, jname in enumerate(isaac13_joint_order):
        mj_idx = isaac13_to_mujoco[i13]
        print(f"  Isaac13[{i13:2d}] {jname:12s} <-> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]}")

    # Mapping: Isaac12 action index -> MuJoCo joint index (for actions)
    isaac12_action_to_mj = np.array(
        [mj_joint_names.index(jname) for jname in isaac12_action_order],
        dtype=np.int32
    )

    print(f"\nMapping - Isaac12 <-> MuJoCo (for action):")
    for i12, jname in enumerate(isaac12_action_order):
        mj_idx = isaac12_action_to_mj[i12]
        print(f"  Isaac12[{i12:2d}] {jname:12s} -> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]}")

    # Waist_2 locked at default
    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]
    print(f"\nWaist_2: MJ idx={waist_mj_idx}, locked at {waist_default:.4f} rad")

    # dq sign fixes
    dq_sign = np.ones(len(isaac13_joint_order), dtype=np.float32)
    if isinstance(dq_sign_fixes_cfg, dict):
        for jn, s in dq_sign_fixes_cfg.items():
            if jn in isaac13_joint_order:
                dq_sign[isaac13_joint_order.index(jn)] = float(s)

    # Joint limits
    joint_limits = {}
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if m.jnt_limited[jid]:
            joint_limits[jname] = (float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1]))

    # ============================================================
    # Load policy
    # ============================================================
    policy = None
    if args.no_policy:
        print("\n--no-policy flag set. Running in STANCE MODE (PD hold only, no policy).")
    elif os.path.exists(policy_path) and os.path.isfile(policy_path):
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        print("Policy loaded successfully!")
    else:
        print(f"\nWARNING: No policy found at {policy_path}. Running in DEBUG MODE.")

    # ============================================================
    # Initialize state
    # ============================================================
    action_raw = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Set initial pose
    init_height = float(config.get("init_height", 0.27))
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles

    warmup_steps = int(5.0 / simulation_dt)

    print(f"\n{'='*60}")
    print(f"V4 Quadruped Sim2Sim - DOG_V5 (12-action policy):")
    print(f"  num_actions: {num_actions}")
    print(f"  num_obs: {num_obs}")
    print(f"  action_scale: {action_scale}")
    print(f"  ang_vel_scale: {ang_vel_scale}")
    print(f"  dof_vel_scale: {dof_vel_scale}")
    print(f"  mass_scale: {mass_scale}")
    print(f"  control_decimation: {control_decimation}")
    print(f"  simulation_dt: {simulation_dt}")
    print(f"  v4_coordinate_remap: {v4_coordinate_remap}")
    print(f"  cmd_init: {cmd}")
    print(f"{'='*60}\n")

    # Keyboard controller
    kb_controller = None
    if not args.no_keyboard:
        kb_controller = KeyboardController(cmd)
        print("  Keyboard: W/S=fwd/back, A/D=left/right, Q/E=turn, SPACE=stop, R=reset")

    print("\nStarting simulation...")

    with mujoco.viewer.launch_passive(m, d) as viewer:

        # Warmup
        print(f"Warmup: settling for {warmup_steps * simulation_dt:.1f}s...")
        for ws in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if ws % 100 == 0:
                viewer.sync()
        viewer.sync()
        print(f"Warmup done. Height: {d.qpos[2]:.4f}m")

        # Reset velocities after warmup
        counter = 0
        d.qvel[:] = 0

        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))
        last_print_time = 0.0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Handle reset
            if kb_controller and kb_controller.reset_requested:
                kb_controller.reset_requested = False
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action_raw[:] = 0
                counter = 0
                start = time.time()
                print("\n[RESET] Robot pose reset!")
                continue

            # Apply PD control
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]

            # Step simulation
            mujoco.mj_step(m, d)
            counter += 1

            # Render
            if counter % render_interval == 0:
                viewer.sync()

            # Policy step
            if counter % control_decimation == 0 and policy is not None:
                quat = d.qpos[3:7]

                # MuJoCo free joint qvel[3:6] is already in body (local) frame
                omega = d.qvel[3:6].copy()

                # Joint positions and velocities (MuJoCo order)
                qj_mujoco = d.qpos[7:].copy()
                dqj_mujoco = d.qvel[6:].copy()

                # Remap to Isaac13 order
                qj_isaac13 = qj_mujoco[isaac13_to_mujoco]
                dqj_isaac13 = dqj_mujoco[isaac13_to_mujoco] * dq_sign

                # Default angles in Isaac13 order
                default_angles_isaac13 = default_angles[isaac13_to_mujoco]

                # Gravity orientation
                gravity_orientation = get_gravity_orientation(quat)

                # V4 coordinate remap
                if v4_coordinate_remap:
                    omega_obs = v4_remap_ang_vel(omega)
                    gravity_obs = v4_remap_gravity(gravity_orientation)
                else:
                    omega_obs = omega
                    gravity_obs = gravity_orientation

                # Scale observations
                omega_obs = omega_obs * ang_vel_scale
                qj = (qj_isaac13 - default_angles_isaac13) * dof_pos_scale
                dqj = dqj_isaac13 * dof_vel_scale

                # Build observation vector (47 dims):
                # [0:3]   ang_vel (3)
                # [3:6]   projected_gravity (3)
                # [6:9]   velocity_commands (3)
                # [9:22]  joint_pos_rel (13)
                # [22:35] joint_vel_rel (13)
                # [35:47] last_action (12)
                obs[0:3] = omega_obs
                obs[3:6] = gravity_obs
                obs[6:9] = cmd * cmd_scale
                obs[9:22] = qj
                obs[22:35] = dqj
                obs[35:47] = action_raw.astype(np.float32)

                # Debug print
                t_now = time.time() - start
                if t_now - last_print_time >= 2.0:
                    last_print_time = t_now
                    pos = d.qpos[0:3]
                    print(f"[t={t_now:5.1f}s] h={pos[2]:.3f}m pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                          f"ncon={d.ncon} act_max={np.max(np.abs(action_raw)):.2f}")
                    print(f"  obs: ang_vel={obs[0:3]} gravity={obs[3:6]} cmd_obs={obs[6:9]}")
                    print(f"  obs: qj_rel[:4]={obs[9:13]} dqj[:4]={obs[22:26]}")

                # Run policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_raw = policy(obs_tensor).detach().numpy().squeeze()

                # Apply actions: lock Waist_2, set 12 action joints
                target_dof_pos[waist_mj_idx] = waist_default

                for i12 in range(num_actions):
                    mj_idx = isaac12_action_to_mj[i12]
                    target_dof_pos[mj_idx] = action_raw[i12] * action_scale + default_angles[mj_idx]

                # Enforce joint limits
                for i, jname in enumerate(mj_joint_names):
                    if jname in joint_limits:
                        low, high = joint_limits[jname]
                        target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)

            # Real-time sync
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if kb_controller:
        kb_controller.stop()
    print("Simulation completed.")
