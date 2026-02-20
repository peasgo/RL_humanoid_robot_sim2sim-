
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ObsTermHistory:
    """Per-term circular buffer mimicking IsaacLab's CircularBuffer behavior.

    On first append after reset, all slots are filled with the same value.
    Buffer order: oldest → newest (matching IsaacLab convention).
    """

    def __init__(self, max_len: int, term_dim: int):
        self.max_len = max_len
        self.term_dim = term_dim
        self._buf = np.zeros((max_len, term_dim), dtype=np.float32)
        self._num_pushes = 0

    def reset(self):
        self._buf[:] = 0.0
        self._num_pushes = 0

    def append(self, data: np.ndarray):
        """Append data (shape: (term_dim,)). First append fills all slots."""
        if self._num_pushes == 0:
            # Fill entire buffer with first observation (IsaacLab behavior)
            for i in range(self.max_len):
                self._buf[i] = data
        else:
            # Shift left (drop oldest), append newest at end
            self._buf[:-1] = self._buf[1:]
            self._buf[-1] = data
        self._num_pushes += 1

    def flatten(self) -> np.ndarray:
        """Return flattened buffer: [oldest_0..oldest_d, ..., newest_0..newest_d]."""
        return self._buf.flatten()


def get_gravity_orientation(quaternion):

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
    parser = argparse.ArgumentParser(description="V6 Humanoid Sim2Sim - Biped (13-action)")
    parser.add_argument("config_file", type=str, help="config file name (e.g., v6_robot.yaml)")
    parser.add_argument("--no-policy", action="store_true", help="Disable policy, only PD hold default stance")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable keyboard control")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup, start with joints at 0 (match IsaacLab reset behavior)")
    parser.add_argument("--debug", action="store_true",
                        help="Print full observation vector for first N policy steps (for comparison with dump_v6_isaac_obs.py)")
    parser.add_argument("--debug-steps", type=int, default=10,
                        help="Number of policy steps to print in debug mode (default: 10)")
    args = parser.parse_args()

    # ================================================================
    # Load config
    # ================================================================
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
        # IsaacLab: sim.dt = 0.005
        simulation_dt = config["simulation_dt"]
        # IsaacLab: decimation = 4  (policy runs every 4 * 0.005 = 0.02s)
        control_decimation = config["control_decimation"]

        # PD gains (informational only - actual gains are in XML position actuators)
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        # Default joint angles in MuJoCo joint order
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        print(f"[DIAG] default_angles after load: {default_angles}")

        # IsaacLab observation scales
        ang_vel_scale = config["ang_vel_scale"]      # 0.2  (base_ang_vel ObsTerm scale)
        dof_pos_scale = config["dof_pos_scale"]      # 1.0  (joint_pos_rel, no extra scale)
        dof_vel_scale = config["dof_vel_scale"]      # 0.05 (joint_vel_rel ObsTerm scale)
        # IsaacLab: JointPositionActionCfg(scale=0.25, use_default_offset=True)
        action_scale = config["action_scale"]         # 0.25
        # velocity_commands has no extra scale in IsaacLab
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)  # [1,1,1]

        num_actions = config["num_actions"]  # 13
        num_obs = config["num_obs"]          # 48
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # ================================================================
    # Load MuJoCo model
    # ================================================================
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # ================================================================
    # Get MuJoCo joint names (non-free joints, XML DFS order)
    # MuJoCo order: pelvis_link, RHIPp, RHIPy, RHIPr, RKNEEp, RANKLEp,
    #               RANKLEy, LHIPp, LHIPy, LHIPr, LKNEEp, LANKLEp, LANKLEy
    # ================================================================
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    num_mj_joints = len(mj_joint_names)
    print(f"MuJoCo joint order ({num_mj_joints} joints): {mj_joint_names}")

    # Actuator -> joint index mapping (for d.ctrl)
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # ================================================================
    # Isaac joint ordering (PhysX BFS order, L-before-R)
    #
    # This is the order used by the trained policy for both observations
    # and actions. Confirmed by running print_v6_joint_order.py.
    #
    # PhysX BFS order:
    #   [0]  pelvis_link  default=0.0
    #   [1]  LHIPp        default=-0.2
    #   [2]  RHIPp        default=-0.2
    #   [3]  LHIPy        default=0.0
    #   [4]  RHIPy        default=0.0
    #   [5]  LHIPr        default=0.0
    #   [6]  RHIPr        default=0.0
    #   [7]  LKNEEp       default=-0.4
    #   [8]  RKNEEp       default=0.4
    #   [9]  LANKLEp      default=0.2
    #   [10] RANKLEp      default=-0.2
    #   [11] LANKLEy      default=0.0
    #   [12] RANKLEy      default=0.0
    # ================================================================
    # ================================================================
    # Observation joint order: PhysX BFS (used by joint_pos_rel, joint_vel_rel)
    # ================================================================
    isaac_joint_order_from_yaml = config.get("isaac_joint_order", None)
    if isaac_joint_order_from_yaml is not None:
        isaac_joint_order = isaac_joint_order_from_yaml
        print(f"Using isaac_joint_order (obs) from YAML config.")
    else:
        isaac_joint_order = [
            'pelvis_link',
            'LHIPp', 'RHIPp',
            'LHIPy', 'RHIPy',
            'LHIPr', 'RHIPr',
            'LKNEEp', 'RKNEEp',
            'LANKLEp', 'RANKLEp',
            'LANKLEy', 'RANKLEy',
        ]
        print(f"Using default isaac_joint_order (L-before-R BFS).")

    print(f"Obs joint order ({len(isaac_joint_order)} joints): {isaac_joint_order}")

    # ================================================================
    # Action joint order: ActionsCfg explicit list (preserve_order=False → query order)
    # In flat_env_cfg.py, joint_names are listed R-before-L, which matches MuJoCo DFS.
    # ================================================================
    action_joint_order_from_yaml = config.get("action_joint_order", None)
    if action_joint_order_from_yaml is not None:
        action_joint_order = action_joint_order_from_yaml
        print(f"Using action_joint_order from YAML config.")
    else:
        # Default: same as ActionsCfg in flat_env_cfg.py = MuJoCo DFS order
        action_joint_order = [
            'pelvis_link',
            'RHIPp', 'RHIPy', 'RHIPr', 'RKNEEp', 'RANKLEp', 'RANKLEy',
            'LHIPp', 'LHIPy', 'LHIPr', 'LKNEEp', 'LANKLEp', 'LANKLEy',
        ]
        print(f"Using default action_joint_order (ActionsCfg / MuJoCo DFS).")

    print(f"Action joint order ({len(action_joint_order)} joints): {action_joint_order}")

    # Validate all joints exist in MuJoCo
    for jname in isaac_joint_order + action_joint_order:
        if jname not in mj_joint_names:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model. Available: {mj_joint_names}")

    # Mapping: Isaac obs index -> MuJoCo joint index
    # Used to reorder MuJoCo joint data into PhysX BFS order for observations.
    isaac_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac_joint_order],
        dtype=np.int32
    )

    # Mapping: Action index -> MuJoCo joint index
    # Used to map policy action output to MuJoCo joints.
    action_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in action_joint_order],
        dtype=np.int32
    )

    # Precompute default angles in both orders
    # IMPORTANT: use .copy() to prevent any later modification from affecting these
    print(f"[DIAG] default_angles before isaac remap: {default_angles}")
    default_angles_isaac = default_angles[isaac_to_mujoco].copy()  # for obs
    default_angles_action = default_angles[action_to_mujoco].copy()  # for actions
    print(f"[DIAG] default_angles_isaac: {default_angles_isaac}")
    # Freeze default_angles to catch any accidental modification
    default_angles.flags.writeable = False

    print(f"\nMapping - Isaac <-> MuJoCo:")
    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        print(f"  Isaac[{i_isaac:2d}] {jname:14s} (def={default_angles_isaac[i_isaac]:+.1f})"
              f" <-> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]} (def={default_angles[mj_idx]:+.1f})")

    # ================================================================
    # Load policy
    # ================================================================
    policy = None
    if args.no_policy:
        print("\n--no-policy flag set. Running in STANCE MODE (PD hold only, no policy).")
    elif os.path.exists(policy_path) and os.path.isfile(policy_path):
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        print("Policy loaded successfully!")
    else:
        print(f"\nWARNING: No policy found at {policy_path}. Running in DEBUG MODE.")

    # ================================================================
    # Initialize state
    # ================================================================
    action_raw = np.zeros(num_actions, dtype=np.float32)  # last policy output (Isaac order)
    target_dof_pos = np.array(default_angles, dtype=np.float32)  # PD targets (MuJoCo order), explicit copy

    # Observation history: per-term buffers (IsaacLab convention)
    obs_history_len = int(config.get("obs_history_length", 1))
    obs_single_dim = int(config.get("obs_single_frame_dim", num_obs))
    # Term dimensions: ang_vel(3), gravity(3), cmd(3), jpos(13), jvel(13), act(13) = 48
    term_dims = [3, 3, 3, num_actions, num_actions, num_actions]
    term_names = ["ang_vel", "gravity", "cmd", "jpos_rel", "jvel", "last_act"]
    assert sum(term_dims) == obs_single_dim, \
        f"Term dims {sum(term_dims)} != obs_single_frame_dim {obs_single_dim}"
    assert obs_single_dim * obs_history_len == num_obs, \
        f"obs_single_dim({obs_single_dim}) * history({obs_history_len}) != num_obs({num_obs})"

    # Create per-term history buffers
    obs_term_histories = [ObsTermHistory(obs_history_len, d) for d in term_dims]
    obs = np.zeros(num_obs, dtype=np.float32)  # full flattened obs (48 dims)
    obs_frame = np.zeros(obs_single_dim, dtype=np.float32)  # single frame (48 dims)
    counter = 0
    policy_step_count = 0

    # COM trajectory recording
    com_trajectory = []  # list of (time, x, y, z)

    # rsl_rl clips observations and actions (defaults: 100.0)
    clip_obs = float(config.get("clip_obs", 100.0))
    clip_actions = float(config.get("clip_actions", 100.0))

    # Initial pose — matches IsaacLab InitialStateCfg:
    #   pos=(0.0, 0.0, 0.55), rot=(0.7071068, 0.0, 0.0, 0.7071068)
    init_height = float(config.get("init_height", 0.55))
    # MuJoCo free joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z)
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]

    if args.no_warmup:
        # Match IsaacLab reset: joints start at default_angles (InitialStateCfg.joint_pos)
        # IsaacLab sets joint_pos = default_joint_pos at reset, so obs joint_pos_rel = 0
        d.qpos[7:] = default_angles
        target_dof_pos[:] = default_angles
        print(f"[no-warmup] Joints initialized to default_angles (matching IsaacLab reset)")
    else:
        # Set joint angles to defaults (MuJoCo order)
        d.qpos[7:] = default_angles

    # Warmup: let the robot settle under gravity with PD holding default pose
    warmup_steps = 0 if args.no_warmup else int(5.0 / simulation_dt)

    print(f"\n{'='*60}")
    print(f"V6 Humanoid Sim2Sim - Biped (13-action policy):")
    print(f"  num_actions: {num_actions}")
    print(f"  num_obs: {num_obs} ({obs_single_dim} x {obs_history_len} history)")
    print(f"  action_scale: {action_scale}")
    print(f"  ang_vel_scale: {ang_vel_scale}")
    print(f"  dof_pos_scale: {dof_pos_scale}")
    print(f"  dof_vel_scale: {dof_vel_scale}")
    print(f"  control_decimation: {control_decimation}")
    print(f"  simulation_dt: {simulation_dt}")
    print(f"  init_height: {init_height}")
    print(f"  cmd_init: {cmd}")
    print(f"  cmd_scale: {cmd_scale}")
    print(f"  clip_obs: {clip_obs}")
    print(f"  clip_actions: {clip_actions}")
    if args.debug:
        print(f"  DEBUG MODE: printing first {args.debug_steps} policy steps")
    print(f"{'='*60}\n")

    # Keyboard controller
    kb_controller = None
    if not args.no_keyboard:
        kb_controller = KeyboardController(cmd)
        print("  Keyboard: W/S=fwd/back, A/D=left/right, Q/E=turn, SPACE=stop, R=reset")

    print("\nStarting simulation...")

    with mujoco.viewer.launch_passive(m, d) as viewer:

        # ============================================================
        # Warmup phase: settle robot under gravity
        # ============================================================
        print(f"Warmup: settling for {warmup_steps * simulation_dt:.1f}s...")
        for ws in range(warmup_steps):
            # Hold default pose via position actuators
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if ws % 100 == 0:
                viewer.sync()
        viewer.sync()

        # --- Detailed warmup end state ---
        warmup_quat = d.qpos[3:7].copy()
        warmup_gravity = get_gravity_orientation(warmup_quat)
        warmup_qj = d.qpos[7:].copy()
        warmup_qj_isaac = warmup_qj[isaac_to_mujoco]
        warmup_qj_rel = warmup_qj_isaac - default_angles_isaac
        print(f"\n--- Warmup End State ---")
        print(f"  height:    {d.qpos[2]:.4f}m  (init: {init_height})")
        print(f"  quat:      {warmup_quat}  (w,x,y,z)")
        print(f"  gravity_b: {warmup_gravity}  (expect ~[0,0,-1])")
        print(f"  ncon:      {d.ncon}")
        print(f"  qvel_max:  {np.max(np.abs(d.qvel)):.4f}")
        print(f"  Joint positions after warmup (Isaac order):")
        for i_isaac, jname in enumerate(isaac_joint_order):
            mj_idx = isaac_to_mujoco[i_isaac]
            print(f"    [{i_isaac:2d}] {jname:14s}  pos={warmup_qj_isaac[i_isaac]:+.4f}"
                  f"  def={default_angles_isaac[i_isaac]:+.4f}"
                  f"  rel={warmup_qj_rel[i_isaac]:+.4f}")
        print(f"  ctrl (actuator targets): {np.array2string(d.ctrl, precision=4, separator=', ')}")
        print(f"--- End Warmup State ---\n")

        # After warmup, reset velocities to zero (clean start for policy)
        # but keep the settled position (don't reset qpos — the robot
        # has found its equilibrium under gravity).
        counter = 0
        sim_time = 0.0  # track simulation time accurately
        d.qvel[:] = 0

        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))
        last_print_time = 0.0

        # ================================================================
        # Main simulation loop
        # IsaacLab timing: obs → policy → step(decimation) → obs → ...
        # We use a sub-step counter within each decimation period.
        # The policy runs BEFORE physics steps (matching Gym semantics).
        # ================================================================
        substep = 0  # counts sub-steps within current decimation period

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Handle reset request
            if kb_controller and kb_controller.reset_requested:
                kb_controller.reset_requested = False
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action_raw[:] = 0
                # Reset observation history buffers
                for hist_buf in obs_term_histories:
                    hist_buf.reset()
                counter = 0
                substep = 0
                policy_step_count = 0
                start = time.time()
                print("\n[RESET] Robot pose reset!")
                continue

            # ========================================================
            # Policy step BEFORE physics (matching IsaacLab Gym semantics)
            # IsaacLab: reset→obs0→policy(obs0)→a0→step(4)→obs1→policy(obs1)→...
            # So policy runs at substep=0 of each decimation period.
            # ========================================================
            if substep == 0 and policy is not None:
                # --- Read MuJoCo state ---
                quat = d.qpos[3:7].copy()  # quaternion [w,x,y,z]

                # MuJoCo freejoint qvel[3:6] is angular velocity in body frame
                omega_body = d.qvel[3:6].copy()

                # Joint positions and velocities (MuJoCo order)
                qj_mujoco = d.qpos[7:].copy()    # 13 joint positions
                dqj_mujoco = d.qvel[6:].copy()    # 13 joint velocities

                # --- Remap to Isaac order ---
                qj_isaac = qj_mujoco[isaac_to_mujoco]
                dqj_isaac = dqj_mujoco[isaac_to_mujoco]

                # --- Compute observation terms ---

                # obs[0:3]: base_ang_vel * ang_vel_scale
                # IsaacLab: ObsTerm(func=mdp.base_ang_vel, scale=0.2)
                omega_obs = omega_body * ang_vel_scale

                # obs[3:6]: projected_gravity_b
                # IsaacLab: ObsTerm(func=mdp.projected_gravity)
                gravity_obs = get_gravity_orientation(quat)

                # obs[6:9]: velocity_commands [vx, vy, wz]
                # IsaacLab: ObsTerm(func=mdp.generated_commands)
                # No extra scale (cmd_scale = [1,1,1])
                cmd_obs = cmd * cmd_scale

                # obs[9:22]: joint_pos_rel = (joint_pos - default_joint_pos) * dof_pos_scale
                # IsaacLab: ObsTerm(func=mdp.joint_pos_rel)  scale=1.0 (default)
                qj_rel = (qj_isaac - default_angles_isaac) * dof_pos_scale

                # obs[22:35]: joint_vel * dof_vel_scale
                # IsaacLab: ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
                dqj_obs = dqj_isaac * dof_vel_scale

                # obs[35:48]: last_action (raw policy output, NOT scaled)
                # IsaacLab: ObsTerm(func=mdp.last_action)
                last_act = action_raw.copy()

                # --- Assemble current frame terms and append to history ---
                current_terms = [omega_obs, gravity_obs, cmd_obs, qj_rel, dqj_obs, last_act]
                for hist_buf, term_data in zip(obs_term_histories, current_terms):
                    hist_buf.append(term_data)

                # --- Build full observation: single frame (no history) ---
                # Layout: [ang_vel(3), gravity(3), cmd(3),
                #          jpos(13), jvel(13), act(13)] = 48
                offset = 0
                for hist_buf in obs_term_histories:
                    flat = hist_buf.flatten()
                    obs[offset:offset + len(flat)] = flat
                    offset += len(flat)

                # --- Clip observations (rsl_rl does this) ---
                obs = np.clip(obs, -clip_obs, clip_obs)

                # --- Debug output ---
                t_wall = time.time() - start
                sim_time = counter * simulation_dt  # current sim time (before stepping)

                # === Step 0/1 detailed output (matches IsaacLab format) ===
                if policy_step_count <= 1:
                    print(f"\n{'='*60}")
                    print(f"  STEP {policy_step_count}")
                    print(f"{'='*60}")
                    print(f"\n  [Observation] shape=({num_obs},)")
                    print(f"  obs = {np.array2string(obs, precision=4, separator=', ', max_line_width=120)}")
                    print(f"\n  [DEBUG] default_angles_isaac = {np.array2string(default_angles_isaac, precision=4, separator=', ')}")
                    print(f"  [DEBUG] qj_isaac             = {np.array2string(qj_isaac, precision=6, separator=', ')}")
                    print(f"  [DEBUG] qj_rel               = {np.array2string(qj_rel, precision=6, separator=', ')}")
                    print(f"\n  [Joint Positions (rad)] (Isaac BFS order)")
                    for i_isaac, jname in enumerate(isaac_joint_order):
                        mj_idx = isaac_to_mujoco[i_isaac]
                        print(f"    [{i_isaac:2d}] {jname:30s}  pos={qj_isaac[i_isaac]:+.6f}  def={default_angles_isaac[i_isaac]:+.4f}  rel={qj_rel[i_isaac]:+.6f}  vel={dqj_isaac[i_isaac]:+.6f}")
                    print()

                if args.debug and policy_step_count < args.debug_steps:
                    print(f"\n{'='*70}")
                    print(f"[DEBUG Policy Step {policy_step_count}] sim_t={sim_time:.4f}s  wall_t={t_wall:.4f}s")
                    print(f"  qpos[0:3] (pos):     {d.qpos[0:3]}")
                    print(f"  qpos[3:7] (quat):    {quat}  (w,x,y,z)")
                    print(f"  qvel[0:3] (lin_vel):  {d.qvel[0:3]}  (world frame)")
                    print(f"  qvel[3:6] (ang_vel):  {omega_body}  (body frame, direct)")
                    print(f"  height:               {d.qpos[2]:.4f}m")
                    print(f"  ncon:                 {d.ncon}")
                    print(f"  ---")
                    print(f"  Current frame (48 dims):")
                    print(f"    ang_vel*{ang_vel_scale}:   {np.array2string(omega_obs, precision=6, separator=', ')}")
                    print(f"    gravity:       {np.array2string(gravity_obs, precision=6, separator=', ')}")
                    print(f"    cmd:           {np.array2string(cmd_obs, precision=6, separator=', ')}")
                    print(f"    joint_pos_rel: {np.array2string(qj_rel, precision=6, separator=', ')}")
                    print(f"    joint_vel*{dof_vel_scale}: {np.array2string(dqj_obs, precision=6, separator=', ')}")
                    print(f"    last_action:   {np.array2string(last_act, precision=6, separator=', ')}")
                    print(f"  --- Full obs ({num_obs} dims = {obs_single_dim}x{obs_history_len} history) ---")
                    offset_dbg = 0
                    for tname, hist_buf in zip(term_names, obs_term_histories):
                        tlen = hist_buf.max_len * hist_buf.term_dim
                        print(f"    {tname:10s}[{offset_dbg}:{offset_dbg+tlen}]: "
                              f"{np.array2string(obs[offset_dbg:offset_dbg+tlen], precision=4, separator=', ', max_line_width=120)}")
                        offset_dbg += tlen
                    print(f"  --- Action (previous) ---")
                    print(f"  action_raw: {np.array2string(action_raw, precision=4, separator=', ')}")
                    print(f"  target_dof_pos (MJ): {np.array2string(target_dof_pos, precision=4, separator=', ')}")
                    print(f"{'='*70}")
                elif not args.debug and policy_step_count < 3:
                    # Brief diagnostics for first 3 steps even without --debug
                    print(f"\n[Policy Step {policy_step_count}] sim_t={sim_time:.4f}s  wall_t={t_wall:.3f}s")
                    print(f"  ang_vel_body={omega_body}  gravity={gravity_obs}")
                    print(f"  obs shape={obs.shape}  obs[0:3]={obs[0:3]}  obs[3:6]={obs[3:6]}")
                    print(f"  cmd={cmd}")

                # --- Run policy inference ---
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_raw_new = policy(obs_tensor).detach().numpy().squeeze()

                # --- Clip actions (rsl_rl does this) ---
                action_raw = np.clip(action_raw_new, -clip_actions, clip_actions)

                if args.debug and policy_step_count < args.debug_steps:
                    print(f"  >>> NEW action_raw: {np.array2string(action_raw, precision=4, separator=', ')}")

                # === Print actions sent (matches IsaacLab format) ===
                if policy_step_count == 0:
                    print(f"\n  [Actions sent at step 0] (Isaac BFS order)")
                    for i_isaac, jname in enumerate(isaac_joint_order):
                        print(f"    [{i_isaac:2d}] {jname:30s}  action={action_raw[i_isaac]:+.6f}")

                policy_step_count += 1

                # --- Apply actions ---
                # IsaacLab: target = action[i] * scale + default_joint_pos[i]
                # action[i] is in PhysX BFS order (preserve_order=False → target order).
                # NOTE: resolve_matching_names with preserve_order=False returns TARGET
                # list order (PhysX BFS), despite misleading docstring text.
                # Confirmed by code: outer loop iterates list_of_strings (target).
                for i_isaac in range(num_actions):
                    mj_idx = isaac_to_mujoco[i_isaac]
                    # action_scale=0.25, use_default_offset=True
                    target_dof_pos[mj_idx] = (
                        action_raw[i_isaac] * action_scale + default_angles[mj_idx]
                    )

                if args.debug and policy_step_count <= args.debug_steps:
                    print(f"  >>> NEW target_dof_pos (MJ): {np.array2string(target_dof_pos, precision=4, separator=', ')}")
                    print(f"  >>> Target per joint (PhysX BFS order):")
                    for i_isaac, jname in enumerate(isaac_joint_order):
                        mj_idx = isaac_to_mujoco[i_isaac]
                        print(f"      [{i_isaac:2d}] {jname:14s}  act={action_raw[i_isaac]:+.4f}"
                              f"  target={target_dof_pos[mj_idx]:+.4f}"
                              f"  (def={default_angles[mj_idx]:+.4f} + {action_raw[i_isaac]:+.4f}*{action_scale})")

            # ========================================================
            # Apply PD control via position actuators
            # MuJoCo position actuator: torque = kp*(ctrl - q) - kv*dq
            # This matches IsaacLab ImplicitActuatorCfg behavior.
            # ========================================================
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]

            # Step simulation
            mujoco.mj_step(m, d)
            counter += 1
            substep += 1

            # --- NaN safety: auto-reset if simulation diverges ---
            if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
                print(f"\n[NaN DETECTED at sim_t={counter * simulation_dt:.3f}s] Auto-resetting...")
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action_raw[:] = 0
                for hist_buf in obs_term_histories:
                    hist_buf.reset()
                substep = 0
                policy_step_count = 0
                mujoco.mj_forward(m, d)
                continue

            # Record COM position (subtree_com[0] = whole-body COM, updated by mj_step)
            com = d.subtree_com[0].copy()  # [x, y, z]
            com_trajectory.append((counter * simulation_dt, com[0], com[1], com[2]))

            # Reset substep counter at end of decimation period
            if substep >= control_decimation:
                substep = 0

            # Track simulation time
            sim_time = counter * simulation_dt

            # Periodic COM print (works with or without policy)
            t_wall = time.time() - start
            if t_wall - last_print_time >= 2.0:
                last_print_time = t_wall
                pos = d.qpos[0:3]
                com_pos = d.subtree_com[0]
                print(f"[sim_t={sim_time:5.1f}s wall_t={t_wall:5.1f}s] h={pos[2]:.3f}m pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
                      f"COM=({com_pos[0]:+.3f},{com_pos[1]:+.3f},{com_pos[2]:.3f}) "
                      f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                      f"ncon={d.ncon} act_max={np.max(np.abs(action_raw)):.2f}")

            # Render
            if counter % render_interval == 0:
                viewer.sync()

            # Real-time sync
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if kb_controller:
        kb_controller.stop()
    print("Simulation completed.")

    # ================================================================
    # Plot COM trajectory
    # ================================================================
    if len(com_trajectory) > 0:
        com_data = np.array(com_trajectory)  # (N, 4): time, x, y, z
        t = com_data[:, 0]
        cx, cy, cz = com_data[:, 1], com_data[:, 2], com_data[:, 3]

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('V6 Humanoid - COM Trajectory', fontsize=14)

        # 1) XY trajectory (top-down view)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(cx, cy, 'b-', linewidth=0.5, alpha=0.7)
        ax1.plot(cx[0], cy[0], 'go', markersize=8, label='Start')
        ax1.plot(cx[-1], cy[-1], 'r^', markersize=8, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('COM XY Trajectory (Top View)')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Z height over time
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t, cz, 'r-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('COM Height over Time')
        ax2.grid(True, alpha=0.3)

        # 3) X, Y over time
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t, cx, 'b-', linewidth=0.5, label='X')
        ax3.plot(t, cy, 'g-', linewidth=0.5, label='Y')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('COM X/Y over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4) 3D trajectory
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot(cx, cy, cz, 'b-', linewidth=0.5, alpha=0.7)
        ax4.scatter(cx[0], cy[0], cz[0], c='g', s=50, marker='o', label='Start')
        ax4.scatter(cx[-1], cy[-1], cz[-1], c='r', s=50, marker='^', label='End')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('COM 3D Trajectory')
        ax4.legend()

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(current_dir, 'com_trajectory.png')
        plt.savefig(save_path, dpi=150)
        print(f"COM trajectory plot saved to: {save_path}")
        plt.show()
