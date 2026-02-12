
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

try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ImportError:
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    """V4坐标系重映射：线速度 [+Z, X, Y] — 与训练代码 v4_base_lin_vel 一致"""
    return np.array([lin_vel_body[2], lin_vel_body[0], lin_vel_body[1]])


def v4_remap_ang_vel(ang_vel_body):
    """V4坐标系重映射：角速度 [X, +Z, Y] — 与训练代码 v4_base_ang_vel 一致"""
    return np.array([ang_vel_body[0], ang_vel_body[2], ang_vel_body[1]])


def v4_remap_gravity(gravity_body):
    """V4坐标系重映射：重力投影 [+Z, X, Y] — 与训练代码 v4_projected_gravity 一致"""
    return np.array([gravity_body[2], gravity_body[0], gravity_body[1]])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control: tau = kp*(target_q - q) + kd*(target_dq - dq)"""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ============================================================
# Keyboard command controller (non-blocking)
# ============================================================
class KeyboardController:
    """Non-blocking keyboard input for velocity commands.
    
    Controls:
        W/S: forward/backward velocity (lin_vel_x)
        A/D: left/right velocity (lin_vel_y)
        Q/E: turn left/right (ang_vel_z)
        SPACE: stop (zero all commands)
        R: reset robot pose
    """
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
        """Background thread reading keyboard input."""
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
                    elif ch == '\x03':  # Ctrl+C
                        self._running = False
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # Fallback: no keyboard control available
            pass
    
    def stop(self):
        self._running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Quadruped Sim2Sim MuJoCo Deployment")
    parser.add_argument("config_file", type=str, help="config file name (e.g., v4_robot.yaml)")
    parser.add_argument("--no-policy", action="store_true", help="Disable policy, only PD hold default stance")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable keyboard control")
    args = parser.parse_args()

    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_paths = [
        os.path.join(current_dir, "configs", args.config_file),
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
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # Fix xml_path if needed
        if not os.path.exists(xml_path):
            for candidate in [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), xml_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), xml_path),
                os.path.basename(xml_path)
            ]:
                if os.path.exists(candidate):
                    xml_path = candidate
                    break

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # NOTE: kps, kds, default_angles are all in MuJoCo full joint order (17 joints)
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

        # Action low-pass filter: smoothed = alpha * old + (1-alpha) * new
        action_filter_alpha = float(config.get("action_filter_alpha", 0.0))

        # Observation low-pass filter: obs = alpha * prev_obs + (1-alpha) * new_obs
        # Smooths high-frequency bounce noise from MuJoCo soft contacts
        obs_filter_alpha = float(config.get("obs_filter_alpha", 0.0))
        # "all" = filter entire obs vector, "vel_only" = only filter velocity components
        # vel_only is more natural: position obs stay true, only velocity noise is smoothed
        obs_filter_mode = str(config.get("obs_filter_mode", "all"))

        # Action ramp-up: gradually increase action magnitude over first N policy steps
        action_ramp_steps = int(config.get("action_ramp_steps", 0))

        dq_sign_fixes_cfg = config.get("dq_sign_fixes", None)
        num_actions = config["num_actions"]   # 16
        num_obs = config["num_obs"]           # 62
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        # V4 特有配置
        v4_coordinate_remap = config.get("v4_coordinate_remap", False)

        # Mass scaling
        mass_scale = float(config.get("mass_scale", 1.0))

    # ============================================================
    # Load MuJoCo model
    # ============================================================
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Mass scaling: proportionally increase mass and inertia for all bodies
    if mass_scale != 1.0:
        original_mass = sum(m.body_mass[bi] for bi in range(m.nbody))
        for bi in range(m.nbody):
            m.body_mass[bi] *= mass_scale
            m.body_inertia[bi] *= mass_scale
        scaled_mass = sum(m.body_mass[bi] for bi in range(m.nbody))
        print(f"Mass scaling: {mass_scale}x  ({original_mass:.3f}kg -> {scaled_mass:.3f}kg)")

    # Build MuJoCo joint name list (exclude freejoint) — 17 joints
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    num_mj_joints = len(mj_joint_names)  # 17
    print(f"MuJoCo joint order ({num_mj_joints} joints): {mj_joint_names}")

    # Build actuator -> joint mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # Effort limits (MuJoCo full joint order, 17)
    effort_limit_pd = np.full((num_mj_joints,), np.inf, dtype=np.float32)
    for act_i in range(m.nu):
        j_pd = int(actuator_to_joint_indices[act_i])
        fr = m.actuator_forcerange[act_i]
        effort_limit_pd[j_pd] = float(max(abs(fr[0]), abs(fr[1])))

    # ============================================================
    # MuJoCo joint damping check
    # With damping=0 in XML, PD control kd directly matches IsaacLab's kd
    # No compensation needed (参考开源sim2sim项目的做法)
    # ============================================================
    mj_joint_damping = np.zeros(num_mj_joints, dtype=np.float32)
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof_adr = m.jnt_dofadr[jid]
        mj_joint_damping[i] = float(m.dof_damping[dof_adr])
    print(f"\nMuJoCo joint damping (should be 0): {mj_joint_damping}")
    print(f"  PD kds (= IsaacLab kds): {kds}")
    if np.any(mj_joint_damping > 0):
        print(f"  WARNING: Non-zero MuJoCo damping detected! PD kd will be higher than intended.")

    # Set initial pose
    # Start at 0.22m: 4 foot contacts with ~1.2cm penetration (acceptable with soft solref).
    # Higher values lose contacts; lower values have too much penetration.
    init_height = 0.22
    d.qpos[2] = init_height
    # Ensure correct orientation (X+90° rotation for quadruped mode)
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    # Joint angles (MuJoCo order, 17 joints)
    d.qpos[7:] = default_angles

    # Warmup: let robot settle with PD control before enabling policy
    warmup_steps = int(5.0 / simulation_dt)  # 5.0 seconds of settling

    # ============================================================
    # Isaac 全关节顺序 (17个，已通过调试确认)
    # 用于 obs 中的 joint_pos_rel 和 joint_vel
    # ============================================================
    isaac17_joint_order = [
        'LHIPp',    # 0  - 左髋俯仰（后左腿）
        'RHIPp',    # 1  - 右髋俯仰（后右腿）
        'LHIPy',    # 2  - 左髋偏航（后左腿）
        'RHIPy',    # 3  - 右髋偏航（后右腿）
        'Waist_2',  # 4  - 腰部旋转（锁定，不受策略控制）
        'LSDp',     # 5  - 左肩俯仰（前左腿）
        'RSDp',     # 6  - 右肩俯仰（前右腿）
        'LKNEEp',   # 7  - 左膝（后左腿）
        'RKNEEP',   # 8  - 右膝（后右腿）
        'LSDy',     # 9  - 左肩偏航（前左腿）
        'RSDy',     # 10 - 右肩偏航（前右腿）
        'LANKLEp',  # 11 - 左踝（后左腿）
        'RANKLEp',  # 12 - 右踝（后右腿）
        'LARMp',    # 13 - 左肘俯仰（前左腿）
        'RARMp',    # 14 - 右肘俯仰（前右腿）
        'LARMAp',   # 15 - 左前臂（前左腿）
        'RARMAP',   # 16 - 右前臂（前右腿）
    ]

    # ============================================================
    # Isaac 动作关节顺序 (16个，排除Waist_2，按Isaac内部索引排序)
    # 用于 obs 中的 last_action 和策略输出
    # ============================================================
    isaac16_action_order = [
        'LHIPp',    # 0  - (Isaac17 idx=0)
        'RHIPp',    # 1  - (Isaac17 idx=1)
        'LHIPy',    # 2  - (Isaac17 idx=2)
        'RHIPy',    # 3  - (Isaac17 idx=3)
        'LSDp',     # 4  - (Isaac17 idx=5)
        'RSDp',     # 5  - (Isaac17 idx=6)
        'LKNEEp',   # 6  - (Isaac17 idx=7)
        'RKNEEP',   # 7  - (Isaac17 idx=8)
        'LSDy',     # 8  - (Isaac17 idx=9)
        'RSDy',     # 9  - (Isaac17 idx=10)
        'LANKLEp',  # 10 - (Isaac17 idx=11)
        'RANKLEp',  # 11 - (Isaac17 idx=12)
        'LARMp',    # 12 - (Isaac17 idx=13)
        'RARMp',    # 13 - (Isaac17 idx=14)
        'LARMAp',   # 14 - (Isaac17 idx=15)
        'RARMAP',   # 15 - (Isaac17 idx=16)
    ]

    print(f"Isaac17 joint order ({len(isaac17_joint_order)} joints): {isaac17_joint_order}")
    print(f"Isaac16 action order ({len(isaac16_action_order)} joints): {isaac16_action_order}")

    # Verify all joints exist in MuJoCo model
    for jname in isaac17_joint_order:
        if jname not in mj_joint_names:
            raise ValueError(f"Joint '{jname}' from Isaac order not found in MuJoCo model. "
                           f"Available: {mj_joint_names}")

    # ============================================================
    # Build mapping 1: isaac17 <-> mujoco17 (全17关节，用于obs)
    # ============================================================
    # isaac17_to_mujoco17[isaac17_idx] = mj_idx
    # 用途: qj_mujoco[isaac17_to_mujoco17] → Isaac17 顺序的 qj
    isaac17_to_mujoco17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac17_joint_order],
        dtype=np.int32
    )

    print(f"\nMapping 1 - Isaac17 <-> MuJoCo17 (for obs joint_pos/joint_vel):")
    print(f"  isaac17_to_mujoco17: {isaac17_to_mujoco17}")
    for i17, jname in enumerate(isaac17_joint_order):
        mj_idx = isaac17_to_mujoco17[i17]
        print(f"  Isaac17[{i17:2d}] {jname:12s} <-> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]}")

    # ============================================================
    # Build mapping 2: isaac16 <-> mujoco16 (16动作关节，用于action)
    # ============================================================
    # 找到每个 isaac16 动作关节在 MuJoCo 17关节中的索引
    # isaac16_action_to_mj17[isaac16_idx] = mj17_idx
    isaac16_action_to_mj17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac16_action_order],
        dtype=np.int32
    )

    # 反向映射: mj17_to_isaac16_action[mj17_idx] = isaac16_idx (仅对16个动作关节有效)
    # 对于 Waist_2 (mj_idx=0)，不在此映射中
    mj17_to_isaac16_action = np.full((num_mj_joints,), -1, dtype=np.int32)
    for i16, mj_idx in enumerate(isaac16_action_to_mj17):
        mj17_to_isaac16_action[mj_idx] = i16

    print(f"\nMapping 2 - Isaac16 <-> MuJoCo17 (for action):")
    print(f"  isaac16_action_to_mj17: {isaac16_action_to_mj17}")
    for i16, jname in enumerate(isaac16_action_order):
        mj_idx = isaac16_action_to_mj17[i16]
        print(f"  Isaac16[{i16:2d}] {jname:12s} -> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]}")

    # ============================================================
    # Waist_2 锁定配置
    # ============================================================
    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]
    print(f"\nWaist_2: MJ idx={waist_mj_idx}, locked at {waist_default:.4f} rad")

    # ============================================================
    # 构建16动作关节的默认角度（MuJoCo中的子集）
    # ============================================================
    # default_angles_action16_mj[i] = 第i个动作关节在MuJoCo中的默认角度
    default_angles_action16_mj = default_angles[isaac16_action_to_mj17]

    # dq sign correction (Isaac17 order)
    dq_sign = np.ones((len(isaac17_joint_order),), dtype=np.float32)
    if isinstance(dq_sign_fixes_cfg, dict):
        for jn, s in dq_sign_fixes_cfg.items():
            if jn in isaac17_joint_order:
                dq_sign[isaac17_joint_order.index(jn)] = float(s)

    # Joint limits (from MuJoCo model)
    joint_limits = {}
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if m.jnt_limited[jid]:
            joint_limits[jname] = (float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1]))

    # Load policy
    policy = None
    if args.no_policy:
        print("\n--no-policy flag set. Running in STANCE MODE (PD hold only, no policy).")
    elif os.path.exists(policy_path) and os.path.isfile(policy_path):
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        print("Policy loaded successfully!")
    else:
        print(f"\nWARNING: No policy found at {policy_path}. Running in DEBUG MODE (holding default stance).")

    # Initialize state
    action_isaac16 = np.zeros(num_actions, dtype=np.float32)  # 16 actions
    action_isaac16_prev = np.zeros(num_actions, dtype=np.float32)  # for low-pass filter
    prev_obs = np.zeros(num_obs, dtype=np.float32)  # for observation low-pass filter
    target_dof_pos = default_angles.copy()  # MuJoCo full joint order (17)
    obs = np.zeros(num_obs, dtype=np.float32)  # 62
    counter = 0
    policy_step_count = 0  # count policy inference steps (for ramp-up)

    print(f"\n{'='*60}")
    print(f"V4 Quadruped Sim2Sim Configuration:")
    print(f"  num_actions: {num_actions} (excluding Waist_2)")
    print(f"  num_obs: {num_obs}")
    print(f"  action_scale: {action_scale}")
    print(f"  action_clip: {action_clip}")
    print(f"  action_filter_alpha: {action_filter_alpha}")
    print(f"  obs_filter_alpha: {obs_filter_alpha} (mode={obs_filter_mode})")
    print(f"  action_ramp_steps: {action_ramp_steps}")
    print(f"  control_decimation: {control_decimation}")
    print(f"  simulation_dt: {simulation_dt}")
    print(f"  v4_coordinate_remap: {v4_coordinate_remap}")
    print(f"  cmd_init: {cmd}")
    print(f"  Waist_2 locked at: {waist_default:.4f} rad")
    print(f"{'='*60}\n")

    # ============================================================
    # Keyboard controller
    # ============================================================
    kb_controller = None
    if not args.no_keyboard:
        kb_controller = KeyboardController(cmd)
        print("\n  Keyboard controls:")
        print("    W/S: forward/backward  |  A/D: left/right  |  Q/E: turn")
        print("    SPACE: stop  |  R: reset pose")

    print("\nStarting simulation...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # ============================================================
        # Warmup phase: let robot settle with position actuators
        # Position actuator: ctrl = target angle, MuJoCo internally computes
        #   tau = kp*(ctrl-q) - kd*dq (隐式积分，更稳定)
        # ============================================================
        print(f"Warmup: settling for {warmup_steps} steps ({warmup_steps * simulation_dt:.1f}s)...")
        for ws in range(warmup_steps):
            # Set target positions for position actuators (actuator order)
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if ws % 100 == 0:
                viewer.sync()
        viewer.sync()
        print(f"Warmup done. Height: {d.qpos[2]:.4f}m, quat: [{d.qpos[3]:.4f}, {d.qpos[4]:.4f}, {d.qpos[5]:.4f}, {d.qpos[6]:.4f}]")

        # Reset counter and velocities for clean policy start
        counter = 0
        d.qvel[:] = 0  # Zero out all velocities for clean start

        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))  # ~60Hz render = every 3 steps at dt=0.005
        last_print_time = 0.0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # ============================================================
            # Handle reset request from keyboard
            # ============================================================
            if kb_controller and kb_controller.reset_requested:
                kb_controller.reset_requested = False
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action_isaac16[:] = 0
                action_isaac16_prev[:] = 0
                prev_obs[:] = 0
                counter = 0
                policy_step_count = 0
                start = time.time()
                print("\n[RESET] Robot pose reset!")
                continue

            # ============================================================
            # Position Actuator Control (all 17 joints)
            # ctrl = target angle, MuJoCo internally computes:
            #   tau = kp*(ctrl-q) - kd*dq (隐式积分)
            # ============================================================

            # Set target positions for position actuators (actuator order)
            d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]

            # Step simulation
            mujoco.mj_step(m, d)
            counter += 1

            # Render at ~60Hz (not every physics step) to reduce GPU load
            if counter % render_interval == 0:
                viewer.sync()

            # ============================================================
            # Policy inference at control frequency
            # ============================================================
            if counter % control_decimation == 0 and policy is not None:
                quat = d.qpos[3:7]  # MuJoCo quaternion [w,x,y,z]
                base_lin_vel_world = d.qvel[0:3].copy()
                base_ang_vel_world = d.qvel[3:6].copy()

                # Transform to body frame
                base_lin_vel = world_to_body(base_lin_vel_world, quat)
                omega = world_to_body(base_ang_vel_world, quat)

                # --------------------------------------------------------
                # MuJoCo → Isaac17 (for observation: joint_pos, joint_vel)
                # --------------------------------------------------------
                qj_mujoco = d.qpos[7:].copy()      # 17 joints, MuJoCo order
                dqj_mujoco = d.qvel[6:].copy()      # 17 joints, MuJoCo order

                # Reorder: MuJoCo order → Isaac17 order (all 17 joints)
                qj_isaac17 = qj_mujoco[isaac17_to_mujoco17]
                dqj_isaac17 = dqj_mujoco[isaac17_to_mujoco17] * dq_sign

                # default_angles is MuJoCo order, convert to Isaac17 order
                default_angles_isaac17 = default_angles[isaac17_to_mujoco17]

                # Compute gravity vector in body frame
                gravity_orientation = get_gravity_orientation(quat)

                # --------------------------------------------------------
                # V4 坐标系重映射
                # --------------------------------------------------------
                if v4_coordinate_remap:
                    base_lin_vel_obs = v4_remap_lin_vel(base_lin_vel)
                    omega_obs = v4_remap_ang_vel(omega)
                    gravity_obs = v4_remap_gravity(gravity_orientation)
                else:
                    base_lin_vel_obs = base_lin_vel
                    omega_obs = omega
                    gravity_obs = gravity_orientation

                # Scale observations
                base_lin_vel_obs = base_lin_vel_obs * lin_vel_scale
                omega_obs = omega_obs * ang_vel_scale
                qj = (qj_isaac17 - default_angles_isaac17) * dof_pos_scale   # 17 joints
                dqj = dqj_isaac17 * dof_vel_scale                             # 17 joints

                # --------------------------------------------------------
                # Build observation vector
                # --------------------------------------------------------
                # obs = [lin_vel(3), ang_vel(3), gravity(3), cmd(3),
                #        joint_pos_rel(17), joint_vel(17), last_action(16)]
                # Total: 62
                obs[0:3] = base_lin_vel_obs
                obs[3:6] = omega_obs
                obs[6:9] = gravity_obs
                obs[9:12] = cmd * cmd_scale
                obs[12:29] = qj                                    # 17 joints (Isaac17 order)
                obs[29:46] = dqj                                   # 17 joints (Isaac17 order)
                obs[46:62] = action_isaac16.astype(np.float32)     # 16 actions (Isaac16 order)

                # Observation low-pass filter: smooth MuJoCo bounce noise
                # obs = alpha * prev_obs + (1-alpha) * new_obs
                if obs_filter_alpha > 0 and policy_step_count > 0:
                    if obs_filter_mode == "vel_only":
                        # Only filter velocity components (more natural motion):
                        # lin_vel(0:3), ang_vel(3:6), joint_vel(29:46)
                        obs[0:6] = obs_filter_alpha * prev_obs[0:6] + (1.0 - obs_filter_alpha) * obs[0:6]
                        obs[29:46] = obs_filter_alpha * prev_obs[29:46] + (1.0 - obs_filter_alpha) * obs[29:46]
                    else:  # "all"
                        obs[:] = obs_filter_alpha * prev_obs + (1.0 - obs_filter_alpha) * obs
                prev_obs[:] = obs

                # Status print (every 2 seconds)
                t_now = time.time() - start
                if t_now - last_print_time >= 2.0:
                    last_print_time = t_now
                    pos = d.qpos[0:3]
                    print(f"[t={t_now:5.1f}s] h={pos[2]:.3f}m pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                          f"ncon={d.ncon} act_max={np.max(np.abs(action_isaac16)):.2f}")

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_isaac16 = policy(obs_tensor).detach().numpy().squeeze()  # 16 actions

                # Action post-processing
                if use_tanh_action:
                    action_isaac16 = np.tanh(action_isaac16)
                if action_clip is not None:
                    action_isaac16 = np.clip(action_isaac16, -action_clip, action_clip)

                # Action ramp-up: gradually increase action magnitude over first N steps
                # Prevents initial shock from normalizer bias (first action can be extreme)
                if action_ramp_steps > 0 and policy_step_count < action_ramp_steps:
                    ramp_factor = float(policy_step_count) / float(action_ramp_steps)
                    action_isaac16 = action_isaac16 * ramp_factor

                # Low-pass filter: smoothed = alpha * old + (1-alpha) * new
                if action_filter_alpha > 0:
                    action_isaac16 = action_filter_alpha * action_isaac16_prev + (1.0 - action_filter_alpha) * action_isaac16
                action_isaac16_prev[:] = action_isaac16

                policy_step_count += 1

                # --------------------------------------------------------
                # Isaac16 → MuJoCo17 (for action: update target_dof_pos)
                # --------------------------------------------------------
                # 1. Waist_2 始终锁定在默认位置
                target_dof_pos[waist_mj_idx] = waist_default

                # 2. 16个动作关节：action * scale + default
                for i16 in range(num_actions):
                    mj_idx = isaac16_action_to_mj17[i16]
                    target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]

                # Apply joint limits (MuJoCo order)
                for i, jname in enumerate(mj_joint_names):
                    if jname in joint_limits:
                        low, high = joint_limits[jname]
                        target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)

            # Timing - maintain real-time simulation
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if kb_controller:
        kb_controller.stop()
    print("Simulation completed.")
