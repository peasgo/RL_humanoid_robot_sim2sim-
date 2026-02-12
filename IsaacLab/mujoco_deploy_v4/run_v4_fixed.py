"""
V4 Quadruped Sim2Sim — 修正版
参考 humanoid-gym sim2sim.py 的简洁做法，去除所有band-aid。

关键修正：
1. action_scale = 0.25 (与训练一致)
2. 观测中 joint_pos/joint_vel 使用全17关节，顺序=MuJoCo顺序=IsaacLab内部顺序
3. 动作输出16关节，顺序=IsaacLab action joint_names 排序后的顺序
4. 去除 mass_scale, action_filter, obs_filter, action_ramp 等band-aid
5. 使用 scipy Rotation 做坐标变换（与humanoid-gym一致）
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import os
import argparse
import threading
import sys
from scipy.spatial.transform import Rotation as R


def get_obs(data, model):
    """从MuJoCo提取观测量，参考humanoid-gym的get_obs()。
    返回: (quat_xyzw, lin_vel_body, ang_vel_body, gravity_body)
    """
    # MuJoCo四元数 [w,x,y,z] → scipy需要 [x,y,z,w]
    quat_wxyz = data.qpos[3:7].copy()
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    rot = R.from_quat(quat_xyzw)

    # 世界系速度 → 体坐标系 (inverse=True 等价于 R^T @ v)
    lin_vel_world = data.qvel[0:3].copy()
    ang_vel_world = data.qvel[3:6].copy()

    lin_vel_body = rot.apply(lin_vel_world, inverse=True)
    ang_vel_body = rot.apply(ang_vel_world, inverse=True)

    # 重力向量在体坐标系的投影
    gravity_body = rot.apply(np.array([0., 0., -1.]), inverse=True)

    return quat_xyzw, lin_vel_body, ang_vel_body, gravity_body


def v4_remap_lin_vel(v):
    """V4坐标系重映射：线速度 [+Z, X, Y]"""
    return np.array([v[2], v[0], v[1]])


def v4_remap_ang_vel(w):
    """V4坐标系重映射：角速度 [X, +Z, Y]"""
    return np.array([w[0], w[2], w[1]])


def v4_remap_gravity(g):
    """V4坐标系重映射：重力投影 [+Z, X, Y]"""
    return np.array([g[2], g[0], g[1]])


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
    parser = argparse.ArgumentParser(description="V4 Quadruped Sim2Sim (Fixed)")
    parser.add_argument("config_file", type=str, help="config file (e.g., v4_robot_fixed.yaml)")
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--no-keyboard", action="store_true")
    args = parser.parse_args()

    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(current_dir, args.config_file), args.config_file]:
        if os.path.exists(p):
            config_path = p
            break
    else:
        raise FileNotFoundError(f"Config not found: {args.config_file}")

    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Parse config
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    action_scale = config["action_scale"]
    action_clip = float(config.get("action_clip", 100.0))
    num_actions = config["num_actions"]   # 16
    num_obs = config["num_obs"]           # 62
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    v4_coordinate_remap = config.get("v4_coordinate_remap", False)
    mass_scale = float(config.get("mass_scale", 1.0))

    # Default angles (MuJoCo order, 17 joints)
    default_angles = np.array(config["default_angles"], dtype=np.float64)
    kps = np.array(config["kps"], dtype=np.float64)
    kds = np.array(config["kds"], dtype=np.float64)

    # ============================================================
    # Load MuJoCo model
    # ============================================================
    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    if mass_scale != 1.0:
        for bi in range(m.nbody):
            m.body_mass[bi] *= mass_scale
            m.body_inertia[bi] *= mass_scale
        print(f"Mass scaling: {mass_scale}x")

    # Build MuJoCo joint name list (exclude freejoint)
    mj_joint_names = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
    num_mj_joints = len(mj_joint_names)  # 17
    print(f"MuJoCo joints ({num_mj_joints}): {mj_joint_names}")

    # ============================================================
    # 关键理解：IsaacLab的关节顺序
    #
    # IsaacLab从USD加载关节，顺序由URDF/USD的树遍历决定。
    # 对于V4机器人，这个顺序恰好与MuJoCo XML的树遍历顺序相同：
    #   Waist_2, RSDp, RSDy, RARMp, RARMAP,
    #   LSDp, LSDy, LARMp, LARMAp,
    #   RHIPp, RHIPy, RKNEEP, RANKLEp,
    #   LHIPp, LHIPy, LKNEEp, LANKLEp
    #
    # 因此：MuJoCo顺序 = IsaacLab内部顺序
    # 观测中的 joint_pos_rel (17维) 和 joint_vel_rel (17维) 直接用MuJoCo顺序
    #
    # 动作关节 (16维, 排除Waist_2):
    #   训练中 joint_names = [RSDp, RSDy, RARMp, RARMAP, LSDp, LSDy, LARMp, LARMAp,
    #                         RHIPp, RHIPy, RKNEEP, RANKLEp, LHIPp, LHIPy, LKNEEp, LANKLEp]
    #   preserve_order=false → 按IsaacLab内部索引排序
    #   由于这些关节在内部的索引就是 1-16 (Waist_2是0)，排序后顺序不变
    #   所以动作顺序 = MuJoCo顺序去掉Waist_2
    # ============================================================

    # 动作关节在MuJoCo中的索引 (跳过Waist_2=index 0)
    waist_mj_idx = mj_joint_names.index('Waist_2')
    action_joint_mj_indices = [i for i in range(num_mj_joints) if i != waist_mj_idx]
    assert len(action_joint_mj_indices) == num_actions, \
        f"Expected {num_actions} action joints, got {len(action_joint_mj_indices)}"

    action_joint_names = [mj_joint_names[i] for i in action_joint_mj_indices]
    print(f"Action joints ({num_actions}): {action_joint_names}")
    print(f"Action joint MJ indices: {action_joint_mj_indices}")

    # 动作关节的默认角度
    default_angles_action = default_angles[action_joint_mj_indices]
    waist_default = default_angles[waist_mj_idx]
    print(f"Waist_2 locked at: {waist_default:.4f} rad (MJ idx={waist_mj_idx})")

    # Build actuator -> joint mapping
    actuator_to_joint = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_to_joint.append(mj_joint_names.index(joint_name))
    actuator_to_joint = np.array(actuator_to_joint, dtype=np.int32)

    # Load policy
    policy = None
    if args.no_policy:
        print("Running in STANCE MODE (no policy).")
    elif os.path.exists(policy_path):
        print(f"Loading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        print("Policy loaded.")
    else:
        print(f"WARNING: Policy not found at {policy_path}. Holding default stance.")

    # ============================================================
    # Initialize
    # ============================================================
    init_height = 0.22
    init_quat = [0.70710678, 0.70710678, 0.0, 0.0]

    action = np.zeros(num_actions, dtype=np.float64)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Set initial pose
    d.qpos[2] = init_height
    d.qpos[3:7] = init_quat
    d.qpos[7:] = default_angles

    print(f"\n{'='*60}")
    print(f"V4 Sim2Sim (Fixed) Configuration:")
    print(f"  action_scale: {action_scale}")
    print(f"  control_decimation: {control_decimation}")
    print(f"  simulation_dt: {simulation_dt}")
    print(f"  v4_coordinate_remap: {v4_coordinate_remap}")
    print(f"  mass_scale: {mass_scale}")
    print(f"  cmd_init: {cmd}")
    print(f"  num_obs: {num_obs}, num_actions: {num_actions}")
    print(f"{'='*60}\n")

    # Keyboard controller
    kb_controller = None
    if not args.no_keyboard:
        kb_controller = KeyboardController(cmd)
        print("  W/S: fwd/back | A/D: left/right | Q/E: turn | SPACE: stop | R: reset")

    # ============================================================
    # Warmup: let robot settle
    # ============================================================
    warmup_seconds = 2.0
    warmup_steps = int(warmup_seconds / simulation_dt)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print(f"Warmup: {warmup_seconds}s...")
        for ws in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            if ws % 100 == 0:
                viewer.sync()
        viewer.sync()
        print(f"Warmup done. Height: {d.qpos[2]:.4f}m")

        # Reset velocities for clean start
        d.qvel[:] = 0
        counter = 0
        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))
        last_print_time = 0.0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Handle reset
            if kb_controller and kb_controller.reset_requested:
                kb_controller.reset_requested = False
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = init_quat
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action[:] = 0
                counter = 0
                start = time.time()
                print("[RESET]")
                continue

            # Apply control
            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            counter += 1

            if counter % render_interval == 0:
                viewer.sync()

            # ============================================================
            # Policy inference at control frequency
            # ============================================================
            if counter % control_decimation == 0 and policy is not None:

                # --- 获取观测 ---
                quat_xyzw, lin_vel_body, ang_vel_body, gravity_body = get_obs(d, m)

                # 关节位置和速度 (MuJoCo顺序 = IsaacLab顺序, 全17关节)
                qj = d.qpos[7:].copy()    # 17 joints
                dqj = d.qvel[6:].copy()   # 17 joints

                # joint_pos_rel = qj - default (与训练中 joint_pos_rel 一致)
                qj_rel = qj - default_angles

                # --- V4坐标系重映射 ---
                if v4_coordinate_remap:
                    lin_vel_obs = v4_remap_lin_vel(lin_vel_body)
                    ang_vel_obs = v4_remap_ang_vel(ang_vel_body)
                    gravity_obs = v4_remap_gravity(gravity_body)
                else:
                    lin_vel_obs = lin_vel_body
                    ang_vel_obs = ang_vel_body
                    gravity_obs = gravity_body

                # --- 构建观测向量 ---
                # obs = [lin_vel(3), ang_vel(3), gravity(3), cmd(3),
                #        joint_pos_rel(17), joint_vel(17), last_action(16)]
                # Total: 3+3+3+3+17+17+16 = 62
                obs[0:3] = lin_vel_obs.astype(np.float32)
                obs[3:6] = ang_vel_obs.astype(np.float32)
                obs[6:9] = gravity_obs.astype(np.float32)
                obs[9:12] = (cmd * cmd_scale).astype(np.float32)
                obs[12:29] = qj_rel.astype(np.float32)       # 17 joints
                obs[29:46] = dqj.astype(np.float32)           # 17 joints
                obs[46:62] = action.astype(np.float32)        # 16 actions (last step)

                # --- 策略推理 ---
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                action[:] = policy(obs_tensor).detach().numpy().squeeze()

                # Clip actions
                action = np.clip(action, -action_clip, action_clip)

                # --- 动作 → 目标关节位置 ---
                # target = action * scale + default
                # 这与训练中 JointPositionAction 的行为一致:
                #   processed_actions = raw_actions * scale + offset
                #   其中 offset = default_joint_pos (use_default_offset=true)
                target_dof_pos[waist_mj_idx] = waist_default  # Waist_2 锁定
                for i, mj_idx in enumerate(action_joint_mj_indices):
                    target_dof_pos[mj_idx] = action[i] * action_scale + default_angles[mj_idx]

                # Status print
                t_now = time.time() - start
                if t_now - last_print_time >= 2.0:
                    last_print_time = t_now
                    pos = d.qpos[0:3]
                    print(f"[t={t_now:5.1f}s] h={pos[2]:.3f}m "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                          f"ncon={d.ncon} act_max={np.max(np.abs(action)):.3f}")

            # Real-time pacing
            dt_elapsed = time.time() - step_start
            if dt_elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - dt_elapsed)

    if kb_controller:
        kb_controller.stop()
    print("Done.")
