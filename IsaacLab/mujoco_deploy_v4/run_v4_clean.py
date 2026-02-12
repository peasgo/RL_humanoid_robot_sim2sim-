#!/usr/bin/env python3
"""
V4 Quadruped Sim2Sim — Clean deployment following humanoid-gym's philosophy.

设计原则 (学习自 humanoid-gym sim2sim.py):
1. 简单直接，不加任何 band-aid (no mass_scale, no action_filter, no obs_filter, no action_ramp)
2. 观测构建严格匹配训练时的 env.yaml
3. 动作处理严格匹配训练时的 JointPositionAction
4. 使用 position actuator (匹配 IsaacLab ImplicitActuator)
5. 如果 sim2sim 不工作，说明物理参数或观测有根本性问题，需要修正根因而非打补丁

关键差异 vs humanoid-gym:
- humanoid-gym: 显式力矩控制 (data.ctrl = tau), 手动PD每个物理步
- 我们: position actuator (d.ctrl = target_angle), MuJoCo隐式积分PD
  这与 IsaacLab ImplicitActuator 更匹配 (都是隐式积分)
- humanoid-gym: 无 empirical_normalization, 手动 obs_scales
- 我们: empirical_normalization=true, policy.pt 内含 normalizer, 无需手动 scale
- humanoid-gym: frame_stack=15, 我们: 单帧
- humanoid-gym: clip_observations=18, 我们: 无 clip (训练时 clip=null)
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


def get_body_frame_data(quat_wxyz, lin_vel_world, ang_vel_world):
    """将世界坐标系的速度转换到机体坐标系。
    
    使用 scipy Rotation (与 humanoid-gym 相同的方法)。
    MuJoCo quat 是 [w,x,y,z], scipy 需要 [x,y,z,w]。
    """
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw)
    
    # 世界坐标系 → 机体坐标系 (inverse=True)
    lin_vel_body = rot.apply(lin_vel_world, inverse=True)
    ang_vel_body = rot.apply(ang_vel_world, inverse=True)
    
    # 重力向量在机体坐标系中的投影
    gravity_body = rot.apply(np.array([0., 0., -1.]), inverse=True)
    
    return lin_vel_body, ang_vel_body, gravity_body


def v4_remap_lin_vel(v):
    """V4坐标系重映射：线速度 [+Z, X, Y]
    对应训练代码: torch.stack([vel[:, 2], vel[:, 0], vel[:, 1]], dim=-1)
    """
    return np.array([v[2], v[0], v[1]])


def v4_remap_ang_vel(w):
    """V4坐标系重映射：角速度 [X, +Z, Y]
    对应训练代码: torch.stack([ang[:, 0], ang[:, 2], ang[:, 1]], dim=-1)
    """
    return np.array([w[0], w[2], w[1]])


def v4_remap_gravity(g):
    """V4坐标系重映射：重力投影 [+Z, X, Y]
    对应训练代码: torch.stack([grav[:, 2], grav[:, 0], grav[:, 1]], dim=-1)
    """
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
    parser = argparse.ArgumentParser(description="V4 Quadruped Clean Sim2Sim")
    parser.add_argument("config_file", type=str, help="config yaml file")
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--no-keyboard", action="store_true")
    args = parser.parse_args()

    # ================================================================
    # Load config
    # ================================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(current_dir, args.config_file), args.config_file]:
        if os.path.exists(p):
            config_path = p
            break
    else:
        raise FileNotFoundError(f"Config not found: {args.config_file}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    simulation_duration = cfg["simulation_duration"]
    simulation_dt = cfg["simulation_dt"]
    control_decimation = cfg["control_decimation"]
    action_scale = cfg["action_scale"]
    num_actions = cfg["num_actions"]   # 16
    num_obs = cfg["num_obs"]           # 62
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)

    # Default angles in MuJoCo joint order (17 joints)
    default_angles = np.array(cfg["default_angles"], dtype=np.float64)

    # ================================================================
    # Load MuJoCo model
    # ================================================================
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Build MuJoCo joint name list (exclude freejoint)
    mj_joint_names = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
    num_mj_joints = len(mj_joint_names)
    print(f"MuJoCo joints ({num_mj_joints}): {mj_joint_names}")

    # Build actuator -> joint mapping
    actuator_to_joint = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_to_joint.append(mj_joint_names.index(jname))
    actuator_to_joint = np.array(actuator_to_joint, dtype=np.int32)

    # ================================================================
    # IsaacLab 关节顺序 (通过 resolve_matching_names 确定)
    #
    # preserve_order=false 时，结果按 list_of_strings (即 IsaacLab 内部顺序) 排列。
    # IsaacLab 内部顺序由 URDF/USD 的关节遍历顺序决定。
    #
    # 17关节全序 (用于 joint_pos_rel, joint_vel_rel 观测):
    # 这是 asset.data.joint_pos 的顺序
    # ================================================================
    isaac17_joint_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy', 'Waist_2',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]

    # 16动作关节序 (用于 policy output 和 last_action 观测):
    # 就是 isaac17 去掉 Waist_2 (index 4)
    isaac16_action_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]

    # Verify all joints exist
    for jn in isaac17_joint_order:
        assert jn in mj_joint_names, f"Joint '{jn}' not in MuJoCo model"

    # ================================================================
    # Build mappings
    # ================================================================
    # isaac17[i] -> mujoco index
    i17_to_mj = np.array([mj_joint_names.index(j) for j in isaac17_joint_order], dtype=np.int32)
    # isaac16[i] -> mujoco index
    i16_to_mj = np.array([mj_joint_names.index(j) for j in isaac16_action_order], dtype=np.int32)

    # Default angles in Isaac17 order
    default_isaac17 = default_angles[i17_to_mj]
    # Default angles for 16 action joints in Isaac16 order
    default_isaac16 = default_angles[i16_to_mj]

    # Waist_2 config
    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]

    print(f"\nIsaac17: {isaac17_joint_order}")
    print(f"Isaac16: {isaac16_action_order}")
    print(f"Waist_2: MJ[{waist_mj_idx}], locked at {waist_default:.4f}")

    # ================================================================
    # Load policy
    # ================================================================
    policy = None
    if args.no_policy:
        print("No policy mode — PD hold only")
    elif os.path.exists(policy_path):
        policy = torch.jit.load(policy_path)
        print(f"Policy loaded: {policy_path}")
        # policy.pt 内含 normalizer (empirical_normalization=true)
        # forward(x) = actor(normalizer(x))
        # 所以我们传入原始观测，不需要手动 scale
    else:
        print(f"WARNING: Policy not found at {policy_path}")

    # ================================================================
    # Initialize
    # ================================================================
    action_isaac16 = np.zeros(num_actions, dtype=np.float64)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Set initial pose
    init_height = 0.22
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]  # X+90° rotation
    d.qpos[7:] = default_angles

    print(f"\nConfig: action_scale={action_scale}, decimation={control_decimation}, dt={simulation_dt}")
    print(f"  num_obs={num_obs}, num_actions={num_actions}")
    print(f"  cmd_init={cmd}")

    # ================================================================
    # Keyboard controller
    # ================================================================
    kb = None
    if not args.no_keyboard:
        kb = KeyboardController(cmd)
        print("  Keyboard: W/S=fwd/back, A/D=left/right, Q/E=turn, SPACE=stop, R=reset")

    # ================================================================
    # Warmup: let robot settle
    # ================================================================
    print("\nStarting simulation...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        warmup_steps = int(2.0 / simulation_dt)
        for ws in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            if ws % 200 == 0:
                viewer.sync()
        d.qvel[:] = 0  # Clean start
        viewer.sync()
        print(f"Warmup done. Height: {d.qpos[2]:.4f}m")

        counter = 0
        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))
        last_print = 0.0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Handle reset
            if kb and kb.reset_requested:
                kb.reset_requested = False
                d.qpos[0:3] = [0, 0, init_height]
                d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                target_dof_pos[:] = default_angles
                action_isaac16[:] = 0
                counter = 0
                start = time.time()
                print("[RESET]")
                continue

            # ============================================================
            # Position actuator control
            # d.ctrl = target_angle → MuJoCo computes:
            #   tau = kp*(ctrl-q) - kd*dq (implicit integration)
            # 这与 IsaacLab ImplicitActuator 的行为一致
            # ============================================================
            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            counter += 1

            if counter % render_interval == 0:
                viewer.sync()

            # ============================================================
            # Policy inference at control frequency
            # ============================================================
            if counter % control_decimation == 0 and policy is not None:
                quat = d.qpos[3:7].copy()  # [w,x,y,z]
                lin_vel_world = d.qvel[0:3].copy()
                ang_vel_world = d.qvel[3:6].copy()

                # Body frame transforms (scipy, same as humanoid-gym)
                lin_vel_body, ang_vel_body, gravity_body = get_body_frame_data(
                    quat, lin_vel_world, ang_vel_world
                )

                # V4 coordinate remap
                lin_vel_obs = v4_remap_lin_vel(lin_vel_body)
                ang_vel_obs = v4_remap_ang_vel(ang_vel_body)
                gravity_obs = v4_remap_gravity(gravity_body)

                # Joint data: MuJoCo order → Isaac17 order
                qj_mj = d.qpos[7:].copy()
                dqj_mj = d.qvel[6:].copy()
                qj_i17 = qj_mj[i17_to_mj]
                dqj_i17 = dqj_mj[i17_to_mj]

                # joint_pos_rel = joint_pos - default_joint_pos (17 joints)
                # 训练时: scale=null → 无缩放
                qj_rel = qj_i17 - default_isaac17

                # joint_vel_rel = joint_vel - default_joint_vel (default_vel=0)
                # 训练时: scale=null → 无缩放
                dqj_rel = dqj_i17

                # --------------------------------------------------------
                # Build observation (严格匹配 env.yaml 的 observations.policy)
                # --------------------------------------------------------
                # 训练时所有 obs 的 scale=null，所以不需要手动缩放
                # empirical_normalization=true → policy.pt 内含 normalizer
                #
                # 顺序: base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
                #        velocity_commands(3), joint_pos(17), joint_vel(17),
                #        actions(16) = 62
                obs[0:3] = lin_vel_obs
                obs[3:6] = ang_vel_obs
                obs[6:9] = gravity_obs
                obs[9:12] = cmd  # velocity_commands (generated_commands)
                obs[12:29] = qj_rel       # 17 joints, Isaac17 order
                obs[29:46] = dqj_rel      # 17 joints, Isaac17 order
                obs[46:62] = action_isaac16.astype(np.float32)  # 16 actions, Isaac16 order

                # 训练时 clip=null → 不 clip
                # (humanoid-gym clips at 18, 但我们训练时没有 clip)

                # Policy inference
                # policy.pt forward: normalizer(x) → actor(x)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                action_isaac16[:] = policy(obs_tensor).detach().numpy().squeeze()

                # 训练时 clip_actions=null → 不 clip
                # (humanoid-gym clips actions, 但我们训练时没有)

                # --------------------------------------------------------
                # Action → target joint positions
                # 训练时: processed_actions = raw_actions * scale + offset
                #   scale = 0.25
                #   offset = default_joint_pos[:, joint_ids] (use_default_offset=true)
                # 然后: asset.set_joint_position_target(processed_actions, joint_ids)
                # --------------------------------------------------------
                # Waist_2 always locked
                target_dof_pos[waist_mj_idx] = waist_default

                # 16 action joints
                for i16 in range(num_actions):
                    mj_idx = i16_to_mj[i16]
                    target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]

                # Status print
                t_now = time.time() - start
                if t_now - last_print >= 2.0:
                    last_print = t_now
                    pos = d.qpos[0:3]
                    print(f"[t={t_now:5.1f}s] h={pos[2]:.3f}m "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                          f"ncon={d.ncon} act_rms={np.sqrt(np.mean(action_isaac16**2)):.3f}")

            # Real-time pacing
            dt_remaining = m.opt.timestep - (time.time() - step_start)
            if dt_remaining > 0:
                time.sleep(dt_remaining)

    if kb:
        kb.stop()
    print("Done.")
