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

    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw)


    lin_vel_body = rot.apply(lin_vel_world, inverse=True)
    ang_vel_body = rot.apply(ang_vel_world, inverse=True)


    gravity_body = rot.apply(np.array([0., 0., -1.]), inverse=True)

    return lin_vel_body, ang_vel_body, gravity_body


def v4_remap_lin_vel(v):
    return np.array([v[2], v[0], v[1]])


def v4_remap_ang_vel(w):
    return np.array([w[0], w[2], w[1]])


def v4_remap_gravity(g):
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
    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)


    default_angles = np.array(cfg["default_angles"], dtype=np.float64)


    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt


    mj_joint_names = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
    num_mj_joints = len(mj_joint_names)
    print(f"MuJoCo joints ({num_mj_joints}): {mj_joint_names}")


    actuator_to_joint = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_to_joint.append(mj_joint_names.index(jname))
    actuator_to_joint = np.array(actuator_to_joint, dtype=np.int32)


    isaac17_joint_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy', 'Waist_2',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]


    isaac16_action_order = [
        'LHIPp', 'RHIPp', 'LHIPy', 'RHIPy',
        'LSDp', 'RSDp', 'LKNEEp', 'RKNEEP', 'LSDy', 'RSDy',
        'LANKLEp', 'RANKLEp', 'LARMp', 'RARMp', 'LARMAp', 'RARMAP',
    ]


    for jn in isaac17_joint_order:
        assert jn in mj_joint_names, f"Joint '{jn}' not in MuJoCo model"


    i17_to_mj = np.array([mj_joint_names.index(j) for j in isaac17_joint_order], dtype=np.int32)

    i16_to_mj = np.array([mj_joint_names.index(j) for j in isaac16_action_order], dtype=np.int32)


    default_isaac17 = default_angles[i17_to_mj]

    default_isaac16 = default_angles[i16_to_mj]


    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]

    print(f"\nIsaac17: {isaac17_joint_order}")
    print(f"Isaac16: {isaac16_action_order}")
    print(f"Waist_2: MJ[{waist_mj_idx}], locked at {waist_default:.4f}")


    policy = None
    if args.no_policy:
        print("No policy mode — PD hold only")
    elif os.path.exists(policy_path):
        policy = torch.jit.load(policy_path)
        print(f"Policy loaded: {policy_path}")


    else:
        print(f"WARNING: Policy not found at {policy_path}")


    action_isaac16 = np.zeros(num_actions, dtype=np.float64)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)


    init_height = 0.23
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles

    print(f"\nConfig: action_scale={action_scale}, decimation={control_decimation}, dt={simulation_dt}")
    print(f"  num_obs={num_obs}, num_actions={num_actions}")
    print(f"  cmd_init={cmd}")


    kb = None
    if not args.no_keyboard:
        kb = KeyboardController(cmd)
        print("  Keyboard: W/S=fwd/back, A/D=left/right, Q/E=turn, SPACE=stop, R=reset")


    print("\nStarting simulation...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        warmup_steps = int(2.0 / simulation_dt)
        for ws in range(warmup_steps):
            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            if ws % 200 == 0:
                viewer.sync()
        d.qvel[:] = 0
        viewer.sync()
        print(f"Warmup done. Height: {d.qpos[2]:.4f}m")

        counter = 0
        start = time.time()
        render_interval = max(1, int(1.0 / 60.0 / simulation_dt))
        last_print = 0.0

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()


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


            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            counter += 1

            if counter % render_interval == 0:
                viewer.sync()


            if counter % control_decimation == 0 and policy is not None:
                quat = d.qpos[3:7].copy()
                lin_vel_world = d.qvel[0:3].copy()


                _, _, gravity_body = get_body_frame_data(
                    quat, lin_vel_world, np.zeros(3)
                )


                ang_vel_body = d.qvel[3:6].copy()


                ang_vel_obs = v4_remap_ang_vel(ang_vel_body)
                gravity_obs = v4_remap_gravity(gravity_body)


                qj_mj = d.qpos[7:].copy()
                dqj_mj = d.qvel[6:].copy()
                qj_i17 = qj_mj[i17_to_mj]
                dqj_i17 = dqj_mj[i17_to_mj]


                qj_rel = qj_i17 - default_isaac17


                dqj_rel = dqj_i17


                obs[0:3] = ang_vel_obs * 0.2
                obs[3:6] = gravity_obs
                obs[6:9] = cmd
                obs[9:26] = qj_rel
                obs[26:43] = dqj_rel * 0.05
                obs[43:59] = action_isaac16.astype(np.float32)


                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                action_isaac16[:] = policy(obs_tensor).detach().numpy().squeeze()


                target_dof_pos[waist_mj_idx] = waist_default


                for i16 in range(num_actions):
                    mj_idx = i16_to_mj[i16]
                    target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]


                t_now = time.time() - start
                if t_now - last_print >= 2.0:
                    last_print = t_now
                    pos = d.qpos[0:3]
                    print(f"[t={t_now:5.1f}s] h={pos[2]:.3f}m "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                          f"ncon={d.ncon} act_rms={np.sqrt(np.mean(action_isaac16**2)):.3f}")


            dt_remaining = m.opt.timestep - (time.time() - step_start)
            if dt_remaining > 0:
                time.sleep(dt_remaining)

    if kb:
        kb.stop()
    print("Done.")
