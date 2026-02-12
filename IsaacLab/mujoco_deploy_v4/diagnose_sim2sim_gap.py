"""
Diagnose sim2sim gap between IsaacLab and MuJoCo for V4 quadruped.

Runs MuJoCo simulation headless for a few seconds and logs detailed observations
at each policy step, comparing with IsaacLab baseline behavior.

IsaacLab baseline (zero command):
  - Height: 0.221-0.222m
  - Forward velocity: oscillates ±0.04 m/s
  - Yaw rate: ±0.15 rad/s
  - act_max: drops to 1.35 by t=1s
  - Position drift: < 0.04m over 10s

Usage:
  cd IsaacLab/mujoco_deploy_v4
  conda run -n isaaclab python diagnose_sim2sim_gap.py v4_robot.yaml
"""

import time
import mujoco
import numpy as np
import torch
import yaml
import os
import argparse

try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ImportError:
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def quat_to_rotmat_wxyz(quat_wxyz):
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
    R_wb = quat_to_rotmat_wxyz(quat_wxyz)
    return R_wb.T @ v_world


def v4_remap_lin_vel(lin_vel_body):
    return np.array([lin_vel_body[2], lin_vel_body[0], lin_vel_body[1]])

def v4_remap_ang_vel(ang_vel_body):
    return np.array([ang_vel_body[0], ang_vel_body[2], ang_vel_body[1]])

def v4_remap_gravity(gravity_body):
    return np.array([gravity_body[2], gravity_body[0], gravity_body[1]])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration in seconds")
    parser.add_argument("--cmd", type=float, nargs=3, default=None, help="Override command [fwd, lat, yaw]")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, args.config_file)
    if not os.path.exists(config_path):
        config_path = args.config_file

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    if not os.path.exists(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if args.cmd is not None:
        cmd = np.array(args.cmd, dtype=np.float32)

    # Load model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Joint names
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)
    num_mj_joints = len(mj_joint_names)

    # Actuator mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint_indices.append(pd_index)
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # Effort limits
    effort_limit_pd = np.full((num_mj_joints,), np.inf, dtype=np.float32)
    for act_i in range(m.nu):
        j_pd = int(actuator_to_joint_indices[act_i])
        fr = m.actuator_forcerange[act_i]
        effort_limit_pd[j_pd] = float(max(abs(fr[0]), abs(fr[1])))

    # Compensate MuJoCo joint damping
    mj_joint_damping = np.zeros(num_mj_joints, dtype=np.float32)
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof_adr = m.jnt_dofadr[jid]
        mj_joint_damping[i] = float(m.dof_damping[dof_adr])
    kds_compensated = np.maximum(kds - mj_joint_damping, 0.0)
    print(f"  MuJoCo damping: {mj_joint_damping[0]:.1f}, kds_compensated: {kds_compensated[:5]}")

    # Isaac joint orders
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

    isaac17_to_mujoco17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac17_joint_order], dtype=np.int32)
    isaac16_action_to_mj17 = np.array(
        [mj_joint_names.index(jname) for jname in isaac16_action_order], dtype=np.int32)

    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]

    # Joint limits
    joint_limits = {}
    for i, jname in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if m.jnt_limited[jid]:
            joint_limits[jname] = (float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1]))

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"Policy loaded: {policy_path}")

    # Init
    init_height = 0.22
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles

    # Warmup
    warmup_steps = 1000
    target_dof_pos = default_angles.copy()
    print(f"Warmup: {warmup_steps} steps ({warmup_steps * simulation_dt:.1f}s)...")
    for ws in range(warmup_steps):
        current_q = d.qpos[7:]
        current_dq = d.qvel[6:]
        tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds_compensated), current_dq, kds_compensated)
        tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)
        d.ctrl[:] = tau[actuator_to_joint_indices]
        mujoco.mj_step(m, d)

    print(f"Warmup done. Height: {d.qpos[2]:.4f}m")
    print(f"  quat: [{d.qpos[3]:.4f}, {d.qpos[4]:.4f}, {d.qpos[5]:.4f}, {d.qpos[6]:.4f}]")

    # Check joint positions after warmup
    qj_mj = d.qpos[7:].copy()
    print(f"\nJoint positions after warmup (MuJoCo order):")
    for i, jname in enumerate(mj_joint_names):
        err = qj_mj[i] - default_angles[i]
        print(f"  {jname:12s}: actual={qj_mj[i]:+.4f}  default={default_angles[i]:+.4f}  err={err:+.6f}")

    # Zero velocities for clean start
    d.qvel[:] = 0
    counter = 0
    action_isaac16 = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    print(f"\n{'='*80}")
    print(f"Sim2Sim Gap Diagnosis - cmd={cmd.tolist()}, duration={args.duration}s")
    print(f"{'='*80}")

    # Collect data
    total_policy_steps = int(args.duration / (simulation_dt * control_decimation))
    
    print(f"\n{'step':>4s} | {'t':>5s} | {'height':>6s} | {'lin_vel_obs':>30s} | {'ang_vel_obs':>30s} | {'gravity_obs':>30s} | {'act_max':>7s} | {'jpos_err_max':>12s} | {'jvel_max':>8s} | {'ncon':>4s}")
    print("-" * 160)

    for policy_step in range(total_policy_steps):
        # Run physics for decimation steps
        for _ in range(control_decimation):
            current_q = d.qpos[7:]
            current_dq = d.qvel[6:]
            tau = pd_control(target_dof_pos, current_q, kps, np.zeros_like(kds_compensated), current_dq, kds_compensated)
            tau = np.clip(tau, -effort_limit_pd, effort_limit_pd)
            d.ctrl[:] = tau[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            counter += 1

        # Observation
        quat = d.qpos[3:7]
        base_lin_vel_world = d.qvel[0:3].copy()
        base_ang_vel_world = d.qvel[3:6].copy()

        base_lin_vel = world_to_body(base_lin_vel_world, quat)
        omega = world_to_body(base_ang_vel_world, quat)

        qj_mujoco = d.qpos[7:].copy()
        dqj_mujoco = d.qvel[6:].copy()

        qj_isaac17 = qj_mujoco[isaac17_to_mujoco17]
        dqj_isaac17 = dqj_mujoco[isaac17_to_mujoco17]
        default_angles_isaac17 = default_angles[isaac17_to_mujoco17]

        gravity_orientation = get_gravity_orientation(quat)

        base_lin_vel_obs = v4_remap_lin_vel(base_lin_vel)
        omega_obs = v4_remap_ang_vel(omega)
        gravity_obs = v4_remap_gravity(gravity_orientation)

        qj = qj_isaac17 - default_angles_isaac17
        dqj = dqj_isaac17

        obs[0:3] = base_lin_vel_obs
        obs[3:6] = omega_obs
        obs[6:9] = gravity_obs
        obs[9:12] = cmd * cmd_scale
        obs[12:29] = qj
        obs[29:46] = dqj
        obs[46:62] = action_isaac16.astype(np.float32)

        # Policy
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action_isaac16 = policy(obs_tensor).detach().numpy().squeeze()

        # Update targets
        target_dof_pos[waist_mj_idx] = waist_default
        for i16 in range(num_actions):
            mj_idx = isaac16_action_to_mj17[i16]
            target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]
        for i, jname in enumerate(mj_joint_names):
            if jname in joint_limits:
                low, high = joint_limits[jname]
                target_dof_pos[i] = np.clip(target_dof_pos[i], low, high)

        # Log
        t = policy_step * simulation_dt * control_decimation
        height = d.qpos[2]
        act_max = np.max(np.abs(action_isaac16))
        jpos_err_max = np.max(np.abs(qj))
        jvel_max = np.max(np.abs(dqj))

        if policy_step < 20 or policy_step % 25 == 0:
            print(f"{policy_step:4d} | {t:5.2f} | {height:6.4f} | "
                  f"[{base_lin_vel_obs[0]:+.4f},{base_lin_vel_obs[1]:+.4f},{base_lin_vel_obs[2]:+.4f}] | "
                  f"[{omega_obs[0]:+.4f},{omega_obs[1]:+.4f},{omega_obs[2]:+.4f}] | "
                  f"[{gravity_obs[0]:+.4f},{gravity_obs[1]:+.4f},{gravity_obs[2]:+.4f}] | "
                  f"{act_max:7.3f} | {jpos_err_max:12.6f} | {jvel_max:8.4f} | {d.ncon:4d}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"Final state:")
    print(f"  Position: ({d.qpos[0]:+.4f}, {d.qpos[1]:+.4f}, {d.qpos[2]:.4f})")
    print(f"  Quaternion: [{d.qpos[3]:.4f}, {d.qpos[4]:.4f}, {d.qpos[5]:.4f}, {d.qpos[6]:.4f}]")
    print(f"  Contacts: {d.ncon}")

    # Detailed per-joint analysis at final step
    print(f"\nFinal joint state (Isaac17 order):")
    print(f"  {'Joint':12s} | {'pos':>8s} | {'default':>8s} | {'pos_rel':>8s} | {'vel':>8s} | {'target':>8s} | {'torque':>8s}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for i17, jname in enumerate(isaac17_joint_order):
        mj_idx = isaac17_to_mujoco17[i17]
        pos = qj_mujoco[mj_idx]
        default = default_angles[mj_idx]
        pos_rel = pos - default
        vel = dqj_mujoco[mj_idx]
        target = target_dof_pos[mj_idx]
        tau_j = (target - pos) * kps[mj_idx] + (0 - vel) * kds[mj_idx]
        print(f"  {jname:12s} | {pos:+8.4f} | {default:+8.4f} | {pos_rel:+8.4f} | {vel:+8.4f} | {target:+8.4f} | {tau_j:+8.2f}")

    # Compare with IsaacLab baseline
    print(f"\n{'='*80}")
    print(f"Comparison with IsaacLab baseline (zero command):")
    print(f"  IsaacLab: height=0.221-0.222m, fwd_vel=±0.04, yaw=±0.15, act_max=1.35 (t=1s)")
    print(f"  MuJoCo:   height={d.qpos[2]:.3f}m, act_max={np.max(np.abs(action_isaac16)):.2f}")
    print(f"  Position drift: ({d.qpos[0]:+.3f}, {d.qpos[1]:+.3f}) vs IsaacLab (~0.03m drift)")
    
    # Check for asymmetry in actions (potential cause of circling)
    print(f"\nAction asymmetry analysis (left vs right):")
    left_right_pairs = [
        ('LHIPp', 'RHIPp', 0, 1),
        ('LHIPy', 'RHIPy', 2, 3),
        ('LSDp', 'RSDp', 4, 5),
        ('LKNEEp', 'RKNEEP', 6, 7),
        ('LSDy', 'RSDy', 8, 9),
        ('LANKLEp', 'RANKLEp', 10, 11),
        ('LARMp', 'RARMp', 12, 13),
        ('LARMAp', 'RARMAP', 14, 15),
    ]
    for lname, rname, lidx, ridx in left_right_pairs:
        la = action_isaac16[lidx]
        ra = action_isaac16[ridx]
        diff = la - ra
        print(f"  {lname:8s}={la:+.4f}  {rname:8s}={ra:+.4f}  diff={diff:+.4f}")
