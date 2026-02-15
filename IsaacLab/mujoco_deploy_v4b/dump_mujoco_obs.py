"""
Dump MuJoCo observations for comparison with Isaac Lab.
Runs headless for N policy steps and saves obs, actions, joint states.

Usage:
  python dump_mujoco_obs.py v4_robot.yaml --steps 50 --forward_vel 0.5
"""
import mujoco
import numpy as np
import torch
import yaml
import os
import argparse


def get_gravity_orientation(quaternion):
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def v4_remap_ang_vel(ang_vel_body):
    return np.array([ang_vel_body[0], ang_vel_body[2], ang_vel_body[1]])


def v4_remap_gravity(gravity_body):
    return np.array([gravity_body[2], gravity_body[0], gravity_body[1]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--forward_vel", type=float, default=0.5)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, args.config_file)
    if not os.path.exists(config_path):
        config_path = args.config_file

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    policy_path = config["policy_path"]
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
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    v4_coordinate_remap = config.get("v4_coordinate_remap", False)

    cmd = np.array([args.forward_vel, 0.0, 0.0], dtype=np.float32)

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

    print(f"MuJoCo joints: {mj_joint_names}")

    # Actuator mapping
    actuator_to_joint_indices = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_to_joint_indices.append(mj_joint_names.index(joint_name))
    actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

    # Isaac joint orders (must match USD/PhysX traversal order from DOG_V5.usd)
    isaac13_joint_order = [
        'LHIPp', 'LHIPy', 'LKNEEp',
        'RHIPp', 'RHIPy', 'RKNEEP',
        'Waist_2',
        'LSDp', 'LSDy', 'LARMp',
        'RSDp', 'RSDy', 'RARMp',
    ]
    # Isaac 12-action order (from flat_env_cfg.py ActionsCfg joint_names list)
    isaac12_action_order = [
        'RSDp', 'RSDy', 'RARMp',
        'LSDp', 'LSDy', 'LARMp',
        'RHIPp', 'RHIPy', 'RKNEEP',
        'LHIPp', 'LHIPy', 'LKNEEp',
    ]

    isaac13_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac13_joint_order], dtype=np.int32)
    isaac12_action_to_mj = np.array([mj_joint_names.index(j) for j in isaac12_action_order], dtype=np.int32)
    waist_mj_idx = mj_joint_names.index('Waist_2')
    waist_default = default_angles[waist_mj_idx]

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"Policy loaded: {policy_path}")

    # Initialize
    init_height = float(config.get("init_height", 0.3))
    d.qpos[2] = init_height
    d.qpos[3:7] = [0.70710678, 0.70710678, 0.0, 0.0]
    d.qpos[7:] = default_angles

    action_raw = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Warmup
    warmup_steps = int(5.0 / simulation_dt)
    print(f"Warmup: {warmup_steps} steps...")
    for _ in range(warmup_steps):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    print(f"Warmup done. Height: {d.qpos[2]:.4f}m")

    # Run policy and dump
    all_data = []
    counter = 0
    policy_step = 0

    for sim_step in range(args.steps * control_decimation):
        d.ctrl[:] = target_dof_pos[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        counter += 1

        if counter % control_decimation == 0:
            quat = d.qpos[3:7]
            omega = d.qvel[3:6].copy()
            qj_mujoco = d.qpos[7:].copy()
            dqj_mujoco = d.qvel[6:].copy()

            qj_isaac13 = qj_mujoco[isaac13_to_mujoco]
            dqj_isaac13 = dqj_mujoco[isaac13_to_mujoco]
            default_angles_isaac13 = default_angles[isaac13_to_mujoco]

            gravity_orientation = get_gravity_orientation(quat)

            if v4_coordinate_remap:
                omega_obs = v4_remap_ang_vel(omega)
                gravity_obs = v4_remap_gravity(gravity_orientation)
            else:
                omega_obs = omega
                gravity_obs = gravity_orientation

            omega_obs_scaled = omega_obs * ang_vel_scale
            qj = (qj_isaac13 - default_angles_isaac13) * dof_pos_scale
            dqj = dqj_isaac13 * dof_vel_scale

            obs[0:3] = omega_obs_scaled
            obs[3:6] = gravity_obs
            obs[6:9] = cmd * cmd_scale
            obs[9:22] = qj
            obs[22:35] = dqj
            obs[35:47] = action_raw.astype(np.float32)

            frame_data = {
                'step': policy_step,
                'obs': obs.copy(),
                'actions': action_raw.copy(),
                'root_pos': d.qpos[0:3].copy(),
                'root_quat': quat.copy(),
                'omega_body': omega.copy(),
                'omega_obs': omega_obs.copy(),
                'gravity_body': gravity_orientation.copy(),
                'gravity_obs': gravity_obs.copy(),
                'qj_mujoco': qj_mujoco.copy(),
                'dqj_mujoco': dqj_mujoco.copy(),
                'qj_isaac13': qj_isaac13.copy(),
                'dqj_isaac13': dqj_isaac13.copy(),
                'default_angles_isaac13': default_angles_isaac13.copy(),
                'joint_names_mujoco': mj_joint_names,
                'joint_names_isaac13': isaac13_joint_order,
            }
            all_data.append(frame_data)

            if policy_step < 5 or policy_step % 10 == 0:
                o = obs
                print(f"\n[Step {policy_step}]")
                print(f"  obs[0:3]  ang_vel:  {o[0:3]}")
                print(f"  obs[3:6]  gravity:  {o[3:6]}")
                print(f"  obs[6:9]  cmd:      {o[6:9]}")
                print(f"  obs[9:22] joint_pos:{o[9:22]}")
                print(f"  obs[22:35]joint_vel:{o[22:35]}")
                print(f"  obs[35:47]last_act: {o[35:47]}")
                print(f"  root_quat (w,x,y,z): {quat}")
                print(f"  omega_body: {omega}")
                print(f"  gravity_body: {gravity_orientation}")
                print(f"  qj_mujoco: {qj_mujoco}")
                print(f"  height: {d.qpos[2]:.4f}m")

            # Run policy
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_raw = policy(obs_tensor).detach().numpy().squeeze()

            # Apply actions
            target_dof_pos[waist_mj_idx] = waist_default
            for i12 in range(num_actions):
                mj_idx = isaac12_action_to_mj[i12]
                target_dof_pos[mj_idx] = action_raw[i12] * action_scale + default_angles[mj_idx]

            policy_step += 1

    # Save
    save_path = os.path.join(current_dir, "mujoco_obs_dump.npz")
    np.savez(save_path, data=all_data)
    print(f"\nSaved {len(all_data)} frames to: {save_path}")
