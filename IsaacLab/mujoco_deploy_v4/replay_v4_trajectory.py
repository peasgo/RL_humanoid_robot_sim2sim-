#!/usr/bin/env python3
"""
V4 Quadruped - Replay IsaacLab trajectory in MuJoCo.

This script loads a recorded HDF5 trajectory from IsaacLab and replays it
in MuJoCo to verify joint ordering and coordinate system alignment.

Usage:
    cd /home/rl/RL-human_robot/IsaacLab/mujoco_deploy_v4
    python replay_v4_trajectory.py v4_trajectory.h5 --xml v4_scene.xml --mode position
"""

import argparse
import h5py
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def load_trajectory(filepath):
    """Load trajectory from HDF5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        data['time'] = f['time'][:]
        data['base_pos'] = f['base_pos'][:]
        data['base_quat'] = f['base_quat'][:]
        data['base_lin_vel'] = f['base_lin_vel'][:]
        data['base_ang_vel'] = f['base_ang_vel'][:]
        data['joint_pos'] = f['joint_pos'][:]
        data['joint_vel'] = f['joint_vel'][:]
        data['actions'] = f['actions'][:]
        data['observations'] = f['observations'][:]
        if 'default_joint_pos' in f:
            data['default_joint_pos'] = f['default_joint_pos'][:]
        data['joint_names'] = [name.decode('utf-8') if isinstance(name, bytes) else name
                               for name in f['joint_names'][:]]
        data['dt'] = f.attrs['dt']
        data['num_steps'] = f.attrs['num_steps']
    return data


def main():
    parser = argparse.ArgumentParser(description="Replay V4 Quadruped IsaacLab trajectory in MuJoCo")
    parser.add_argument("trajectory", type=str, help="Path to trajectory HDF5 file")
    parser.add_argument("--xml", type=str, default="v4_scene.xml", help="MuJoCo scene XML file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--mode", type=str, default="position", choices=["position", "velocity"],
                        help="Replay mode: position (set qpos/qvel) or velocity (use recorded velocities)")
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from: {args.trajectory}")
    traj = load_trajectory(args.trajectory)
    print(f"Loaded {traj['num_steps']} steps, dt={traj['dt']:.4f}s")
    print(f"\nIsaacLab joint names ({len(traj['joint_names'])} joints):")
    for i, name in enumerate(traj['joint_names']):
        print(f"  [{i}] {name}")

    if 'default_joint_pos' in traj:
        print(f"\nDefault joint positions:")
        for i, name in enumerate(traj['joint_names']):
            print(f"  [{i}] {name}: {traj['default_joint_pos'][i]:.4f}")

    # Load MuJoCo model
    xml_path = Path(args.xml)
    if not xml_path.exists():
        xml_path = Path(__file__).parent / args.xml

    print(f"\nLoading MuJoCo model from: {xml_path}")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    # Get MuJoCo joint names (excluding freejoint)
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    print(f"\nMuJoCo joint names ({len(mj_joint_names)} joints):")
    for i, name in enumerate(mj_joint_names):
        print(f"  [{i}] {name}")

    # Build joint mapping: IsaacLab index -> MuJoCo qpos address
    isaac_to_mj_qpos = []
    isaac_to_mj_qvel = []
    mapped_count = 0
    print(f"\nJoint mapping (IsaacLab -> MuJoCo):")
    for i, isaac_name in enumerate(traj['joint_names']):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, isaac_name)
        if jid == -1:
            print(f"  WARNING: Joint '{isaac_name}' not found in MuJoCo model!")
            isaac_to_mj_qpos.append(-1)
            isaac_to_mj_qvel.append(-1)
        else:
            qpos_addr = m.jnt_qposadr[jid]
            qvel_addr = m.jnt_dofadr[jid]
            isaac_to_mj_qpos.append(qpos_addr)
            isaac_to_mj_qvel.append(qvel_addr)
            mj_idx = mj_joint_names.index(isaac_name) if isaac_name in mj_joint_names else -1
            print(f"  Isaac[{i}] {isaac_name} -> MuJoCo qpos[{qpos_addr}] qvel[{qvel_addr}] (mj_idx={mj_idx})")
            mapped_count += 1

    print(f"\nMapped {mapped_count}/{len(traj['joint_names'])} joints")

    # Replay trajectory
    print(f"\nReplaying trajectory in {args.mode} mode...")
    print("Press ESC to exit")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        start_time = time.time()

        while viewer.is_running() and step < traj['num_steps']:
            step_start = time.time()

            if args.mode == "position":
                # Set base position and quaternion
                d.qpos[0:3] = traj['base_pos'][step]
                d.qpos[3:7] = traj['base_quat'][step]

                # Set joint positions
                for i in range(len(traj['joint_names'])):
                    if isaac_to_mj_qpos[i] >= 0:
                        d.qpos[isaac_to_mj_qpos[i]] = traj['joint_pos'][step][i]

                # Set velocities
                d.qvel[0:3] = traj['base_lin_vel'][step]
                d.qvel[3:6] = traj['base_ang_vel'][step]

                for i in range(len(traj['joint_names'])):
                    if isaac_to_mj_qvel[i] >= 0:
                        d.qvel[isaac_to_mj_qvel[i]] = traj['joint_vel'][step][i]

                # Forward kinematics (no dynamics step)
                mujoco.mj_forward(m, d)

            else:  # velocity mode
                d.qvel[0:3] = traj['base_lin_vel'][step]
                d.qvel[3:6] = traj['base_ang_vel'][step]

                for i in range(len(traj['joint_names'])):
                    if isaac_to_mj_qvel[i] >= 0:
                        d.qvel[isaac_to_mj_qvel[i]] = traj['joint_vel'][step][i]

                mujoco.mj_step(m, d)

            viewer.sync()

            # Print info periodically
            if step % 50 == 0:
                print(f"Step {step}/{traj['num_steps']} | "
                      f"Time: {traj['time'][step]:.2f}s | "
                      f"Base height: {d.qpos[2]:.3f}m | "
                      f"Base quat: [{d.qpos[3]:.3f}, {d.qpos[4]:.3f}, {d.qpos[5]:.3f}, {d.qpos[6]:.3f}]")

            # Timing control
            elapsed = time.time() - step_start
            target_dt = traj['dt'] / args.speed
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            step += 1

        total_time = time.time() - start_time
        print(f"\nReplay completed in {total_time:.2f}s (real time)")
        print(f"Simulated time: {traj['time'][-1]:.2f}s")
        print(f"Speed ratio: {traj['time'][-1]/total_time:.2f}x")


if __name__ == "__main__":
    main()
