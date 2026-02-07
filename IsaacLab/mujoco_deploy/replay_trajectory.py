#!/usr/bin/env python3
"""Replay IsaacLab trajectory in MuJoCo."""

import argparse
import h5py
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def quaternion_to_mujoco(quat_wxyz):
    """Convert quaternion from (w,x,y,z) to MuJoCo format (w,x,y,z)."""
    return quat_wxyz


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
    parser = argparse.ArgumentParser(description="Replay IsaacLab trajectory in MuJoCo")
    parser.add_argument("trajectory", type=str, help="Path to trajectory HDF5 file")
    parser.add_argument("--xml", type=str, default="scene.xml", help="MuJoCo scene XML file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--mode", type=str, default="position", choices=["position", "velocity"],
                        help="Replay mode: position (set qpos/qvel) or velocity (use recorded velocities)")
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from: {args.trajectory}")
    traj = load_trajectory(args.trajectory)
    print(f"Loaded {traj['num_steps']} steps, dt={traj['dt']:.4f}s")
    print(f"Joint names: {traj['joint_names']}")

    # Load MuJoCo model
    xml_path = Path(args.xml)
    if not xml_path.exists():
        xml_path = Path(__file__).parent / args.xml

    print(f"Loading MuJoCo model from: {xml_path}")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    # Get joint mapping
    joint_indices = []
    for jname in traj['joint_names']:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            print(f"Warning: Joint '{jname}' not found in MuJoCo model")
        else:
            joint_indices.append(jid)

    print(f"Mapped {len(joint_indices)}/{len(traj['joint_names'])} joints")

    # Replay trajectory
    print(f"\nReplaying trajectory in {args.mode} mode...")
    print("Press ESC to exit")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        start_time = time.time()

        while viewer.is_running() and step < traj['num_steps']:
            step_start = time.time()

            if args.mode == "position":
                # Set positions and velocities directly
                d.qpos[0:3] = traj['base_pos'][step]
                d.qpos[3:7] = traj['base_quat'][step]

                # Set joint positions (skip freejoint, starts at index 7)
                for i, jid in enumerate(joint_indices):
                    qpos_addr = m.jnt_qposadr[jid]
                    d.qpos[qpos_addr] = traj['joint_pos'][step][i]

                # Set velocities
                d.qvel[0:3] = traj['base_lin_vel'][step]
                d.qvel[3:6] = traj['base_ang_vel'][step]

                for i, jid in enumerate(joint_indices):
                    qvel_addr = m.jnt_dofadr[jid]
                    d.qvel[qvel_addr] = traj['joint_vel'][step][i]

                # Forward kinematics
                mujoco.mj_forward(m, d)

            else:  # velocity mode
                # Use recorded velocities to update state
                d.qvel[0:3] = traj['base_lin_vel'][step]
                d.qvel[3:6] = traj['base_ang_vel'][step]

                for i, jid in enumerate(joint_indices):
                    qvel_addr = m.jnt_dofadr[jid]
                    d.qvel[qvel_addr] = traj['joint_vel'][step][i]

                # Step simulation
                mujoco.mj_step(m, d)

            # Update viewer
            viewer.sync()

            # Print info
            if step % 50 == 0:
                print(f"Step {step}/{traj['num_steps']} | "
                      f"Time: {traj['time'][step]:.2f}s | "
                      f"Base height: {d.qpos[2]:.3f}m")

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
