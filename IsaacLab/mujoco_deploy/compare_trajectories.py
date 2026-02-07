#!/usr/bin/env python3
"""Compare IsaacLab trajectory with MuJoCo replay results."""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_trajectory(filepath):
    """Load trajectory from HDF5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key == 'joint_names':
                data[key] = [name.decode('utf-8') if isinstance(name, bytes) else name
                            for name in f[key][:]]
            else:
                data[key] = f[key][:]
        for attr in f.attrs:
            data[attr] = f.attrs[attr]
    return data


def plot_comparison(isaac_traj, mujoco_traj=None, output_dir="trajectory_plots"):
    """Plot trajectory data for analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    time = isaac_traj['time']

    # Plot base position
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Base Position')
    labels = ['X', 'Y', 'Z']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, isaac_traj['base_pos'][:, i], label='IsaacLab', linewidth=2)
        if mujoco_traj:
            ax.plot(mujoco_traj['time'], mujoco_traj['base_pos'][:, i],
                   label='MuJoCo', linestyle='--', linewidth=2)
        ax.set_ylabel(f'{label} (m)')
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'base_position.png', dpi=150)
    plt.close()

    # Plot base velocity
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Base Linear Velocity (Body Frame)')
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, isaac_traj['base_lin_vel'][:, i], label='IsaacLab', linewidth=2)
        if mujoco_traj:
            ax.plot(mujoco_traj['time'], mujoco_traj['base_lin_vel'][:, i],
                   label='MuJoCo', linestyle='--', linewidth=2)
        ax.set_ylabel(f'{label} (m/s)')
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'base_velocity.png', dpi=150)
    plt.close()

    # Plot joint positions
    num_joints = isaac_traj['joint_pos'].shape[1]
    joint_names = isaac_traj.get('joint_names', [f'Joint_{i}' for i in range(num_joints)])

    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2*num_joints))
    fig.suptitle('Joint Positions')
    if num_joints == 1:
        axes = [axes]

    for i, (ax, jname) in enumerate(zip(axes, joint_names)):
        ax.plot(time, isaac_traj['joint_pos'][:, i], label='IsaacLab', linewidth=2)
        if mujoco_traj:
            ax.plot(mujoco_traj['time'], mujoco_traj['joint_pos'][:, i],
                   label='MuJoCo', linestyle='--', linewidth=2)
        ax.set_ylabel(f'{jname} (rad)')
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'joint_positions.png', dpi=150)
    plt.close()

    # Plot actions
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2*num_joints))
    fig.suptitle('Actions (Policy Output)')
    if num_joints == 1:
        axes = [axes]

    for i, (ax, jname) in enumerate(zip(axes, joint_names)):
        ax.plot(time, isaac_traj['actions'][:, i], linewidth=2)
        ax.set_ylabel(f'{jname}')
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'actions.png', dpi=150)
    plt.close()

    print(f"Plots saved to: {output_dir}")

    # Print statistics
    print("\n=== Trajectory Statistics ===")
    print(f"Duration: {time[-1]:.2f}s")
    print(f"Steps: {len(time)}")
    print(f"Base height range: [{isaac_traj['base_pos'][:, 2].min():.3f}, "
          f"{isaac_traj['base_pos'][:, 2].max():.3f}] m")
    print(f"Base height mean: {isaac_traj['base_pos'][:, 2].mean():.3f} m")
    print(f"Base height std: {isaac_traj['base_pos'][:, 2].std():.3f} m")

    if mujoco_traj:
        print("\n=== Comparison with MuJoCo ===")
        pos_error = np.abs(isaac_traj['base_pos'] - mujoco_traj['base_pos'])
        print(f"Base position error (mean): {pos_error.mean(axis=0)}")
        print(f"Base position error (max): {pos_error.max(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="Compare and visualize trajectories")
    parser.add_argument("isaac_trajectory", type=str, help="IsaacLab trajectory HDF5 file")
    parser.add_argument("--mujoco_trajectory", type=str, default=None,
                       help="MuJoCo trajectory HDF5 file (optional)")
    parser.add_argument("--output_dir", type=str, default="trajectory_plots",
                       help="Output directory for plots")
    args = parser.parse_args()

    print(f"Loading IsaacLab trajectory: {args.isaac_trajectory}")
    isaac_traj = load_trajectory(args.isaac_trajectory)

    mujoco_traj = None
    if args.mujoco_trajectory:
        print(f"Loading MuJoCo trajectory: {args.mujoco_trajectory}")
        mujoco_traj = load_trajectory(args.mujoco_trajectory)

    plot_comparison(isaac_traj, mujoco_traj, args.output_dir)


if __name__ == "__main__":
    main()
