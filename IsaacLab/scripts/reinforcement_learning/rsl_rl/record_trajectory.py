#!/usr/bin/env python3
"""Record trajectory from IsaacLab for MuJoCo replay."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record trajectory from IsaacLab.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (use 1 for recording).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps to record")
parser.add_argument("--output", type=str, default="trajectory.h5", help="Output trajectory file")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")

import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import h5py
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnv, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1  # Force single environment
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Load checkpoint
    resume_path = get_checkpoint_path(
        agent_cfg.load_run, agent_cfg.load_checkpoint, train_task_name
    )
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    # Wrap environment
    env = RslRlVecEnvWrapper(env)

    # Load policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Prepare data storage
    trajectory_data = {
        'time': [],
        'base_pos': [],
        'base_quat': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'joint_pos': [],
        'joint_vel': [],
        'actions': [],
        'observations': [],
        'default_joint_pos': None,
        'joint_names': None,
    }

    # Get joint names and defaults
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()

    trajectory_data['joint_names'] = joint_names
    trajectory_data['default_joint_pos'] = default_joint_pos

    print(f"[INFO] Recording trajectory for {args_cli.max_steps} steps...")
    print(f"[INFO] Joint names: {joint_names}")
    print(f"[INFO] Default joint positions: {default_joint_pos}")

    # Reset environment
    obs, _ = env.get_observations()

    # Run simulation
    for step in range(args_cli.max_steps):
        # Get action from policy
        with torch.no_grad():
            actions = policy(obs)

        # Step environment
        obs, _, _, _, _ = env.step(actions)

        # Record data (only first environment)
        robot_data = robot.data
        trajectory_data['time'].append(step * env_cfg.sim.dt * env_cfg.decimation)
        trajectory_data['base_pos'].append(robot_data.root_pos_w[0].cpu().numpy())
        trajectory_data['base_quat'].append(robot_data.root_quat_w[0].cpu().numpy())
        trajectory_data['base_lin_vel'].append(robot_data.root_lin_vel_b[0].cpu().numpy())
        trajectory_data['base_ang_vel'].append(robot_data.root_ang_vel_b[0].cpu().numpy())
        trajectory_data['joint_pos'].append(robot_data.joint_pos[0].cpu().numpy())
        trajectory_data['joint_vel'].append(robot_data.joint_vel[0].cpu().numpy())
        trajectory_data['actions'].append(actions[0].cpu().numpy())
        trajectory_data['observations'].append(obs[0].cpu().numpy())

        if step % 50 == 0:
            print(f"[INFO] Step {step}/{args_cli.max_steps}")

    # Save trajectory
    output_path = Path(args_cli.output)
    print(f"[INFO] Saving trajectory to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('time', data=np.array(trajectory_data['time']))
        f.create_dataset('base_pos', data=np.array(trajectory_data['base_pos']))
        f.create_dataset('base_quat', data=np.array(trajectory_data['base_quat']))
        f.create_dataset('base_lin_vel', data=np.array(trajectory_data['base_lin_vel']))
        f.create_dataset('base_ang_vel', data=np.array(trajectory_data['base_ang_vel']))
        f.create_dataset('joint_pos', data=np.array(trajectory_data['joint_pos']))
        f.create_dataset('joint_vel', data=np.array(trajectory_data['joint_vel']))
        f.create_dataset('actions', data=np.array(trajectory_data['actions']))
        f.create_dataset('observations', data=np.array(trajectory_data['observations']))
        f.create_dataset('default_joint_pos', data=default_joint_pos)

        # Store joint names as strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('joint_names', data=np.array(joint_names, dtype=dt))

        # Store metadata
        f.attrs['num_steps'] = len(trajectory_data['time'])
        f.attrs['dt'] = env_cfg.sim.dt * env_cfg.decimation
        f.attrs['task_name'] = train_task_name

    print(f"[INFO] Trajectory saved successfully!")
    print(f"[INFO] Total steps: {len(trajectory_data['time'])}")
    print(f"[INFO] Duration: {trajectory_data['time'][-1]:.2f}s")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
