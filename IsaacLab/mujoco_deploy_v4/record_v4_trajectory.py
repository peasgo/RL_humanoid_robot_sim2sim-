# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V4 Quadruped - Record trajectory from IsaacLab for MuJoCo replay.
Records joint positions, velocities, base state, observations, and actions.

Usage:
    cd /home/rl/RL-human_robot/IsaacLab/mujoco_deploy_v4
    conda run --live-stream --name isaaclab python record_v4_trajectory.py \
        --task Isaac-Velocity-Flat-V4-Quadruped-Play-v0 \
        --num_envs 1 \
        --headless \
        --max_steps 800 \
        --checkpoint /path/to/model.pt
"""

import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

# local imports
sys.path.insert(0, "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy")
import cli_args  # isort: skip
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Record V4 Quadruped trajectory from IsaacLab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=800, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--output", type=str, default="v4_trajectory.h5", help="Output HDF5 file path.")
parser.add_argument("--max_steps", type=int, default=800, help="Maximum number of steps to record.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# required args sanity check
if args_cli.task is None:
    raise SystemExit(
        "[ERROR] --task is required. Example: --task Isaac-Velocity-Flat-V4-Quadruped-Play-v0"
    )
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import h5py
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def save_trajectory(filepath, data_groups, joint_names, default_joint_pos, sim_dt, decimation, num_steps):
    """Save recorded trajectory to HDF5 file."""
    print(f"\n{'='*60}")
    print(f"Saving trajectory to {filepath}...")
    print(f"  Steps recorded: {num_steps}")
    print(f"  dt: {sim_dt * decimation:.4f}s")
    print(f"  Joint names: {joint_names}")
    print(f"{'='*60}")

    with h5py.File(filepath, 'w') as f_h5:
        for k, v in data_groups.items():
            if len(v) > 0:
                f_h5.create_dataset(k, data=np.array(v))
                print(f"  Dataset '{k}': shape={np.array(v).shape}")

        f_h5.attrs['dt'] = sim_dt * decimation
        f_h5.attrs['num_steps'] = num_steps

        # Store joint names
        dt_str = h5py.string_dtype(encoding='utf-8')
        dset = f_h5.create_dataset('joint_names', (len(joint_names),), dtype=dt_str)
        dset[:] = joint_names

        # Store default joint positions
        if default_joint_pos is not None:
            f_h5.create_dataset('default_joint_pos', data=default_joint_pos)

    print(f"Trajectory saved successfully to {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")

    # Print joint order summary
    print(f"\nIsaacLab Joint Order (CRITICAL for sim2sim):")
    for i, name in enumerate(joint_names):
        default_val = default_joint_pos[i] if default_joint_pos is not None else 0.0
        print(f"  [{i:2d}] {name:15s} (default={default_val:.4f})")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Record V4 Quadruped trajectory."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("IsaacLab", "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    recorded_steps = 0
    max_steps = args_cli.max_steps

    # output file
    recording_file = args_cli.output
    if os.path.exists(recording_file):
        os.remove(recording_file)
        print(f"[INFO] Removed existing file: {recording_file}")

    data_groups = {
        'time': [],
        'observations': [],
        'actions': [],
        'joint_pos': [],
        'joint_vel': [],
        'base_pos': [],
        'base_quat': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
    }

    joint_names = None
    default_joint_pos = None

    try:
        # create environment
        print(f"[INFO] Creating environment: {args_cli.task}")
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        print(f"[INFO] Environment created successfully!")

        # load checkpoint
        if args_cli.checkpoint:
            resume_path = args_cli.checkpoint
        else:
            resume_path = get_checkpoint_path(log_root_path, "model", agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        log_dir = log_root_path

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner and load policy
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        print(f"[INFO] Model loaded successfully!")

        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # export policy
        try:
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            policy_nn = ppo_runner.alg.actor_critic

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        print(f"[INFO] Policy exported to: {export_model_dir}/policy.pt")

        dt = env.unwrapped.step_dt

        # reset environment
        obs, _ = env.get_observations()
        print(f"[INFO] Environment reset. Observation shape: {obs.shape}")

        timestep = 0

        # Print joint names at start
        robot = env.unwrapped.scene["robot"]
        joint_names = list(robot.joint_names)
        default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy().copy()

        print(f"\n{'='*60}")
        print(f"V4 Quadruped IsaacLab Joint Names ({len(joint_names)} joints):")
        for i, name in enumerate(joint_names):
            print(f"  [{i:2d}] {name:15s} (default={default_joint_pos[i]:.4f})")
        print(f"{'='*60}\n")

        sim_dt = env_cfg.sim.dt
        decimation = getattr(env_cfg, "decimation", 1)
        print(f"[INFO] sim_dt={sim_dt}, decimation={decimation}, step_dt={sim_dt*decimation}")
        print(f"[INFO] Starting recording... max_steps={max_steps}")

        # simulate and record
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                if recorded_steps < max_steps:
                    robot = env.unwrapped.scene["robot"]

                    base_pos = robot.data.root_pos_w[0].cpu().numpy()
                    base_quat = robot.data.root_quat_w[0].cpu().numpy()

                    lin_vel_w = robot.data.root_lin_vel_w[0]
                    ang_vel_b = robot.data.root_ang_vel_b[0]

                    base_lin_vel = lin_vel_w.cpu().numpy()
                    base_ang_vel = ang_vel_b.cpu().numpy()

                    joint_pos = robot.data.joint_pos[0].cpu().numpy()
                    joint_vel = robot.data.joint_vel[0].cpu().numpy()

                    action_np = actions[0].cpu().numpy()
                    obs_np = obs[0].cpu().numpy()

                    data_groups['time'].append(timestep * sim_dt * decimation)
                    data_groups['base_pos'].append(base_pos.copy())
                    data_groups['base_quat'].append(base_quat.copy())
                    data_groups['base_lin_vel'].append(base_lin_vel.copy())
                    data_groups['base_ang_vel'].append(base_ang_vel.copy())
                    data_groups['joint_pos'].append(joint_pos.copy())
                    data_groups['joint_vel'].append(joint_vel.copy())
                    data_groups['actions'].append(action_np.copy())
                    data_groups['observations'].append(obs_np.copy())

                    recorded_steps += 1

                    if recorded_steps % 100 == 0:
                        print(f"[REC] Step {recorded_steps}/{max_steps} | "
                              f"Base height: {base_pos[2]:.3f}m | "
                              f"Joint pos[0:3]: [{joint_pos[0]:.3f}, {joint_pos[1]:.3f}, {joint_pos[2]:.3f}]")

                    if recorded_steps >= max_steps:
                        print(f"\n[INFO] Recording complete! {recorded_steps} steps recorded.")
                        save_trajectory(recording_file, data_groups, joint_names,
                                       default_joint_pos, sim_dt, decimation, recorded_steps)
                        break

            timestep += 1

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()

    except Exception as e:
        print(f"\n[ERROR] Exception during recording: {e}")
        traceback.print_exc()

        # Save whatever we have so far
        if recorded_steps > 0 and joint_names is not None:
            print(f"\n[INFO] Saving partial trajectory ({recorded_steps} steps)...")
            save_trajectory(recording_file, data_groups, joint_names,
                           default_joint_pos, 
                           getattr(env_cfg, 'sim', type('', (), {'dt': 0.005})).dt,
                           getattr(env_cfg, 'decimation', 4),
                           recorded_steps)
        else:
            print(f"[WARNING] No data recorded. Cannot save trajectory.")
        raise


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
