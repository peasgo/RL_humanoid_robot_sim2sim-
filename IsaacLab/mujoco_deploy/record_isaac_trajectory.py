# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# required args sanity check (Hydra wrapper will fail with None)
if args_cli.task is None:
    raise SystemExit(
        "[ERROR] --task is required. Example: --task Isaac-<TaskName>-Play (use a task registered in IsaacLab)."
    )
# always enable cameras to record video
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

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("IsaacLab", "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Initialize recorded_steps
    recorded_steps = 0
    max_steps = args_cli.video_length

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # open HDF5 file
    recording_file = "h6.h5"
    if os.path.exists(recording_file):
        os.remove(recording_file)
    
    f_h5 = h5py.File(recording_file, 'w')
    
    # Define groups
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

    # load checkpoint
    # resume_path = get_checkpoint_path(log_root_path, "model", agent_cfg.load_checkpoint)
    resume_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-30_22-56-13/model_29999.pt"
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 3. Create RSL-RL runner
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()

    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # ------------------------------------------------------------------
            # RECORDING
            if recorded_steps < max_steps:
               
                # Get states directly from env.unwrapped.scene
                robot = env.unwrapped.scene["robot"]
                
                # Base Position & Quat (World Frame)
                # robot.data.root_pos_w -> [num_envs, 3]
                # robot.data.root_quat_w -> [num_envs, 4] (w, x, y, z)
                base_pos = robot.data.root_pos_w[0].cpu().numpy()
                base_quat = robot.data.root_quat_w[0].cpu().numpy() # scalar-first (w, x, y, z) usually in IsaacLab
                
                # Base Velocity
                # For MuJoCo Free Joint qvel:
                # qvel[0:3] -> Linear Velocity in WORLD Frame
                # qvel[3:6] -> Angular Velocity in BODY Frame
                
                # Use engine internal properties directly (Reliable)
                lin_vel_w = robot.data.root_lin_vel_w[0] # World Linear
                ang_vel_b = robot.data.root_ang_vel_b[0] # Body Angular

                base_lin_vel = lin_vel_w.cpu().numpy()
                base_ang_vel = ang_vel_b.cpu().numpy()
                
                # Joint States
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                joint_vel = robot.data.joint_vel[0].cpu().numpy()
                
                # Actions & Obs
                action_np = actions[0].cpu().numpy()
                obs_np = obs[0].cpu().numpy()
                
                sim_dt = env_cfg.sim.dt
                # If decimation is available in env_cfg use it, otherwise assume 1 or step_dt/sim_dt
                decimation = getattr(env_cfg, "decimation", 1)
                
                data_groups['time'].append(timestep * sim_dt * decimation)
                data_groups['base_pos'].append(base_pos)
                data_groups['base_quat'].append(base_quat)
                data_groups['base_lin_vel'].append(base_lin_vel)
                data_groups['base_ang_vel'].append(base_ang_vel)
                data_groups['joint_pos'].append(joint_pos)
                data_groups['joint_vel'].append(joint_vel)
                data_groups['actions'].append(action_np)
                data_groups['observations'].append(obs_np)
                
                recorded_steps += 1
                
                if recorded_steps == max_steps:
                    print(f"Recording complete! Saving to {recording_file}...")
                    for k, v in data_groups.items():
                        f_h5.create_dataset(k, data=np.array(v))
                    
                    # Attributes
                    f_h5.attrs['dt'] = sim_dt * decimation
                    f_h5.attrs['num_steps'] = max_steps
                    
                    # Store joint names
                    joint_names = robot.joint_names
                    # HDF5 string support
                    dt_str = h5py.string_dtype(encoding='utf-8')
                    dset = f_h5.create_dataset('joint_names', (len(joint_names),), dtype=dt_str)
                    dset[:] = joint_names

                    f_h5.close()
                    print("Trajectory saved.")
                    # break # Exit after recording
                    
            if recorded_steps % 100 == 0 and recorded_steps <= max_steps:
                 print(f"Simulating step {timestep} (Recorded: {recorded_steps}/{max_steps})")
            # ------------------------------------------------------------------

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1
            if recorded_steps >= max_steps:
                break
            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app even if main() errors
        simulation_app.close()
