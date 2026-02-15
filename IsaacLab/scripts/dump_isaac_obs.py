"""Dump Isaac Lab observations for comparison with MuJoCo deployment.
Saves the first few frames of obs, joint_pos, joint_vel, actions, etc.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump Isaac obs for sim2sim debug")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--forward_vel", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Isaac-Velocity-Flat-V4-Quadruped-Play-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable randomization
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.observations.policy.enable_corruption = False

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make("Isaac-Velocity-Flat-V4-Quadruped-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()

    all_data = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            # Override velocity command
            cmd_manager = env.unwrapped.command_manager
            cmd = cmd_manager.get_command("base_velocity")
            cmd[:, 0] = args_cli.forward_vel
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0

            actions = policy(obs)
            
            # Record BEFORE stepping
            robot = env.unwrapped.scene["robot"]
            
            frame_data = {
                'step': step,
                'obs': obs[0].detach().cpu().numpy().copy(),
                'actions': actions[0].detach().cpu().numpy().copy(),
                'root_pos_w': robot.data.root_pos_w[0].detach().cpu().numpy().copy(),
                'root_quat_w': robot.data.root_quat_w[0].detach().cpu().numpy().copy(),
                'root_lin_vel_b': robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy(),
                'root_ang_vel_b': robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy(),
                'projected_gravity_b': robot.data.projected_gravity_b[0].detach().cpu().numpy().copy(),
                'joint_pos': robot.data.joint_pos[0].detach().cpu().numpy().copy(),
                'joint_vel': robot.data.joint_vel[0].detach().cpu().numpy().copy(),
                'default_joint_pos': robot.data.default_joint_pos[0].detach().cpu().numpy().copy(),
                'joint_names': [robot.joint_names[i] for i in range(len(robot.joint_names))],
            }
            all_data.append(frame_data)
            
            if step < 5 or step % 10 == 0:
                o = frame_data['obs']
                print(f"\n[Step {step}]")
                print(f"  obs[0:3]  ang_vel:  {o[0:3]}")
                print(f"  obs[3:6]  gravity:  {o[3:6]}")
                print(f"  obs[6:9]  cmd:      {o[6:9]}")
                print(f"  obs[9:22] joint_pos:{o[9:22]}")
                print(f"  obs[22:35]joint_vel:{o[22:35]}")
                print(f"  obs[35:47]last_act: {o[35:47]}")
                print(f"  root_quat_w (w,x,y,z): {frame_data['root_quat_w']}")
                print(f"  root_ang_vel_b: {frame_data['root_ang_vel_b']}")
                print(f"  projected_gravity_b: {frame_data['projected_gravity_b']}")
                print(f"  joint_pos: {frame_data['joint_pos']}")
                print(f"  default_joint_pos: {frame_data['default_joint_pos']}")
                print(f"  joint_names: {frame_data['joint_names']}")

            obs, _, _, _ = env.step(actions)

    # Save
    save_path = os.path.join(os.path.dirname(resume_path), "isaac_obs_dump.npz")
    np.savez(save_path, data=all_data)
    print(f"\nSaved {len(all_data)} frames to: {save_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
