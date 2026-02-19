"""Dump Isaac Lab observations for V6 humanoid - for comparison with MuJoCo deployment.
Run with:
  cd IsaacLab
  python scripts/dump_v6_isaac_obs.py --num_envs 1 --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump V6 humanoid Isaac obs for sim2sim debug")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--steps", type=int, default=20)
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


@hydra_task_config("Isaac-Velocity-Flat-V6-Humanoid-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable randomization
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.physics_material = None
    env_cfg.observations.policy.enable_corruption = False

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make("Isaac-Velocity-Flat-V6-Humanoid-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()

    for step in range(args_cli.steps):
        with torch.inference_mode():
            # Set zero velocity command
            cmd_manager = env.unwrapped.command_manager
            cmd = cmd_manager.get_command("base_velocity")
            cmd[:, 0] = 0.0
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0

            actions = policy(obs)

            robot = env.unwrapped.scene["robot"]

            o = obs[0].detach().cpu().numpy()
            a = actions[0].detach().cpu().numpy()
            joint_names = robot.joint_names

            print(f"\n{'='*70}")
            print(f"[Isaac Step {step}]")
            print(f"  root_pos_w:         {robot.data.root_pos_w[0].cpu().numpy()}")
            print(f"  root_quat_w (wxyz): {robot.data.root_quat_w[0].cpu().numpy()}")
            print(f"  root_ang_vel_b:     {robot.data.root_ang_vel_b[0].cpu().numpy()}")
            print(f"  projected_gravity_b:{robot.data.projected_gravity_b[0].cpu().numpy()}")
            print(f"  height:             {robot.data.root_pos_w[0, 2].item():.4f}m")
            print(f"  joint_names:        {joint_names}")

            # Per-joint state
            jpos = robot.data.joint_pos[0].cpu().numpy()
            jdef = robot.data.default_joint_pos[0].cpu().numpy()
            jvel = robot.data.joint_vel[0].cpu().numpy()
            print(f"\n  --- Joint state (Isaac order) ---")
            for i, jname in enumerate(joint_names):
                print(f"    [{i:2d}] {jname:14s}  pos={jpos[i]:+.6f}  def={jdef[i]:+.4f}"
                      f"  rel={jpos[i]-jdef[i]:+.6f}  vel={jvel[i]:+.6f}")

            # Per-element observation (matching compare_with_isaac.py format)
            print(f"\n  --- Full observation (48 dims, element by element) ---")
            obs_labels = []
            obs_labels.append(("obs[0]", "ang_vel_x * 0.2"))
            obs_labels.append(("obs[1]", "ang_vel_y * 0.2"))
            obs_labels.append(("obs[2]", "ang_vel_z * 0.2"))
            obs_labels.append(("obs[3]", "gravity_x"))
            obs_labels.append(("obs[4]", "gravity_y"))
            obs_labels.append(("obs[5]", "gravity_z"))
            obs_labels.append(("obs[6]", "cmd_vx"))
            obs_labels.append(("obs[7]", "cmd_vy"))
            obs_labels.append(("obs[8]", "cmd_wz"))
            for i, jname in enumerate(joint_names):
                obs_labels.append((f"obs[{9+i}]", f"pos_rel_{jname}"))
            for i, jname in enumerate(joint_names):
                obs_labels.append((f"obs[{22+i}]", f"vel_{jname}*0.05"))
            for i, jname in enumerate(joint_names):
                obs_labels.append((f"obs[{35+i}]", f"last_act_{jname}"))

            for idx, (label, desc) in enumerate(obs_labels):
                if idx < len(o):
                    print(f"    {label:8s} {desc:25s} = {o[idx]:+.8f}")

            # Action per joint
            print(f"\n  --- Action ({len(a)} dims) ---")
            for i, jname in enumerate(joint_names):
                if i < len(a):
                    target = a[i] * 0.25 + jdef[i]
                    print(f"    [{i:2d}] {jname:14s}  act={a[i]:+.6f}"
                          f"  target={target:+.6f}  (def={jdef[i]:+.4f})")

            obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
