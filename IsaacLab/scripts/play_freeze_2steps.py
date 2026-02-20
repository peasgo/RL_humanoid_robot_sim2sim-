"""Play V6 humanoid for exactly 2 policy steps then FREEZE.
Keeps the IsaacLab viewer open so you can inspect the pose.

Usage:
  isaaclab -p IsaacLab/scripts/play_freeze_2steps.py --task Isaac-Velocity-Flat-V6-Humanoid-v0 --num_envs 1 --forward_vel 0.0 --lateral_vel -0.5
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play V6 humanoid - freeze after 2 steps.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-V6-Humanoid-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--forward_vel", type=float, default=0.0)
parser.add_argument("--lateral_vel", type=float, default=-0.5)
parser.add_argument("--yaw_vel", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=2, help="Number of policy steps before freezing")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.forward_vel, args_cli.forward_vel)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.lateral_vel, args_cli.lateral_vel)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.yaw_vel, args_cli.yaw_vel)

    if hasattr(env_cfg, "events"):
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False

    # Resolve checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        possible_log_roots = [
            os.path.join("logs", "rsl_rl", agent_cfg.experiment_name),
            os.path.join("IsaacLab", "logs", "rsl_rl", agent_cfg.experiment_name),
            os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", agent_cfg.experiment_name),
        ]
        resume_path = None
        for log_root_path in possible_log_roots:
            log_root_path = os.path.abspath(log_root_path)
            if os.path.isdir(log_root_path):
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                break
        if resume_path is None:
            print(f"[ERROR] No log directory found. Use --checkpoint /path/to/model.pt")
            return

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names

    max_steps = args_cli.max_steps

    print(f"\n{'='*60}")
    print(f"  Will run {max_steps} policy steps then FREEZE")
    print(f"  Command: vx={args_cli.forward_vel:+.2f}  vy={args_cli.lateral_vel:+.2f}  wz={args_cli.yaw_vel:+.2f}")
    print(f"  Joint names: {joint_names}")
    print(f"{'='*60}\n")

    def print_state(step_idx, obs_tensor, actions_tensor=None):
        obs_np = obs_tensor[0].cpu().numpy()
        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        joint_vel = robot.data.joint_vel[0].cpu().numpy()
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        root_quat = robot.data.root_quat_w[0].cpu().numpy()

        print(f"\n{'='*60}")
        print(f"  STEP {step_idx}")
        print(f"{'='*60}")

        print(f"\n  [Observation] shape={obs_np.shape}")
        print(f"  obs = {np.array2string(obs_np, precision=4, separator=', ', max_line_width=120)}")

        print(f"\n  [Joint Positions (rad)]")
        for i, name in enumerate(joint_names):
            print(f"    [{i:2d}] {name:30s}  pos={joint_pos[i]:+.6f}  vel={joint_vel[i]:+.6f}")

        if actions_tensor is not None:
            act_np = actions_tensor[0].cpu().numpy()
            print(f"\n  [Actions sent at step {step_idx}] (Isaac BFS order)")
            for i, name in enumerate(joint_names):
                print(f"    [{i:2d}] {name:30s}  action={act_np[i]:+.6f}")

        print(f"\n  [Base] pos=({root_pos[0]:+.4f}, {root_pos[1]:+.4f}, {root_pos[2]:.4f})  "
              f"quat=({root_quat[0]:.4f}, {root_quat[1]:.4f}, {root_quat[2]:.4f}, {root_quat[3]:.4f})")

    # Print Step 0 (before any action)
    print_state(0, obs)

    # Run exactly max_steps
    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            print_state(step, obs, actions)
            obs, _, _, _ = env.step(actions)

    # Print final state after last step
    print_state(max_steps, obs)

    print(f"\n{'='*60}")
    print(f"  FROZEN after {max_steps} steps. Viewer stays open.")
    print(f"  Close the IsaacLab window to exit.")
    print(f"{'='*60}\n")

    # Keep viewer alive without stepping physics
    while simulation_app.is_running():
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
