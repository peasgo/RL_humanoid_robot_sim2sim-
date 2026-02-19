"""Play V6 humanoid with FIXED velocity command (forward/backward/lateral).

Usage:
  # 前进 0.5 m/s
  isaaclab -p IsaacLab/scripts/play_forward_backward.py --task Isaac-Velocity-Flat-V6-Humanoid-v0 --num_envs 1 --forward_vel 0.5

  # 后退 0.5 m/s
  isaaclab -p IsaacLab/scripts/play_forward_backward.py --task Isaac-Velocity-Flat-V6-Humanoid-v0 --num_envs 1 --forward_vel -0.5

  # 侧移 0.3 m/s
  isaaclab -p IsaacLab/scripts/play_forward_backward.py --task Isaac-Velocity-Flat-V6-Humanoid-v0 --num_envs 1 --forward_vel 0.0 --lateral_vel 0.3

  # 指定checkpoint
  isaaclab -p IsaacLab/scripts/play_forward_backward.py --task Isaac-Velocity-Flat-V6-Humanoid-v0 --num_envs 1 --forward_vel 0.5 \
    --checkpoint /home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-19_11-55-18/model_9000.pt
"""

import argparse
import sys
import os

# --- CLI ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play V6 humanoid with fixed velocity command.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-V6-Humanoid-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--forward_vel", type=float, default=0.5, help="Forward(+)/backward(-) velocity (m/s)")
parser.add_argument("--lateral_vel", type=float, default=0.0, help="Left(+)/right(-) lateral velocity (m/s)")
parser.add_argument("--yaw_vel", type=float, default=0.0, help="Yaw angular velocity (rad/s)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after sim launch ---
import gymnasium as gym
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_yaw
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Fix command ranges to the requested velocity
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.forward_vel, args_cli.forward_vel)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.lateral_vel, args_cli.lateral_vel)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.yaw_vel, args_cli.yaw_vel)

    # Disable randomization for clean playback
    if hasattr(env_cfg, "events"):
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None

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
                print(f"[INFO] Found log directory: {log_root_path}")
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                break
        if resume_path is None:
            print(f"[ERROR] No log directory found. Use --checkpoint /path/to/model.pt")
            return

    # Create env & load policy
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    print(f"\n{'='*60}")
    print(f"  Command: vx={args_cli.forward_vel:+.2f}  vy={args_cli.lateral_vel:+.2f}  wz={args_cli.yaw_vel:+.2f}")
    print(f"{'='*60}\n")

    step_count = 0
    print_interval = int(1.0 / dt)

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            step_count += 1
            if step_count % print_interval == 0:
                robot = env.unwrapped.scene["robot"]
                root_quat_t = robot.data.root_quat_w
                lin_vel_w_t = robot.data.root_lin_vel_w
                lin_vel_body = quat_apply_yaw(root_quat_t, lin_vel_w_t)[0].cpu().numpy()
                pos = robot.data.root_pos_w[0].cpu().numpy()
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy()

                t = step_count * dt
                print(
                    f"[t={t:6.1f}s] "
                    f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}) | "
                    f"body_vel=({lin_vel_body[0]:+.3f}, {lin_vel_body[1]:+.3f}) | "
                    f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f})"
                )

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
