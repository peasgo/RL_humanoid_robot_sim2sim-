"""Play V6 humanoid with FIXED forward-only command.
Override the velocity command to pure forward (lin_vel_x=0.5, lin_vel_y=0, ang_vel_z=0).
Prints actual vs commanded velocity each second for diagnosis.

Usage:
  isaaclab -p IsaacLab/scripts/play_forward_only.py --task Isaac-Velocity-Flat-V6-Humanoid-v0 --num_envs 1
"""

import argparse
import sys
import os

# --- CLI ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play V6 humanoid with forward-only command.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-V6-Humanoid-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--forward_vel", type=float, default=0.5, help="Forward velocity command (m/s)")
parser.add_argument("--lateral_vel", type=float, default=0.0, help="Lateral velocity command (m/s)")
parser.add_argument("--yaw_vel", type=float, default=0.0, help="Yaw angular velocity command (rad/s)")
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
    """Play with fixed forward command."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Force command ranges to our fixed values (tiny range so it's effectively constant)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.forward_vel, args_cli.forward_vel)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.lateral_vel, args_cli.lateral_vel)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.yaw_vel, args_cli.yaw_vel)

    # Try multiple possible log locations
    possible_log_roots = [
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name),
        os.path.join("IsaacLab", "logs", "rsl_rl", agent_cfg.experiment_name),
        os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", agent_cfg.experiment_name),
    ]

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = None
        for log_root_path in possible_log_roots:
            log_root_path = os.path.abspath(log_root_path)
            if os.path.isdir(log_root_path):
                print(f"[INFO] Found log directory: {log_root_path}")
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                break
        if resume_path is None:
            print(f"[ERROR] No log directory found. Tried: {[os.path.abspath(p) for p in possible_log_roots]}")
            print(f"[ERROR] Use --checkpoint /path/to/model.pt to specify directly.")
            return

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    # Print robot info
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(f"\n{'='*60}")
    print(f"Fixed command: lin_vel_x={args_cli.forward_vel}, lin_vel_y={args_cli.lateral_vel}, ang_vel_z={args_cli.yaw_vel}")
    print(f"Root quaternion (w,x,y,z): {np.round(root_quat, 4)}")
    print(f"Joint names: {joint_names}")
    print(f"{'='*60}\n")

    def print_step_details(step_idx, obs_tensor, robot_ref):
        """Print observation and joint angles for a given step."""
        obs_np = obs_tensor[0].cpu().numpy()
        joint_pos = robot_ref.data.joint_pos[0].cpu().numpy()
        joint_vel = robot_ref.data.joint_vel[0].cpu().numpy()

        print(f"\n{'='*60}")
        print(f"  STEP {step_idx}")
        print(f"{'='*60}")

        # Observation vector
        print(f"\n  [Observation] shape={obs_np.shape}")
        print(f"  obs = {np.array2string(obs_np, precision=4, separator=', ', max_line_width=120)}")

        # Joint positions (angles)
        print(f"\n  [Joint Positions (rad)]")
        for i, name in enumerate(joint_names):
            print(f"    [{i:2d}] {name:30s}  pos={joint_pos[i]:+.6f}  vel={joint_vel[i]:+.6f}")
        print()

    # --- Step 0: before any action ---
    print_step_details(0, obs, robot)

    step_count = 0
    print_interval = int(1.0 / dt)  # print every ~1 second

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            step_count += 1

            # --- Step 1: after first action ---
            if step_count == 1:
                robot = env.unwrapped.scene["robot"]
                act_np = actions[0].cpu().numpy()
                print(f"\n  [Actions sent at step 0]")
                for i, name in enumerate(joint_names):
                    print(f"    [{i:2d}] {name:30s}  action={act_np[i]:+.6f}")
                print_step_details(1, obs, robot)

            if step_count % print_interval == 0:
                robot = env.unwrapped.scene["robot"]
                # World frame velocity
                lin_vel_w = robot.data.root_lin_vel_w[0].cpu().numpy()
                ang_vel_w = robot.data.root_ang_vel_w[0].cpu().numpy()
                # Body frame velocity (heading-aligned)
                root_quat_t = robot.data.root_quat_w  # (N, 4)
                lin_vel_w_t = robot.data.root_lin_vel_w  # (N, 3)
                lin_vel_body = quat_apply_yaw(root_quat_t, lin_vel_w_t)[0].cpu().numpy()
                # Position
                pos = robot.data.root_pos_w[0].cpu().numpy()

                # Get the actual command being sent
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy()

                t = step_count * dt
                print(f"[t={t:6.1f}s] pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f})")
                print(f"  cmd:  vx={cmd[0]:+.3f}  vy={cmd[1]:+.3f}  wz={cmd[2]:+.3f}")
                print(f"  body: vx={lin_vel_body[0]:+.3f}  vy={lin_vel_body[1]:+.3f}")
                print(f"  world: vx={lin_vel_w[0]:+.3f}  vy={lin_vel_w[1]:+.3f}  vz={lin_vel_w[2]:+.3f}")
                print()

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
