"""Test V4 quadruped walking straight with fixed forward velocity command.
Records position trajectory to check if the robot walks in a straight line.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test V4 walk straight")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
parser.add_argument("--forward_vel", type=float, default=0.5, help="Forward velocity command")
parser.add_argument("--lateral_vel", type=float, default=0.0, help="Lateral velocity command")
parser.add_argument("--yaw_vel", type=float, default=0.0, help="Yaw angular velocity command")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Isaac-Velocity-Flat-V4-Quadruped-Play-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable randomization for clean test
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

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    # Storage for trajectory
    positions = []
    orientations = []
    joint_positions = []
    actions_history = []
    velocities = []

    forward_vel = args_cli.forward_vel
    lateral_vel = args_cli.lateral_vel
    yaw_vel = args_cli.yaw_vel
    print(f"\n{'='*60}")
    print(f"Testing V4 walk: fwd={forward_vel}, lat={lateral_vel}, yaw={yaw_vel}")
    print(f"Steps: {args_cli.steps}, dt: {dt}")
    print(f"{'='*60}\n")

    for step in range(args_cli.steps):
        with torch.inference_mode():
            # Override velocity command
            cmd_manager = env.unwrapped.command_manager
            cmd = cmd_manager.get_command("base_velocity")
            cmd[:, 0] = forward_vel
            cmd[:, 1] = lateral_vel
            cmd[:, 2] = yaw_vel

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # Record data
            robot = env.unwrapped.scene["robot"]
            pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
            quat = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
            jpos = robot.data.joint_pos[0].detach().cpu().numpy().copy()
            vel = robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy()
            act = actions[0].detach().cpu().numpy().copy()

            positions.append(pos)
            orientations.append(quat)
            joint_positions.append(jpos)
            actions_history.append(act)
            velocities.append(vel)

            if step % 50 == 0:
                print(f"[Step {step:4d}] pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}) "
                      f"vel_body=({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}) "
                      f"act_max={np.max(np.abs(act)):.3f}")

    positions = np.array(positions)
    orientations = np.array(orientations)
    joint_positions = np.array(joint_positions)
    actions_history = np.array(actions_history)
    velocities = np.array(velocities)

    # Analysis
    print(f"\n{'='*60}")
    print("TRAJECTORY ANALYSIS")
    print(f"{'='*60}")

    # 1. Straight line check
    start_pos = positions[0, :2]
    end_pos = positions[-1, :2]
    travel_vec = end_pos - start_pos
    travel_dist = np.linalg.norm(travel_vec)
    lateral_drift = np.max(np.abs(positions[:, 1] - positions[0, 1]))
    print(f"\nTravel distance: {travel_dist:.3f} m")
    print(f"Start: ({start_pos[0]:.3f}, {start_pos[1]:.3f})")
    print(f"End:   ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    print(f"Max lateral drift (Y): {lateral_drift:.4f} m")
    print(f"Straightness ratio: {lateral_drift / max(travel_dist, 0.001):.4f} (lower=straighter)")

    # 2. Height stability
    heights = positions[:, 2]
    print(f"\nHeight: mean={np.mean(heights):.4f}, std={np.std(heights):.4f}, "
          f"range=[{np.min(heights):.4f}, {np.max(heights):.4f}]")

    # 3. Action smoothness / jitter analysis
    action_diffs = np.diff(actions_history, axis=0)
    print(f"\nAction smoothness:")
    print(f"  Mean |action_diff|: {np.mean(np.abs(action_diffs)):.6f}")
    print(f"  Max  |action_diff|: {np.max(np.abs(action_diffs)):.6f}")
    print(f"  Std of actions: {np.mean(np.std(actions_history, axis=0)):.4f}")

    # Per-joint action jitter
    print(f"\nPer-joint action jitter (std of consecutive diffs):")
    joint_names_action = [
        "RSDp", "RSDy", "RARMp",
        "LSDp", "LSDy", "LARMp",
        "RHIPp", "RHIPy", "RKNEEP",
        "LHIPp", "LHIPy", "LKNEEp",
    ]
    for j in range(min(len(joint_names_action), actions_history.shape[1])):
        jdiff = action_diffs[:, j]
        sign_changes = np.sum(np.diff(np.sign(jdiff)) != 0)
        print(f"  {joint_names_action[j]:10s}: diff_std={np.std(jdiff):.5f}, "
              f"sign_changes={sign_changes}/{len(jdiff)-1} "
              f"({'HIGH JITTER' if sign_changes > 0.7 * (len(jdiff)-1) else 'ok'})")

    # 4. Joint velocity analysis (from obs)
    print(f"\nJoint position oscillation (std of joint_pos over last 200 steps):")
    last_jpos = joint_positions[-200:]
    for j in range(min(13, joint_positions.shape[1])):
        std = np.std(last_jpos[:, j])
        print(f"  Joint {j:2d}: std={std:.5f}")

    # Save data
    save_path = os.path.join(os.path.dirname(resume_path), "walk_straight_test.npz")
    np.savez(save_path,
             positions=positions,
             orientations=orientations,
             joint_positions=joint_positions,
             actions=actions_history,
             velocities=velocities)
    print(f"\nData saved to: {save_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
