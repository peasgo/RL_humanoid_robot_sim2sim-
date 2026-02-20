"""Dump Isaac Lab Step0 FULL STATE for V6 humanoid — for obs-construction verification.

Exports to .npz:
  root_pos_w        (3,)    — world position
  root_quat_w       (4,)    — quaternion wxyz
  root_lin_vel_w    (3,)    — world-frame linear velocity
  root_ang_vel_b    (3,)    — body-frame angular velocity
  joint_pos_isaac   (13,)   — joint positions in Isaac order
  joint_vel_isaac   (13,)   — joint velocities in Isaac order
  default_joint_pos (13,)   — default joint positions in Isaac order
  cmd               (3,)    — velocity command [vx, vy, wz]
  last_action       (13,)   — last action (zeros at Step0)
  obs               (48,)   — the observation vector Isaac computed
  joint_names       (13,)   — joint name strings in Isaac order

Run with:
  cd IsaacLab
  python scripts/dump_v6_isaac_state_step0.py --num_envs 1 --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump V6 humanoid Isaac Step0 state for obs verification")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output", type=str, default="isaac_step0_state.npz")
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

    # Disable ALL randomization so state is deterministic
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.physics_material = None
    # Disable joint reset randomization — we want exact default pose
    env_cfg.events.reset_robot_joints = None
    # Disable base pose randomization — we want exact init pose
    env_cfg.events.reset_base = None
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

    # Get Step0 observation (BEFORE any action is taken)
    obs, _ = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    cmd_manager = env.unwrapped.command_manager
    joint_names = list(robot.joint_names)
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()

    labels = (
        ["ang_vel_x*0.2", "ang_vel_y*0.2", "ang_vel_z*0.2",
         "grav_x", "grav_y", "grav_z",
         "cmd_vx", "cmd_vy", "cmd_wz"]
        + [f"pos_rel_{jn}" for jn in joint_names]
        + [f"vel_{jn}*0.05" for jn in joint_names]
        + [f"last_act_{jn}" for jn in joint_names]
    )

    def dump_step(step_name, obs_tensor, last_act_np):
        """Extract and print state for a given step."""
        # Force zero velocity command
        cmd = cmd_manager.get_command("base_velocity")
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0

        # Re-read obs with zero command
        obs_now, _ = env.get_observations()

        root_pos_w = robot.data.root_pos_w[0].cpu().numpy()
        root_quat_w = robot.data.root_quat_w[0].cpu().numpy()
        root_lin_vel_w = robot.data.root_lin_vel_w[0].cpu().numpy()
        root_ang_vel_b = robot.data.root_ang_vel_b[0].cpu().numpy()
        projected_gravity = robot.data.projected_gravity_b[0].cpu().numpy()
        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        joint_vel = robot.data.joint_vel[0].cpu().numpy()
        cmd_np = cmd[0].cpu().numpy()
        obs_np = obs_now[0].cpu().numpy()

        print(f"\n{'='*70}")
        print(f"Isaac {step_name} State Dump")
        print(f"{'='*70}")
        print(f"  root_pos_w:         {root_pos_w}")
        print(f"  root_quat_w (wxyz): {root_quat_w}")
        print(f"  root_lin_vel_w:     {root_lin_vel_w}")
        print(f"  root_ang_vel_b:     {root_ang_vel_b}")
        print(f"  projected_gravity:  {projected_gravity}")
        print(f"  cmd:                {cmd_np}")
        print(f"  last_action:        {last_act_np}")
        print(f"  joint_names (Isaac order): {joint_names}")

        print(f"\n  --- Joint state (Isaac order) ---")
        for i, jname in enumerate(joint_names):
            print(f"    [{i:2d}] {jname:14s}  pos={joint_pos[i]:+.8f}  "
                  f"def={default_joint_pos[i]:+.4f}  "
                  f"rel={joint_pos[i]-default_joint_pos[i]:+.8f}  "
                  f"vel={joint_vel[i]:+.8f}")

        print(f"\n  --- Observation (48 dims) ---")
        for i, label in enumerate(labels):
            if i < len(obs_np):
                print(f"    obs[{i:2d}] {label:25s} = {obs_np[i]:+.8f}")

        return {
            "root_pos_w": root_pos_w.astype(np.float64),
            "root_quat_w": root_quat_w.astype(np.float64),
            "root_lin_vel_w": root_lin_vel_w.astype(np.float64),
            "root_ang_vel_b": root_ang_vel_b.astype(np.float64),
            "projected_gravity": projected_gravity.astype(np.float64),
            "joint_pos_isaac": joint_pos.astype(np.float64),
            "joint_vel_isaac": joint_vel.astype(np.float64),
            "default_joint_pos": default_joint_pos.astype(np.float64),
            "cmd": cmd_np.astype(np.float64),
            "last_action": last_act_np.astype(np.float64),
            "obs": obs_np.astype(np.float64),
            "joint_names": np.array(joint_names),
        }

    # ---- Step 0: before any action ----
    last_action_step0 = np.zeros(13, dtype=np.float32)
    step0_data = dump_step("Step0", obs, last_action_step0)

    # ---- Step 1: take one policy step, then dump ----
    with torch.inference_mode():
        # Force zero cmd before policy
        cmd = cmd_manager.get_command("base_velocity")
        cmd[:, 0] = 0.0; cmd[:, 1] = 0.0; cmd[:, 2] = 0.0

        actions = policy(obs)
        action_np = actions[0].detach().cpu().numpy()
        print(f"\n  --- Policy action (Step0 -> Step1) ---")
        for i, jname in enumerate(joint_names):
            if i < len(action_np):
                print(f"    [{i:2d}] {jname:14s}  act={action_np[i]:+.8f}")

        obs, _, _, _ = env.step(actions)

    step1_data = dump_step("Step1", obs, action_np.astype(np.float32))

    # Save both steps
    output_path = args_cli.output
    save_dict = {}
    for key, val in step0_data.items():
        save_dict[f"step0_{key}"] = val
    for key, val in step1_data.items():
        save_dict[f"step1_{key}"] = val
    np.savez(output_path, **save_dict)
    print(f"\n[SAVED] {output_path}")
    print(f"  Contains step0_* and step1_* for all state variables")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
