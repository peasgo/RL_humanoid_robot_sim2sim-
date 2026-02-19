"""Play OLD V6 humanoid checkpoint that was trained with the original obs config.

The old checkpoint used:
  - history_length = 0 (no obs stacking) → actor input = 48
  - No separate critic obs (shared with actor) → critic input = 48

This script overrides env_cfg at runtime to match the old obs space,
WITHOUT modifying flat_env_cfg.py (so your current training is safe).

Usage:
  isaaclab -p IsaacLab/scripts/play_old_checkpoint.py \
    --task Isaac-Velocity-Flat-V6-Humanoid-v0 \
    --num_envs 1 \
    --checkpoint /home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-18_12-58-46/model_6000.pt
"""

import argparse
import sys
import os

# --- CLI ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play OLD V6 humanoid checkpoint (obs=48).")
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
    """Play old checkpoint with restored obs config."""
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ============================================================
    # KEY: Override obs config to match old checkpoint (48-dim)
    # ============================================================
    # 1. Remove history stacking: None = no history (base class default)
    env_cfg.observations.policy.history_length = None
    env_cfg.observations.policy.enable_corruption = False

    # 2. Remove separate critic obs → actor and critic share same 48-dim obs
    #    We need to rebuild the ObservationsCfg without the critic group
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    from isaaclab.utils import configclass

    @configclass
    class OldPolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OldObservationsCfg:
        policy: OldPolicyCfg = OldPolicyCfg()

    env_cfg.observations = OldObservationsCfg()

    print("[INFO] Overridden obs config for old checkpoint:")
    print("  - policy.history_length = None (no stacking, raw 48-dim)")
    print("  - No critic obs group (shared with actor)")
    # ============================================================

    # Force command ranges
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.forward_vel, args_cli.forward_vel)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.lateral_vel, args_cli.lateral_vel)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.yaw_vel, args_cli.yaw_vel)

    # Disable randomization for clean playback
    if hasattr(env_cfg, 'events'):
        if hasattr(env_cfg.events, 'base_external_force_torque'):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, 'push_robot'):
            env_cfg.events.push_robot = None

    # Resolve checkpoint path
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

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading OLD model from: {resume_path}")
    print(f"[INFO] Obs space: {env.observation_space}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    # Print robot info
    robot = env.unwrapped.scene["robot"]
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(f"\n{'='*60}")
    print(f"Fixed command: lin_vel_x={args_cli.forward_vel}, lin_vel_y={args_cli.lateral_vel}, ang_vel_z={args_cli.yaw_vel}")
    print(f"Root quaternion (w,x,y,z): {np.round(root_quat, 4)}")
    print(f"Obs shape: {obs.shape}")
    print(f"{'='*60}\n")

    step_count = 0
    print_interval = int(1.0 / dt)

    # Get joint names for printing
    joint_names = robot.data.joint_names
    print(f"[INFO] Joint names ({len(joint_names)}): {joint_names}")
    print(f"[INFO] Default joint pos: {robot.data.default_joint_pos[0].cpu().numpy().round(4)}")
    print(f"[INFO] dt={dt:.4f}s, print_interval={print_interval} steps (~1s)\n")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            step_count += 1
            if step_count % print_interval == 0:
                robot = env.unwrapped.scene["robot"]
                # Position & orientation
                pos = robot.data.root_pos_w[0].cpu().numpy()
                root_quat = robot.data.root_quat_w[0].cpu().numpy()
                # Velocities
                lin_vel_w = robot.data.root_lin_vel_w[0].cpu().numpy()
                ang_vel_w = robot.data.root_ang_vel_w[0].cpu().numpy()
                root_quat_t = robot.data.root_quat_w
                lin_vel_w_t = robot.data.root_lin_vel_w
                lin_vel_body = quat_apply_yaw(root_quat_t, lin_vel_w_t)[0].cpu().numpy()
                ang_vel_body = robot.data.root_ang_vel_b[0].cpu().numpy()
                # Command
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy()
                # Joints
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                joint_vel = robot.data.joint_vel[0].cpu().numpy()
                default_pos = robot.data.default_joint_pos[0].cpu().numpy()
                joint_pos_rel = joint_pos - default_pos
                # Actions
                act = actions[0].cpu().numpy()
                # Obs
                obs_np = obs[0].cpu().numpy()

                t = step_count * dt
                print(f"{'='*70}")
                print(f"[t={t:6.1f}s] step={step_count}")
                print(f"  pos:  x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:.3f}")
                print(f"  quat: w={root_quat[0]:.4f}  x={root_quat[1]:.4f}  y={root_quat[2]:.4f}  z={root_quat[3]:.4f}")
                print(f"  cmd:  vx={cmd[0]:+.3f}  vy={cmd[1]:+.3f}  wz={cmd[2]:+.3f}")
                print(f"  body_vel:  vx={lin_vel_body[0]:+.3f}  vy={lin_vel_body[1]:+.3f}")
                print(f"  world_vel: vx={lin_vel_w[0]:+.3f}  vy={lin_vel_w[1]:+.3f}  vz={lin_vel_w[2]:+.3f}")
                print(f"  body_ang_vel: wx={ang_vel_body[0]:+.3f}  wy={ang_vel_body[1]:+.3f}  wz={ang_vel_body[2]:+.3f}")
                print(f"  joint_pos_rel: {np.round(joint_pos_rel, 3)}")
                print(f"  joint_vel:     {np.round(joint_vel, 2)}")
                print(f"  actions:       {np.round(act, 3)}")
                print(f"  obs (shape={obs_np.shape}): min={obs_np.min():.3f} max={obs_np.max():.3f} mean={obs_np.mean():.3f}")
                print()

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
