"""在 IsaacLab 中测试 V4 四足策略的行为。
验证策略在 IsaacLab 环境中是否能正确行走、响应命令。

用法:
    cd /home/rl/RL-human_robot/IsaacLab
    conda run -n isaaclab python scripts/test_v4_policy_isaaclab.py --headless
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

# Import the V4 quadruped config
from isaaclab_tasks.manager_based.locomotion.velocity.config.v4_quadruped.flat_env_cfg import V4QuadrupedFlatEnvCfg


def main():
    # Create environment
    env_cfg = V4QuadrupedFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Load the exported policy
    policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-10_15-49-36/exported/policy.pt"
    print(f"Loading policy: {policy_path}")
    policy = torch.jit.load(policy_path)
    policy.eval()

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_manager.action_term_dim}")

    # Get robot reference
    robot = env.scene["robot"]

    print("\n" + "=" * 70)
    print("Phase 1: Zero velocity command (10 seconds)")
    print("=" * 70)

    # Override command to zero
    cmd_manager = env.command_manager
    
    total_steps = 0
    for phase in range(2):
        if phase == 0:
            phase_name = "ZERO CMD"
            phase_steps = 500  # 500 * 0.02 = 10s
            target_cmd = [0.0, 0.0, 0.0]
        else:
            phase_name = "FORWARD CMD (0.5 m/s)"
            phase_steps = 500
            target_cmd = [0.5, 0.0, 0.0]
            print(f"\n{'=' * 70}")
            print(f"Phase 2: Forward velocity command = {target_cmd} (10 seconds)")
            print(f"{'=' * 70}")

        for step in range(phase_steps):
            total_steps += 1

            # Override velocity command
            cmd = cmd_manager.get_command("base_velocity")
            cmd[:, 0] = target_cmd[0]  # forward
            cmd[:, 1] = target_cmd[1]  # lateral
            cmd[:, 2] = target_cmd[2]  # yaw

            # Policy inference
            with torch.no_grad():
                actions = policy(obs.cpu()).to(obs.device)

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]

            # Print status every 1 second (50 steps)
            if step % 50 == 0:
                pos = robot.data.root_pos_w[0].cpu().numpy()
                quat = robot.data.root_quat_w[0].cpu().numpy()
                lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
                ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
                
                # V4 remapped velocities
                fwd_vel = lin_vel[2]   # local +Z = forward
                lat_vel = lin_vel[0]   # local X = lateral
                yaw_rate = ang_vel[1]  # local Y = yaw
                
                action_np = actions[0].cpu().numpy()
                
                print(f"  [{phase_name} t={step*0.02:.1f}s] "
                      f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) "
                      f"fwd_vel={fwd_vel:+.3f} lat_vel={lat_vel:+.3f} yaw={yaw_rate:+.3f} "
                      f"act_max={np.max(np.abs(action_np)):.2f}")

            # Handle resets
            if terminated.any() or truncated.any():
                print(f"  [RESET at step {step}] terminated={terminated.any()}, truncated={truncated.any()}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("Test complete!")
    print(f"{'=' * 70}")

    env.close()


if __name__ == "__main__":
    main()
