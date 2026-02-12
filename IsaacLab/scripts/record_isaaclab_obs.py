"""
Record exact observations from IsaacLab for comparison with MuJoCo.
Saves first 50 policy steps of obs and actions to a .npz file.
"""
import torch
import numpy as np
import argparse
import sys
import os

# Must be before any omniverse imports
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomotion.velocity.config.v4_quadruped.flat_env_cfg import V4QuadrupedFlatEnvCfg

def main():
    env_cfg = V4QuadrupedFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-10_15-49-36/exported/policy.pt"
    device = "cuda:0"
    policy = torch.jit.load(policy_path, map_location=device)
    print(f"Policy loaded: {policy_path} on {device}")

    obs, _ = env.reset()
    obs_data = obs["policy"]
    print(f"Observation shape: {obs_data.shape}")

    # Set zero command
    env.command_manager.get_term("base_velocity").vel_command_b[:] = 0.0

    all_obs = []
    all_actions = []
    all_pos = []
    all_quat = []

    num_steps = 100
    for step in range(num_steps):
        # Record observation
        obs_np = obs_data.detach().cpu().numpy().squeeze()
        all_obs.append(obs_np.copy())

        # Get robot state
        root_state = env.scene["robot"].data.root_state_w
        pos = root_state[0, :3].cpu().numpy()
        quat = root_state[0, 3:7].cpu().numpy()  # IsaacLab uses [w,x,y,z]
        all_pos.append(pos.copy())
        all_quat.append(quat.copy())

        # Policy inference
        action = policy(obs_data)
        all_actions.append(action.cpu().detach().numpy().squeeze().copy())

        # Step
        obs, _, _, _, _ = env.step(action)
        obs_data = obs["policy"]

        # Ensure zero command
        env.command_manager.get_term("base_velocity").vel_command_b[:] = 0.0

        if step < 10 or step % 10 == 0:
            print(f"  Step {step:3d}: pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:.4f}) "
                  f"obs[:3]=[{obs_np[0]:+.4f},{obs_np[1]:+.4f},{obs_np[2]:+.4f}] "
                  f"obs[6:9]=[{obs_np[6]:+.4f},{obs_np[7]:+.4f},{obs_np[8]:+.4f}] "
                  f"act_max={np.max(np.abs(all_actions[-1])):.3f}")

    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)
    all_pos = np.array(all_pos)
    all_quat = np.array(all_quat)

    save_path = "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy_v4/isaaclab_reference_obs.npz"
    np.savez(save_path, obs=all_obs, actions=all_actions, pos=all_pos, quat=all_quat)
    print(f"\nSaved {num_steps} steps to {save_path}")
    print(f"  obs shape: {all_obs.shape}")
    print(f"  actions shape: {all_actions.shape}")

    env.close()

if __name__ == "__main__":
    main()
