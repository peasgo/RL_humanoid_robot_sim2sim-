"""Compare RSL-RL inference policy vs JIT exported policy on the same obs."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Isaac-Velocity-Flat-V6-Humanoid-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.physics_material = None
    env_cfg.events.reset_robot_joints = None
    env_cfg.events.reset_base = None
    env_cfg.observations.policy.enable_corruption = False

    ckpt = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-18_12-58-46/model_6000.pt"
    jit_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v6_humanoid_flat/2026-02-18_12-58-46/exported/policy.pt"

    env = gym.make("Isaac-Velocity-Flat-V6-Humanoid-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(ckpt)
    policy_rsl = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    jit_policy = torch.jit.load(jit_path, map_location=device)

    obs, _ = env.get_observations()
    print(f"\nobs device={obs.device} dtype={obs.dtype} shape={obs.shape}")
    print(f"obs[0,:6] = {obs[0,:6].cpu().numpy()}")

    with torch.inference_mode():
        act_rsl = policy_rsl(obs)
        act_jit = jit_policy(obs)

    print(f"\nRSL policy:  {act_rsl[0].cpu().numpy()}")
    print(f"JIT policy:  {act_jit[0].cpu().numpy()}")
    print(f"Max diff:    {(act_rsl - act_jit).abs().max().item():.6e}")
    print(f"Match:       {torch.allclose(act_rsl, act_jit, atol=1e-4)}")

    # Also test with exact zero obs on CPU
    obs_zero = torch.zeros(1, 48, device=device)
    obs_zero[0, 5] = -1.0
    obs_zero[0, 10] = 0.2; obs_zero[0, 11] = 0.2
    obs_zero[0, 16] = 0.4; obs_zero[0, 17] = -0.4
    obs_zero[0, 18] = -0.2; obs_zero[0, 19] = 0.2

    with torch.inference_mode():
        act_rsl2 = policy_rsl(obs_zero)
        act_jit2 = jit_policy(obs_zero)

    print(f"\n--- With manually constructed obs ---")
    print(f"RSL policy:  {act_rsl2[0].cpu().numpy()}")
    print(f"JIT policy:  {act_jit2[0].cpu().numpy()}")
    print(f"Max diff:    {(act_rsl2 - act_jit2).abs().max().item():.6e}")
    print(f"Match:       {torch.allclose(act_rsl2, act_jit2, atol=1e-4)}")

    simulation_app.close()


main()
