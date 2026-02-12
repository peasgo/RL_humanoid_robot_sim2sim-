"""在 IsaacLab 中测试 V4b 策略，记录obs和位置数据。
验证策略在训练环境中是否能正确响应不同cmd。

用法:
    cd /home/rl/RL-human_robot/IsaacLab
    /home/rl/anaconda3/envs/isaaclab/bin/python scripts/test_v4b_isaaclab.py --headless
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.v4_quadruped.flat_env_cfg import V4QuadrupedFlatEnvCfg


def main():
    env_cfg = V4QuadrupedFlatEnvCfg()
    env_cfg.scene.num_envs = 5
    # 禁用随机化以便对比
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 加载v4b策略
    policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/exported/policy.pt"
    print(f"Loading policy: {policy_path}")
    policy = torch.jit.load(policy_path, map_location=env.device)
    policy.eval()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"Observation shape: {obs.shape}")

    robot = env.scene["robot"]

    # 定义不同cmd
    cmds = torch.tensor([
        [0.5, 0, 0],     # env0: 前进
        [-0.3, 0, 0],    # env1: 后退
        [0, 0.3, 0],     # env2: 左移
        [0, 0, 0.5],     # env3: 左转
        [0, 0, 0],       # env4: 站立
    ], dtype=torch.float32, device=env.device)
    cmd_names = ["FWD 0.5", "BWD 0.3", "LEFT 0.3", "TURN_L 0.5", "STAND"]

    # 强制设置cmd
    env.command_manager.get_command("base_velocity")[:] = cmds

    init_pos = robot.data.root_pos_w.clone()
    
    # 打印初始obs
    print(f"\n初始obs (env0):")
    o = obs[0].cpu().numpy()
    print(f"  lin_vel  = {o[0:3]}")
    print(f"  ang_vel  = {o[3:6]}")
    print(f"  gravity  = {o[6:9]}")
    print(f"  cmd      = {o[9:12]}")
    print(f"  qj_rel   = {o[12:29]}")
    print(f"  dqj      = {o[29:46]}")
    print(f"  last_act = {o[46:62]}")

    # 记录数据
    all_obs = []
    all_actions = []
    all_pos = []

    for step in range(500):
        # 强制cmd
        env.command_manager.get_command("base_velocity")[:] = cmds

        with torch.no_grad():
            actions = policy(obs)

        obs_dict, _, _, _, _ = env.step(actions)
        obs = obs_dict["policy"]

        if step < 10 or step % 50 == 49:
            all_obs.append(obs.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            all_pos.append(robot.data.root_pos_w.cpu().numpy())

        if step % 100 == 99:
            pos = robot.data.root_pos_w
            print(f"\nStep {step+1}:")
            for i in range(5):
                fwd = -(pos[i, 1] - init_pos[i, 1]).item()
                lat = (pos[i, 0] - init_pos[i, 0]).item()
                h = pos[i, 2].item()
                print(f"  {cmd_names[i]:12s}: fwd={fwd:+.3f}m lat={lat:+.3f}m h={h:.3f}m")

    # 最终结果
    print("\n" + "="*80)
    print("最终结果 (500步后)")
    print("="*80)
    pos = robot.data.root_pos_w
    for i in range(5):
        fwd = -(pos[i, 1] - init_pos[i, 1]).item()
        lat = (pos[i, 0] - init_pos[i, 0]).item()
        h = pos[i, 2].item()
        print(f"  {cmd_names[i]:12s}: fwd={fwd:+.3f}m lat={lat:+.3f}m h={h:.3f}m")

    # 保存数据
    save_path = "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy_v4b/isaaclab_v4b_obs.npz"
    np.savez(save_path,
             obs=np.array(all_obs),
             actions=np.array(all_actions),
             pos=np.array(all_pos))
    print(f"\n数据已保存到: {save_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
