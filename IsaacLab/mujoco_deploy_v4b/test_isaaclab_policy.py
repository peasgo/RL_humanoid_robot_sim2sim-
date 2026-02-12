#!/usr/bin/env python3
"""在IsaacLab环境中直接运行策略，验证策略在训练环境中是否能正确响应cmd。

如果策略在IsaacLab中也不能正确响应cmd，那问题在策略训练本身。
如果策略在IsaacLab中能正确响应，那问题在MuJoCo部署的sim2sim gap。
"""
import torch
import numpy as np
import argparse
import sys
import os

# 需要在isaaclab环境中运行
# 用法: /home/rl/anaconda3/envs/isaaclab/bin/python -m isaaclab.app.run -- IsaacLab/mujoco_deploy_v4b/test_isaaclab_policy.py

def main():
    from isaaclab.app import AppLauncher
    
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([
        "--headless",
        "--num_envs", "4",
    ])
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    
    # 导入V4配置
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from source.isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.v4_quadruped.flat_env_cfg import V4QuadrupedFlatEnvCfg
    
    # 创建环境
    env_cfg = V4QuadrupedFlatEnvCfg()
    env_cfg.scene.num_envs = 4
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 加载策略
    policy_path = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/exported/policy.pt"
    policy = torch.jit.load(policy_path, map_location=env.device)
    policy.eval()
    
    # 设置不同的cmd给4个环境
    cmds = [
        [0.5, 0, 0],    # env0: 前进
        [-0.3, 0, 0],   # env1: 后退
        [0, 0.3, 0],    # env2: 左移
        [0, 0, 0.5],    # env3: 左转
    ]
    cmd_names = ["FWD 0.5", "BWD 0.3", "LEFT 0.3", "TURN_L 0.5"]
    
    # 记录初始位置
    obs, _ = env.reset()
    
    # 强制设置cmd
    cmd_tensor = torch.tensor(cmds, dtype=torch.float32, device=env.device)
    env.command_manager.get_command("base_velocity")[:] = cmd_tensor
    
    init_pos = env.scene["robot"].data.root_pos_w.clone()
    init_quat = env.scene["robot"].data.root_link_quat_w.clone()
    
    print(f"初始位置: {init_pos}")
    print(f"初始四元数: {init_quat}")
    
    # 运行500步
    for step in range(500):
        # 强制cmd（防止被resample覆盖）
        env.command_manager.get_command("base_velocity")[:] = cmd_tensor
        
        with torch.no_grad():
            actions = policy(obs)
        
        obs, _, _, _, _ = env.step(actions)
        
        if step % 100 == 99:
            pos = env.scene["robot"].data.root_pos_w
            vel = env.scene["robot"].data.root_lin_vel_b
            print(f"\nStep {step+1}:")
            for i in range(4):
                dx = pos[i, 0] - init_pos[i, 0]
                dy = pos[i, 1] - init_pos[i, 1]
                # V4: forward = world -Y
                fwd = -(pos[i, 1] - init_pos[i, 1]).item()
                lat = (pos[i, 0] - init_pos[i, 0]).item()
                h = pos[i, 2].item()
                print(f"  {cmd_names[i]:12s}: fwd={fwd:+.3f}m lat={lat:+.3f}m h={h:.3f}m "
                      f"vel_b={vel[i].cpu().numpy()}")
    
    # 最终结果
    print("\n" + "="*80)
    print("最终结果 (500步后)")
    print("="*80)
    pos = env.scene["robot"].data.root_pos_w
    for i in range(4):
        fwd = -(pos[i, 1] - init_pos[i, 1]).item()
        lat = (pos[i, 0] - init_pos[i, 0]).item()
        h = pos[i, 2].item()
        print(f"  {cmd_names[i]:12s}: fwd={fwd:+.3f}m lat={lat:+.3f}m h={h:.3f}m")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
