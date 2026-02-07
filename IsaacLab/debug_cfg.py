import sys
import os

# Add the source directory to sys.path so we can import isaaclab modules
sys.path.append("/home/rl/RL-human_robot/IsaacLab/source/isaaclab_tasks")
sys.path.append("/home/rl/RL-human_robot/IsaacLab/source/isaaclab_rl")
sys.path.append("/home/rl/RL-human_robot/IsaacLab/source")

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.v3_prone.agents.rsl_rl_ppo_cfg import V3StandToPronePPORunnerCfg
    
    cfg = V3StandToPronePPORunnerCfg()
    print(f"Policy Class Name: {cfg.policy.class_name}")
    print(f"Noise Std Type: {cfg.policy.noise_std_type}")
    print(f"Init Noise Std: {cfg.policy.init_noise_std}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

