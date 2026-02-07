# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid PPO Training Configuration
======================================

RSL-RL PPO 算法的训练超参数配置。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class V3HumanoidRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    V3 人形机器人 PPO 训练配置 - 复杂地形版
    
    网络结构较大，适合复杂任务。
    """

    # === 训练参数 ===
    num_steps_per_env = 24           # 每个环境收集步数
    max_iterations = 15000           # 训练迭代次数
    save_interval = 500              # 模型保存间隔
    experiment_name = "V3Humanoid_rough"  # 实验名称（日志目录）
    empirical_normalization = False  # 是否使用经验归一化
    run_name = ""                    # 运行名称（可选）
    logger = "tensorboard"           # 日志记录器
    resume = False                   # 是否恢复训练
    load_run = None                  # 加载的运行目录
    load_checkpoint = None           # 加载的检查点

    # === 策略网络结构 ===
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                  # 初始探索噪声
        actor_hidden_dims=[512, 256, 128],   # Actor 网络层数
        critic_hidden_dims=[512, 256, 128],  # Critic 网络层数
        activation="elu",                    # 激活函数
    )

    # === PPO 算法超参数 ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # 价值损失系数
        use_clipped_value_loss=True,   # 是否使用裁剪的价值损失
        clip_param=0.2,                # PPO 裁剪参数
        entropy_coef=0.01,             # 熵系数（控制探索）
        num_learning_epochs=5,         # 每次更新的学习轮数
        num_mini_batches=8,            # mini-batch 数量
        learning_rate=1.0e-3,          # 学习率
        schedule="adaptive",           # 学习率调度策略
        gamma=0.99,                    # 折扣因子
        lam=0.95,                      # GAE lambda
        desired_kl=0.01,               # 目标 KL 散度
        max_grad_norm=1.0,             # 梯度裁剪
    )


@configclass
class V3HumanoidFlatPPORunnerCfg(V3HumanoidRoughPPORunnerCfg):
    """
    V3 人形机器人 PPO 训练配置 - 平地版
    
    网络结构较小，适合简单任务，训练更快。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "V3Humanoid_flat"
        
        # 平地任务可以用更小的网络
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]
        
        # 可以适当减少训练迭代次数
        self.max_iterations = 10000


@configclass
class V3HumanoidLegsOnlyPPORunnerCfg(V3HumanoidFlatPPORunnerCfg):
    """
    V3 人形机器人 PPO 训练配置 - 仅腿部控制版
    
    最简化的配置，适合初期训练。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "V3Humanoid_legs_only"
        
        # 更小的网络
        self.policy.actor_hidden_dims = [128, 64, 32]
        self.policy.critic_hidden_dims = [128, 64, 32]
        
        # 更少的迭代
        self.max_iterations = 8000
        
        # 更大的探索噪声（初期训练）
        self.policy.init_noise_std = 1.5
