# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Stand-to-Prone PPO Training Configuration
======================================================

站立到倒地任务的 RSL-RL PPO 算法训练超参数配置。

关键设计考虑：
1. 较大的网络结构：处理复杂的多阶段动作
2. 较小的学习率：保证训练稳定性
3. 较大的探索噪声：鼓励探索不同的倒地策略
4. 较长的训练时间：多阶段任务需要更多迭代
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class V3StandToPronePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    V3 人形机器人站立到倒地任务 PPO 训练配置
    
    任务特点：
    - 多阶段动作：站立 -> 深蹲 -> 跪地 -> 俯卧
    - 接触拓扑突变：需要处理复杂的接触状态变化
    - 软着陆要求：需要学习缓冲动作
    
    网络设计：
    - 使用较大的网络（512-256-128）处理复杂的状态空间
    - ELU 激活函数在连续控制中表现更好
    """

    # === 训练参数 ===
    num_steps_per_env = 32           # 每个环境收集步数（较长的 horizon）
    max_iterations = 10000           # 训练迭代次数（多阶段任务需要更多迭代）
    save_interval = 500              # 模型保存间隔
    experiment_name = "V3Humanoid_stand_to_prone"  # 实验名称
    empirical_normalization = True   # 启用经验归一化（关键：防止梯度爆炸）
    run_name = ""                    # 运行名称
    logger = "tensorboard"           # 日志记录器
    resume = False                   # 是否恢复训练
    load_run = None                  # 加载的运行目录
    load_checkpoint = None           # 加载的检查点

    # === 策略网络结构 ===
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,                  # 关键修复：降低初始噪声从0.5到0.1，提高训练初期稳定性
        noise_std_type="log",                # 使用log参数化，保证std始终为正
        actor_hidden_dims=[512, 256, 128],   # Actor 网络层数
        critic_hidden_dims=[512, 256, 128],  # Critic 网络层数
        activation="elu",                    # ELU 激活函数
    )

    # === PPO 算法超参数 ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # 价值损失系数
        use_clipped_value_loss=True,   # 使用裁剪的价值损失
        clip_param=0.2,                # PPO 裁剪参数
        entropy_coef=0.01,             # 熵系数
        num_learning_epochs=5,         # 每次更新的学习轮数
        num_mini_batches=8,            # mini-batch 数量
        learning_rate=3.0e-4,          # 学习率（标准值）
        schedule="adaptive",           # 自适应学习率调度
        gamma=0.99,                    # 折扣因子
        lam=0.95,                      # GAE lambda
        desired_kl=0.01,               # 目标 KL 散度
        max_grad_norm=1.0,             # 梯度裁剪（关键：防止梯度爆炸）
    )


@configclass
class V3CrawlingPPORunnerCfg(V3StandToPronePPORunnerCfg):
    """
    V3 人形机器人四足爬行任务 PPO 训练配置
    
    从俯卧姿态开始，学习四足爬行。
    相比站立到倒地任务，这是一个更简单的任务。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "V3Humanoid_crawling"
        
        # 爬行任务可以用稍小的网络
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]
        
        # 较少的迭代次数
        self.max_iterations = 15000
        
        # 较小的探索噪声（任务相对简单）
        self.policy.init_noise_std = 1.0


@configclass
class V3StandToProneCurriculumPPORunnerCfg(V3StandToPronePPORunnerCfg):
    """
    V3 人形机器人站立到倒地任务 PPO 训练配置 - 课程学习版
    
    使用课程学习策略，分阶段训练：
    1. 阶段1：从俯卧姿态开始，学习保持平衡
    2. 阶段2：从跪姿开始，学习手撑地
    3. 阶段3：从深蹲开始，学习跪下
    4. 阶段4：从站立开始，完整过程
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "V3Humanoid_stand_to_prone_curriculum"
        
        # 课程学习需要更多迭代
        self.max_iterations = 30000
        
        # 更大的探索噪声（初期阶段）
        self.policy.init_noise_std = 1.5
        
        # 更小的学习率（保证稳定过渡）
        self.algorithm.learning_rate = 3.0e-4
