# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ParallelhumanRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO 训练配置（软体四足模型 Parallelhuman，复杂地形版）"""

    # === rollout / logging ===
    num_steps_per_env = 24           # 每个环境收集步数
    max_iterations = 10000            # 训练迭代次数
    save_interval = 500        
    experiment_name = "Parallelhuman_rough"
    empirical_normalization = False  # 若 obs 未归一化，可设 True

    # === policy 网络结构 ===
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                  # 初始探索噪声（降低以避免超限）
        actor_hidden_dims=[512, 256, 128],   # Actor 网络层数
        critic_hidden_dims=[512, 256, 128],  # Critic 网络层数
        activation="elu",                    # 激活函数
    )

    # === PPO 算法超参数 ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ParallelhumanFlatPPORunnerCfg(ParallelhumanRoughPPORunnerCfg):
    """PPO 训练配置（软体四足模型 Parallelhuman，平地版）"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "Parallelhuman_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

