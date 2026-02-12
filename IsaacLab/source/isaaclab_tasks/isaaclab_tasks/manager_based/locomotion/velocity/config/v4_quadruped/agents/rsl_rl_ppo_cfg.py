# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class V4QuadrupedFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """V4四足行走平地PPO训练配置"""

    num_steps_per_env = 24
    max_iterations = 23500
    save_interval = 500
    experiment_name = "v4_quadruped_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # 从1.0降到0.5，平衡探索与平滑（sim2real关键参数）
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 从0.01降到0.005，减少探索噪声，鼓励确定性策略
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,  # 从0.01降到0.008，更保守的策略更新
        max_grad_norm=1.0,
    )
