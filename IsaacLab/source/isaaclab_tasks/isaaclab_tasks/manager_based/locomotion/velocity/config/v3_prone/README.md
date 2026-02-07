# V3 人形机器人站立到倒地（Stand-to-Prone）训练指南

## 概述

本指南详细说明如何使用 Isaac Lab 训练 V3 人形机器人完成从站立姿态到俯卧姿态的过渡动作。

### 任务目标

1. **站立到倒地**：机器人从直立站立姿态平滑过渡到俯卧/四足姿态
2. **动作过程自然**：深蹲 → 跪地 → 手撑地 → 俯卧
3. **软着陆**：避免高冲击力损坏硬件

### 技术挑战

- **接触拓扑突变**：双足支撑 → 膝盖接触 → 手掌接触 → 躯干着地
- **质心高度下降**：从 ~0.30m 降到 ~0.10m
- **多阶段奖励设计**：引导动作过程

---

## 文件结构

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/v3_prone/
├── __init__.py                    # 环境注册
├── prone_env_cfg.py               # 环境配置（奖励、观测、动作）
└── agents/
    ├── __init__.py                # 导出配置
    └── rsl_rl_ppo_cfg.py          # PPO 训练超参数
```

---

## 已注册的环境

| 环境 ID | 描述 | 用途 |
|---------|------|------|
| `Isaac-V3Humanoid-StandToProne-v0` | 站立到俯卧完整过程 | 主要训练任务 |
| `Isaac-V3Humanoid-Crawling-v0` | 四足爬行（从俯卧开始） | 爬行训练 |
| `Isaac-V3Humanoid-StandToProne-Curriculum-v0` | 课程学习版本 | 分阶段训练 |

---

## 训练命令

### 1. 基础训练（站立到倒地）

```bash
cd IsaacLab

# 无头模式训练（推荐，速度快）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-V3Humanoid-StandToProne-v0 \
    --headless

# 带可视化训练（调试用）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-V3Humanoid-StandToProne-v0
```

### 2. 课程学习训练（推荐）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-V3Humanoid-StandToProne-Curriculum-v0 \
    --headless
```

### 3. 四足爬行训练

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-V3Humanoid-Crawling-v0 \
    --headless
```

### 4. 测试/演示训练好的模型

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-V3Humanoid-StandToProne-v0 \
    --num_envs=50
```

---

## 奖励函数详解

### 任务目标奖励

| 奖励项 | 权重 | 描述 |
|--------|------|------|
| `height_descent` | +3.0 | 鼓励降低质心高度到目标 0.10m |
| `descent_progress` | +2.0 | 奖励持续下降的进度 |
| `orientation_prone` | +1.5 | 鼓励躯干保持水平 |
| `alive` | +1.0 | 存活奖励 |

### 安全与平滑惩罚

| 惩罚项 | 权重 | 描述 |
|--------|------|------|
| `soft_landing` | -0.3 | 惩罚高冲击力（>100N） |
| `velocity_penalty` | -0.5 | 惩罚过快的下降速度（>0.5m/s） |
| `joint_limits` | -0.2 | 惩罚接近关节极限 |
| `action_rate` | -0.01 | 惩罚动作突变 |
| `joint_acc` | -1e-6 | 惩罚关节加速度 |

---

## 观测空间

| 观测变量 | 维度 | 描述 |
|----------|------|------|
| `base_lin_vel` | 3 | 基座线速度 |
| `base_ang_vel` | 3 | 基座角速度 |
| `projected_gravity` | 3 | 重力投影（感知倾斜） |
| `joint_pos` | 26 | 关节位置（相对默认） |
| `joint_vel` | 26 | 关节速度 |
| `last_action` | 26 | 上一步动作 |

**总观测维度**: 87

---

## 训练参数

### PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 5e-4 | 较小，保证稳定 |
| `num_steps_per_env` | 32 | 较长的 horizon |
| `max_iterations` | 20000 | 多阶段任务需要更多迭代 |
| `entropy_coef` | 0.015 | 稍大，鼓励探索 |
| `clip_param` | 0.2 | PPO 裁剪参数 |
| `gamma` | 0.99 | 折扣因子 |

### 网络结构

- **Actor**: [512, 256, 128]
- **Critic**: [512, 256, 128]
- **激活函数**: ELU

---

## 调试建议

### 1. 观察 TensorBoard 指标

```bash
tensorboard --logdir=logs/rsl_rl/V3Humanoid_stand_to_prone
```

关键指标：
- `Episode/Reward`: 应该逐渐上升
- `Episode/Length`: 应该逐渐变长（没有过早终止）
- `Reward/height_descent`: 应该逐渐增加
- `Reward/soft_landing`: 应该接近 0（没有高冲击）

### 2. 常见问题

**问题**: 机器人直接摔倒
- **原因**: 探索噪声太大或奖励设计不当
- **解决**: 减小 `init_noise_std` 或增加 `alive` 奖励权重

**问题**: 机器人不动
- **原因**: 惩罚项权重过大
- **解决**: 减小 `velocity_penalty` 和 `soft_landing` 权重

**问题**: 动作抖动
- **原因**: 动作缩放过大
- **解决**: 减小 `action_scale`（当前 0.25）

### 3. 分阶段调试

1. **先训练爬行**：使用 `Isaac-V3Humanoid-Crawling-v0`，确保机器人能在俯卧姿态保持平衡
2. **再训练完整过程**：使用 `Isaac-V3Humanoid-StandToProne-v0`

---

## V3 机器人关节结构

```
V3 人形机器人 (26 DOF)
├── 腰部 (2 DOF)
│   ├── Waist
│   └── Waist_2
├── 右腿 (6 DOF)
│   ├── RHIPp (髋部俯仰)
│   ├── RHIPy (髋部偏航)
│   ├── RHIPr (髋部横滚)
│   ├── RKNEEP (膝盖)
│   ├── RANKLEp (踝部俯仰)
│   └── RANKLEy (踝部偏航)
├── 左腿 (6 DOF)
│   └── (同右腿，前缀 L)
├── 右臂 (6 DOF)
│   ├── RSDp (肩部俯仰)
│   ├── RSDy (肩部偏航)
│   ├── RSDr (肩部横滚)
│   ├── RARMp (上臂)
│   ├── RARMAP (前臂)
│   └── RARMAy (手腕)
└── 左臂 (6 DOF)
    └── (同右臂，前缀 L)
```

---

## 进阶：自定义奖励

如果需要修改奖励函数，编辑 `prone_env_cfg.py` 中的 `V3StandToProneRewardsCfg` 类：

```python
@configclass
class V3StandToProneRewardsCfg:
    # 修改权重
    height_descent = RewTerm(
        func=height_descent_reward,
        weight=5.0,  # 增加权重
        params={
            "target_height": 0.08,  # 更低的目标高度
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
```

---

## 参考资料

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL 库](https://github.com/leggedrobotics/rsl_rl)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
