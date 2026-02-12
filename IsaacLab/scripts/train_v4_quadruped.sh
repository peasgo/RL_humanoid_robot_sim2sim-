#!/bin/bash
# V4 四足机器人训练脚本
# =====================
#
# 使用方法:
#   cd IsaacLab
#   bash scripts/train_v4_quadruped.sh          # 默认训练（headless）
#   bash scripts/train_v4_quadruped.sh --play    # 播放已训练的策略
#   bash scripts/train_v4_quadruped.sh --viz     # 带可视化训练
#
# 环境名称: Isaac-Velocity-Flat-V4-Quadruped-v0
# 训练框架: RSL-RL (PPO)
# 预计训练时间: ~2-4小时 (3000 iterations, 4096 envs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ISAACLAB_DIR"

# 默认参数
NUM_ENVS=4096
MAX_ITERATIONS=3000
TASK="Isaac-Velocity-Flat-V4-Quadruped-v0"
HEADLESS="--headless"

# 解析参数
if [ "$1" == "--play" ]; then
    echo "=========================================="
    echo "  V4 四足机器人 - 播放模式"
    echo "=========================================="
    PLAY_TASK="Isaac-Velocity-Flat-V4-Quadruped-Play-v0"
    conda run -n isaaclab python scripts/reinforcement_learning/rsl_rl/play.py \
        --task "$PLAY_TASK" \
        --num_envs 50 \
        ${2:+"$2"} ${3:+"$3"} ${4:+"$4"}
    exit 0
fi

if [ "$1" == "--viz" ]; then
    HEADLESS=""
    NUM_ENVS=64
    echo "=========================================="
    echo "  V4 四足机器人 - 可视化训练模式"
    echo "  环境数量: $NUM_ENVS"
    echo "=========================================="
fi

echo "=========================================="
echo "  V4 四足机器人四足行走训练"
echo "=========================================="
echo "  任务: $TASK"
echo "  环境数量: $NUM_ENVS"
echo "  最大迭代: $MAX_ITERATIONS"
echo "  Headless: $HEADLESS"
echo "=========================================="

conda run -n isaaclab python scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    $HEADLESS \
    ${1:+"$1"} ${2:+"$2"} ${3:+"$3"}
