# Outline

phrl

1. urdf倒入为usd文件

2. 写一份phrl.config

3. 参考h1.py,使用manager_based method 写一份环境配置（包含观测空间，actionspace，reward，events等）

4. register phrl.config

python /home/rl/RL-human_robot/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Parallelhuman-v0  --load_run /h/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-16_15-21-36 --checkpoint /home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-16_15-21-36/model_17000.pt  

python /home/rl/RL-human_robot/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Parallelhuman-v0  --resume  --load_run=2026-01-15_16-47-56 --headless

unset ROS_PACKAGE_PATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

python /home/rl/RL-human_robot/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Parallelhuman-v0  --load_run /h/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-19_22-02-54 --checkpoint /home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-19_22-02-54/model_9999.pt




































