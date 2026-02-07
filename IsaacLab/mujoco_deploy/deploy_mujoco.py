"""
MuJoCo Sim2Sim 推理脚本
用于将 Isaac Lab 训练的策略部署到 MuJoCo
"""

import torch
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


# 配置路径
POLICY_PATH = "/home/rl/RL-human_robot/IsaacLab/logs/rsl_rl/Parallelhuman_flat/2026-01-19_22-02-54/exported/policy.pt"
XML_PATH = "/home/rl/RL-human_robot/IsaacLab/mujoco_deploy/WholeAssembleV2_mujoco_v2.xml"

# 关节顺序 (与Isaac Lab中actuators定义顺序一致)
# legs: RHipP, RHipY, RHipR, RKneeP, LHipP, LHipY, LHipR, LKneeP
# ankles: RAankleP, RAnkleR, LAnkleP, LAnkleR

# 默认关节位置 (来自 parallelhuman.py)
DEFAULT_JOINT_POS = np.array([0.0] * 12)

# 动作缩放 (来自 velocity_env_cfg.py)
ACTION_SCALE = 0.5   # 必须与训练配置保持一致 (0.5)

# 动作安全裁剪与非线性压缩
ACTION_CLIP = 1.0
# 动作低通滤波系数
ACTION_FILTER_ALPHA = 0.1

# 控制频率参数 (与Isaac Lab一致)
DECIMATION = 4   # 每次策略推理执行4步仿真
SIM_DT = 0.005   # MuJoCo时间步长

# MuJoCo中关节在qpos中的顺序 (XML定义顺序)
# 右腿完整 + 左腿完整
MUJOCO_JOINT_ORDER = [
    "RHipP", "RHipY", "RHipR", "RKneeP", "RAankleP", "RAnkleR",
    "LHipP", "LHipY", "LHipR", "LKneeP", "LAnkleP", "LAnkleR"
]

# 观测关节顺序（按模型声明顺序）
OBS_JOINT_ORDER = MUJOCO_JOINT_ORDER.copy()

# 动作关节顺序（按训练时 actuators 定义顺序）
ACTION_JOINT_ORDER = [
    "RHipP", "RHipY", "RHipR", "RKneeP",
    "LHipP", "LHipY", "LHipR", "LKneeP",
    "RAankleP", "RAnkleR", "LAnkleP", "LAnkleR"
]

# 从MuJoCo顺序到动作顺序的映射
MUJOCO_TO_ACTION = [MUJOCO_JOINT_ORDER.index(j) for j in ACTION_JOINT_ORDER]
# 从动作顺序到MuJoCo顺序的映射
ACTION_TO_MUJOCO = [ACTION_JOINT_ORDER.index(j) for j in MUJOCO_JOINT_ORDER]


class MujocoDeployer:
    def __init__(self, xml_path, policy_path):
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 加载策略
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        
        # 观测关节索引（按模型顺序）
        self.obs_joint_qpos_adr = []  # qpos地址
        self.obs_joint_qvel_adr = []  # qvel地址
        for name in OBS_JOINT_ORDER:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.obs_joint_qpos_adr.append(self.model.jnt_qposadr[jid])
            self.obs_joint_qvel_adr.append(self.model.jnt_dofadr[jid])

        # 动作关节索引（按训练动作顺序）
        self.actuator_ids = []
        self.action_joint_ranges = []
        self.action_joint_qpos_adr = []
        for name in ACTION_JOINT_ORDER:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.action_joint_qpos_adr.append(self.model.jnt_qposadr[jid])
            # jnt_range 可能为空，对其进行保护性读取
            try:
                self.action_joint_ranges.append(self.model.jnt_range[jid].copy())
            except Exception:
                self.action_joint_ranges.append((-10.0, 10.0))
            # actuator 名称在 XML 中与 joint 同名 (position actuators)，安全获取 id
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            except Exception:
                # 若找不到 actuator，则记录 -1，后续跳过
                aid = -1
            self.actuator_ids.append(aid)
        
        # 状态变量
        self.last_action = np.zeros(len(ACTION_JOINT_ORDER))
        # 先使用静止指令，验证站立稳定性
        self.command = np.array([0.0, 0.0, 0.0])  # vx, vy, yaw_rate

    def _safe_action(self, action_raw: np.ndarray) -> np.ndarray:
        """对策略输出进行安全处理，避免数值爆炸"""
        action_raw = np.asarray(action_raw, dtype=np.float32)
        # 防止 NaN/Inf
        action_raw = np.nan_to_num(action_raw, nan=0.0, posinf=0.0, neginf=0.0)
        # 压缩到 [-1, 1]
        action = np.tanh(action_raw)
        # 兜底裁剪
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        return action
        
    def get_obs(self):
        """
        获取观察值 (与Isaac Lab完全一致)
        观察空间: 48维
        - base_lin_vel (3): 基座线速度 (body坐标系)
        - base_ang_vel (3): 基座角速度 (body坐标系)
        - projected_gravity (3): 投影重力
        - velocity_commands (3): 速度指令 vx, vy, yaw_rate
        - joint_pos (12): 关节相对位置
        - joint_vel (12): 关节速度
        - last_action (12): 上一帧动作
        """
        # 获取基座四元数 (MuJoCo: w,x,y,z)
        quat = self.data.qpos[3:7]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy用xyzw
        rot_matrix = rot.as_matrix()
        
        # 使用 mj_objectVelocity 直接获取 body frame 下的线速度和角速度
        # MuJoCo 的 qvel 对于 free joint:
        # qvel[0:3] 是世界系线速度
        # qvel[3:6] 是 body 系角速度
        # 直接使用 mj_objectVelocity(..., flg_local=1) 更稳健，直接拿到 body 系下的 (ang, lin)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, vel6, 1)
        
        # mj_objectVelocity 结果顺序: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
        base_ang_vel = vel6[0:3]
        base_lin_vel = vel6[3:6]
        
        # 投影重力 (重力方向在body坐标系下的表示)
        gravity_world = np.array([0, 0, -1])
        projected_gravity = rot_matrix.T @ gravity_world
        
        # 速度指令
        velocity_commands = self.command.copy()
        
        # 关节位置 (按Isaac Lab顺序, 相对于默认位置)
        joint_pos = np.array([self.data.qpos[adr] for adr in self.obs_joint_qpos_adr])
        joint_pos = joint_pos - DEFAULT_JOINT_POS
        
        # 关节速度 (按Isaac Lab顺序)
        joint_vel = np.array([self.data.qvel[adr] for adr in self.obs_joint_qvel_adr])
        
        # 拼接观察值
        # 此时所有的观测值都应该是：
        # 1. 坐标系正确对齐 (Base在Body系)
        # 2. 这里的数值应当是 Raw Physics Data (除非RSL_RL由额外的Normalize层，通常是指 normalize_obs=True)
        # 如果训练时开启了 normalize_obs，而这里输入未归一化的数据，不仅 scale 不对，还会缺少 running_mean 的偏移。
        # 但在Sim2Sim调试阶段，首先确保物理量本身的定义一致。
        
        obs = np.concatenate([
            base_lin_vel,       # 3
            base_ang_vel,       # 3
            projected_gravity,  # 3
            velocity_commands,  # 3
            joint_pos,          # 12
            joint_vel,          # 12
            self.last_action    # 12
        ])
        
        return obs
    
    def step(self, action):
        """执行一步动作"""
        # 裁剪动作范围
        action = np.clip(action, -1.0, 1.0)
        # 动作低通滤波，抑制抖动
        action = (1.0 - ACTION_FILTER_ALPHA) * self.last_action + ACTION_FILTER_ALPHA * action
        
        # 计算目标关节位置
        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE

        # 关节限位裁剪 (与IsaacLab URDF一致)
        joint_limits_dict = {
            'RHipP': (-1.0, 1.0), 'RHipY': (-0.5, 0.5), 'RHipR': (-0.5, 0.5),
            'RKneeP': (-2.0, 0.2), 'RAankleP': (-0.5, 0.5), 'RAnkleR': (-0.3, 0.3),
            'LHipP': (-1.0, 1.0), 'LHipY': (-0.5, 0.5), 'LHipR': (-0.5, 0.5),
            'LKneeP': (-2.0, 0.5), 'LAnkleP': (-0.5, 0.5), 'LAnkleR': (-0.3, 0.3)
        }
        for i, jname in enumerate(ACTION_JOINT_ORDER):
            if jname in joint_limits_dict:
                low, high = joint_limits_dict[jname]
                target_pos[i] = np.clip(target_pos[i], low, high)
        
        # 设置actuator控制信号
        # 设置actuator控制信号
        # 注意: MuJoCo position actuator 的 ctrl 实际上是目标位置（position），
        # 但为了更保守地部署，避免策略直接把目标位置推远导致瞬间大力矩，
        # 这里提供两种模式：
        # 1) 直接写入目标位置（与训练一致）
        # 2) 写入当前位置 + 少量增量 (target_pos_delta_mode) 作为缓和版本
        target_pos_direct = target_pos
        # 读取当前关节位置
        current_pos = np.array([self.data.qpos[adr] for adr in self.action_joint_qpos_adr])

        # 如果需要保守模式，将目标设为当前位置 + delta * small_gain
        TARGET_DELTA_SAFE_GAIN = 0.2
        target_pos_safe = current_pos + (target_pos - current_pos) * TARGET_DELTA_SAFE_GAIN

        use_safe_delta = True
        if use_safe_delta:
            write_pos = target_pos_safe
        else:
            write_pos = target_pos_direct

        for i, aid in enumerate(self.actuator_ids):
            if aid < 0:
                # 跳过缺失的 actuator
                continue
            # 写入期望位置到 ctrl
            self.data.ctrl[aid] = float(write_pos[i])
        
        # 执行多步仿真 (decimation)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        
        # 记录动作
        self.last_action = action.copy()
    
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始高度
        self.data.qpos[2] = 0.35
        
        # 设置初始关节位置 (按Isaac Lab顺序)
        for i, adr in enumerate(self.obs_joint_qpos_adr):
            self.data.qpos[adr] = DEFAULT_JOINT_POS[i]
        
        # 前向运动学
        mujoco.mj_forward(self.model, self.data)
        
        # 重置状态
        self.last_action = np.zeros(len(ACTION_JOINT_ORDER))
    
    def run(self):
        """运行仿真"""
        self.reset()
        
        step_count = 0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                # 获取观察
                obs = self.get_obs()
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # 策略推理
                with torch.no_grad():
                    action_raw = self.policy(obs_tensor).squeeze().cpu().numpy()

                # 策略输出安全处理
                action = self._safe_action(action_raw)

                # 基于观测的额外安全缩放（防止观测不匹配导致策略饱和输出）
                base_ang = obs[3:6]
                ang_norm = np.linalg.norm(base_ang)
                # 阶梯缩放策略：角速度大时强制降低动作幅值
                if ang_norm > 2.0:
                    scale = 0.1
                elif ang_norm > 1.0:
                    scale = 0.3
                elif ang_norm > 0.5:
                    scale = 0.6
                else:
                    scale = 1.0
                action = action * scale

                # 当没有速度指令并且基座速度很小时，应用死区以保持静止
                if np.linalg.norm(self.command) < 1e-3:
                    lin_norm = np.linalg.norm(obs[0:3])
                    if lin_norm < 0.02 and ang_norm < 0.02:
                        action = np.zeros_like(action)

                # 小幅度动作置零（防止抖动）
                if np.all(np.abs(action) < 0.02):
                    action = np.zeros_like(action)
                
                # 调试信息 (前10步 + 每50步)
                if step_count < 10 or step_count % 50 == 0:
                    print(f"\n=== Step {step_count} ===")
                    print(f"base_height: {self.data.qpos[2]:.4f}")
                    print(f"base_pos_xy: [{self.data.qpos[0]:.3f}, {self.data.qpos[1]:.3f}]")
                    print(f"action_raw: {action_raw}")
                    print(f"action: {action}")
                
                # 如果机器人摔倒则重置
                if self.data.qpos[2] < 0.15:
                    print(f"\n机器人摔倒! height={self.data.qpos[2]:.3f}, 重置中...")
                    self.reset()
                    step_count = 0
                    continue
                
                # 执行动作
                self.step(action)
                
                # 更新显示
                viewer.sync()
                step_count += 1


def main():
    print("加载策略和模型...")
    deployer = MujocoDeployer(XML_PATH, POLICY_PATH)
    
    print("开始仿真...")
    print("速度指令: vx=0.5, vy=0.0, yaw_rate=0.0")
    print("观察维度: 48")
    print("按 Ctrl+C 退出")
    
    deployer.run()


if __name__ == "__main__":
    main()
