# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Stand-to-Prone (Down) Task Configuration
=====================================================

任务目标：从站立姿态 -> 平滑过渡到 -> 俯卧姿态 (Stand -> Squat -> Hands Down -> Prone)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import torch
import math

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.v3_humanoid import V3_HUMANOID_CFG


# ============================================================
# 辅助函数: 观测
# ============================================================

def base_height_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回机器人质心高度（Z坐标），带数值保护"""
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[:, 2].unsqueeze(-1)
    # 数值保护：裁剪到合理范围 [0, 5]米
    height = torch.clamp(height, min=0.0, max=5.0)
    return torch.nan_to_num(height, nan=1.0, posinf=5.0, neginf=0.0)

# ============================================================
# 辅助函数: 奖励逻辑
# ============================================================

def orientation_prone_measure(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_pitch: float = 0.42,
    z_tol: float = 0.15,
    y_tol: float = 0.15,
    height_threshold: float = 0.25,
) -> torch.Tensor:
    """
    四足姿态引导奖励函数（带高度门控）
    
    V3机器人坐标系：X=右，Y=前，Z=上
    
    重力投影方向推导：
    - 站立时：gravity_b ≈ (0, 0, -1)
    - 前倾时：gravity_b.y 变**正**，gravity_b.z 趋向 0
    - 后仰时：gravity_b.y 变**负**
    
    四足支撑目标姿态：
    1. 躯干Z轴水平 → gravity_body.z ≈ 0
    2. 躯干前倾约25° → gravity_body.y ≈ +0.42（sin(25°)≈0.42）
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
        target_pitch: 目标前倾角度对应的重力Y分量（正值=前倾）
                     +0.42对应25度，+0.5对应30度，+0.7对应45度
        z_tol: Z轴容差（允许偏差范围）
        y_tol: Y轴容差（允许偏差范围）
        height_threshold: 高度门控阈值，只有低于此高度才激活姿态约束
    
    Returns:
        姿态偏差惩罚值，越小越好（站立阶段返回0）
    """
    robot = env.scene[asset_cfg.name]
    
    # 使用 Isaac Lab 内置的 projected_gravity_b（与 mdp.projected_gravity 一致）
    gravity_body = robot.data.projected_gravity_b
    root_pos = robot.data.root_pos_w
    
    # 高度门控：只有蹲下来了，才开始在意是不是趴平了
    height = root_pos[:, 2]
    mask = (height < height_threshold).float()
    
    # 使用"区间内不惩罚，区间外再惩罚"的 hinge 形式，训练更稳定
    #
    # 约束逻辑（V3坐标系：X右，Y前，Z上）：
    # 1. Z轴应该水平（gravity_body.z ≈ 0）
    z_err = torch.clamp(torch.abs(gravity_body[:, 2]) - z_tol, min=0.0)
    
    # 2. Y轴应该前倾（gravity_body.y ≈ target_pitch，正值）
    #    前倾时 gravity_body.y > 0
    y_err = torch.clamp(torch.abs(gravity_body[:, 1] - target_pitch) - y_tol, min=0.0)

    reward = torch.square(z_err) + torch.square(y_err)
    # 数值保护：裁剪到合理范围
    reward = torch.clamp(reward, min=0.0, max=10.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=0.0)
    
    # 应用高度门控：站立阶段不惩罚姿态
    return reward * mask

def height_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tol: float = 0.03,
) -> torch.Tensor:
    """
    高度进度奖励（严格版）:
    - 区间内（|h-h*|<=tol）给满分 1
    - 区间外用 exp(- (|h-h*|-tol)^2 / 0.002) 快速衰减
    
    关键修复：衰减系数从0.1降至0.002，避免"站桩"
    - 站立(0.302m)时：outside=0.092, reward≈exp(-4.23)≈0.015
    - 下蹲(0.25m)时：outside=0.04, reward≈exp(-0.8)≈0.449
    - 目标(0.19m)时：reward=1.0
    - 扑倒(0.055m)时：outside=0.115, reward≈exp(-6.61)≈0.001（几乎为0）
    """
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    current_height = base_pos[:, 2]
    error = torch.abs(current_height - target_height)
    outside = torch.clamp(error - tol, min=0.0, max=5.0)
    reward = torch.where(error <= tol, torch.ones_like(error), torch.exp(-torch.square(outside) / 0.002))
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

def height_too_low_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.12,
) -> torch.Tensor:
    """
    过低高度惩罚：阻止机器人扑倒在地面上
    
    四足支撑的目标高度是0.19m，低于0.12m说明已经趴平了。
    这个惩罚在 height < min_height 时线性增大：
    - height = 0.12m → penalty = 0
    - height = 0.06m → penalty = (0.12-0.06)/0.12 = 0.5
    - height = 0.0m  → penalty = 1.0
    
    关键作用：打破"扑倒"局部最优。之前没有任何机制阻止高度降到0.19m以下，
    策略发现扑倒能拿 forward_lean 满分，而 reach_target_height 在0.055m时
    奖励≈0但不是负的。这个惩罚让扑倒变成负奖励。
    
    Args:
        min_height: 最低允许高度（m），低于此高度开始惩罚
    
    Returns:
        惩罚值 [0, 1]，越低越大（用于负权重）
    """
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    current_height = base_pos[:, 2]
    
    # 低于 min_height 时惩罚，越低惩罚越大
    below = torch.clamp(min_height - current_height, min=0.0)
    penalty = below / min_height  # 归一化到 [0, 1]
    
    return torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=0.0)

def soft_landing_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 50.0 # N
) -> torch.Tensor:
    """
    软着陆: 惩罚过大的接触力冲击。
    这迫使机器人"慢慢"趴下，而不是直接摔在地上。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    force_norm = torch.norm(forces, dim=-1)
    
    # 取所有接触点中最大的力
    max_impact = torch.max(force_norm, dim=-1).values
    
    # 超过阈值的部分进行惩罚，限制最大惩罚
    penalty = torch.clamp(max_impact - threshold, min=0.0, max=1000.0)
    # 缩放系数
    reward = penalty * 0.01
    return torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=0.0)

# 缓存手部body索引，避免每步都遍历查找
_hand_body_indices_cache: dict = {}

def hand_forward_reach_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    hand_body_names: list = ["LARMAy", "RARMAy"],
    height_threshold: float = 0.30,
) -> torch.Tensor:
    """
    手臂前伸奖励：引导手臂向前方伸展（带高度门控）
    
    根据参考轨迹，双足→四足的关键动作序列：
    1. 下蹲（高度降低）
    2. 手臂从身体侧面向前方伸展 0.4-0.6m
    3. 手着地形成前支撑
    
    奖励手末端在Y轴（前方）的位置超过质心位置。
    高度门控：只有当质心高度 < height_threshold 时才激活，
    避免在站立阶段就鼓励手臂前伸（那样会失去平衡）。
    
    V3坐标系：X=右，Y=前，Z=上
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
        hand_body_names: 手部body名称列表
        height_threshold: 高度门控阈值，低于此高度才激活奖励
    
    Returns:
        手臂前伸奖励值 [0, 1]
    """
    global _hand_body_indices_cache
    robot = env.scene[asset_cfg.name]
    
    # 缓存手部body索引（只在第一次调用时查找）
    cache_key = asset_cfg.name
    if cache_key not in _hand_body_indices_cache:
        indices, names = robot.find_bodies(hand_body_names)
        _hand_body_indices_cache[cache_key] = indices
        print(f"[hand_forward_reach] Found hand bodies: {names} at indices {indices}")
    
    hand_indices = _hand_body_indices_cache[cache_key]
    if len(hand_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 高度门控：只有在下蹲后才激活手臂前伸奖励
    current_height = robot.data.root_pos_w[:, 2]
    mask = (current_height < height_threshold).float()
    
    # 获取质心Y坐标（前方位置）
    com_y = robot.data.root_pos_w[:, 1]
    
    # 获取手部body的世界坐标
    body_pos = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
    
    # 计算手部Y坐标的平均值
    hand_y_positions = torch.stack([body_pos[:, idx, 1] for idx in hand_indices], dim=1)
    avg_hand_y = hand_y_positions.mean(dim=1)
    
    # 奖励手部在质心前方的距离
    # forward_reach = 手Y坐标 - 质心Y坐标（正值表示在前方）
    forward_reach = avg_hand_y - com_y
    
    # 使用平滑奖励：手越往前，奖励越高，最大为1
    # forward_reach=0.1m → reward≈0.39
    # forward_reach=0.3m → reward≈0.78
    # forward_reach=0.5m → reward≈0.92
    reward = torch.tanh(torch.clamp(forward_reach * 3.0, min=0.0, max=3.0))
    
    # 应用高度门控
    reward = reward * mask
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


# 缓存肢端body索引
_limb_body_indices_cache: dict = {}

def quadruped_stance_reward(
    env: ManagerBasedRLEnv,
    left_hand_sensor_cfg: SceneEntityCfg,
    right_hand_sensor_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
    height_threshold: float = 0.25,
    hand_body_names: list = ["LARMAy", "RARMAy"],
    foot_body_names: list = ["LANKLEy", "RANKLEy"],
    ground_z_threshold: float = 0.03,
    weight_force_threshold: float = 20.0,
) -> torch.Tensor:
    """
    四足支撑奖励（完全接地版）:
    
    "完全接地"的三重判定：
    1. 有接触力（force > force_threshold）— 确实在接触
    2. 肢端Z坐标贴地（Z < ground_z_threshold）— 掌/脚底贴地面
    3. 接触力足够大（force > weight_force_threshold）— 在承重，不是轻碰
    
    位置约束：
    - 手端应在质心前方（Y > com_Y）
    - 脚端应在质心后方（Y < com_Y）
    
    每个肢端的"接地质量"是连续值 [0, 1]，而非二值 0/1：
    - grounded_score = contact × position_valid × height_score × force_score
    - height_score = exp(-z²/σ²)  越贴地越高
    - force_score = tanh(force/threshold)  力越大越高
    
    V3坐标系：X=右，Y=前，Z=上
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
        height_threshold: 激活四足支撑奖励的质心高度阈值（m）
        force_threshold: 判定接触的最小力阈值（N）
        hand_body_names: 手部末端body名称
        foot_body_names: 脚部末端body名称
        ground_z_threshold: 肢端贴地的Z坐标阈值（m），越小要求越严格
        weight_force_threshold: 承重力阈值（N），超过此值才算"完全接地"
    
    Returns:
        四足支撑奖励，四肢完全接地时奖励最大
    """
    global _limb_body_indices_cache
    robot = env.scene[asset_cfg.name]
    current_height = robot.data.root_pos_w[:, 2]
    
    # 只有在低高度时才激活四足支撑奖励
    height_mask = (current_height < height_threshold).float()
    
    # 缓存肢端body索引
    cache_key = f"{asset_cfg.name}_limbs"
    if cache_key not in _limb_body_indices_cache:
        all_names = hand_body_names + foot_body_names
        indices, names = robot.find_bodies(all_names)
        _limb_body_indices_cache[cache_key] = {
            "hand_indices": indices[:len(hand_body_names)],
            "foot_indices": indices[len(hand_body_names):],
        }
        print(f"[quadruped_stance] Hand bodies: {names[:len(hand_body_names)]} at {indices[:len(hand_body_names)]}")
        print(f"[quadruped_stance] Foot bodies: {names[len(hand_body_names):]} at {indices[len(hand_body_names):]}")
    
    cached = _limb_body_indices_cache[cache_key]
    hand_indices = cached["hand_indices"]
    foot_indices = cached["foot_indices"]
    
    # 获取质心位置
    com_y = robot.data.root_pos_w[:, 1]  # 前方
    
    # 获取所有body位置和姿态
    body_pos = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
    
    # === 接触力检测 ===
    def limb_contact_force(sensor_name: str) -> torch.Tensor:
        """返回肢端的最大接触力（连续值），而非0/1"""
        try:
            sensor = env.scene.sensors[sensor_name]
            forces = sensor.data.net_forces_w_history[:, 0, :, :]
            max_force = torch.norm(forces, dim=-1).max(dim=-1).values
            return max_force
        except Exception:
            return torch.zeros(env.num_envs, device=env.device)

    lh_force = limb_contact_force(left_hand_sensor_cfg.name)
    rh_force = limb_contact_force(right_hand_sensor_cfg.name)
    lf_force = limb_contact_force(left_foot_sensor_cfg.name)
    rf_force = limb_contact_force(right_foot_sensor_cfg.name)
    
    # === 接地质量评分（连续值 [0, 1]）===
    def grounded_score(body_idx: int, contact_force: torch.Tensor, is_hand: bool) -> torch.Tensor:
        """
        计算单个肢端的"完全接地"质量分数
        
        综合考虑：
        1. 是否有接触（force > threshold）
        2. 肢端高度是否贴地（Z接近0）
        3. 接触力是否足够大（承重）
        4. 位置是否合理（手在前/脚在后）
        """
        if body_idx < 0 or body_idx >= body_pos.shape[1]:
            return torch.zeros(env.num_envs, device=env.device)
        
        limb_y = body_pos[:, body_idx, 1]  # 前方坐标
        limb_z = body_pos[:, body_idx, 2]  # 高度
        
        # 1. 接触判定（二值门控）
        has_contact = (contact_force > force_threshold).float()
        
        # 2. 贴地程度（连续值）：Z越接近0分数越高
        # exp(-z²/σ²): z=0→1.0, z=0.03→0.71, z=0.05→0.29, z=0.1→0.0
        sigma_z = ground_z_threshold  # 0.03m
        height_score = torch.exp(-torch.square(torch.clamp(limb_z, min=0.0)) / (sigma_z ** 2))
        
        # 3. 承重程度（连续值）：力越大分数越高，饱和于weight_force_threshold
        # tanh(force/threshold): 10N→0.46, 20N→0.76, 50N→0.96
        force_score = torch.tanh(contact_force / weight_force_threshold)
        
        # 4. 位置合理性（二值门控）
        if is_hand:
            # 手在质心前方
            position_ok = (limb_y > com_y - 0.05).float()
        else:
            # 脚在质心后方
            position_ok = (limb_y < com_y + 0.05).float()
        
        # 综合评分 = 接触 × 贴地 × 承重 × 位置
        score = has_contact * height_score * force_score * position_ok
        
        return torch.clamp(score, min=0.0, max=1.0)
    
    # 计算四个肢端的接地质量
    lh_score = grounded_score(hand_indices[0], lh_force, is_hand=True) if len(hand_indices) > 0 else torch.zeros(env.num_envs, device=env.device)
    rh_score = grounded_score(hand_indices[1], rh_force, is_hand=True) if len(hand_indices) > 1 else torch.zeros(env.num_envs, device=env.device)
    lf_score = grounded_score(foot_indices[0], lf_force, is_hand=False) if len(foot_indices) > 0 else torch.zeros(env.num_envs, device=env.device)
    rf_score = grounded_score(foot_indices[1], rf_force, is_hand=False) if len(foot_indices) > 1 else torch.zeros(env.num_envs, device=env.device)

    # 总接地质量 = 四个肢端分数之和（最大4.0）
    total_grounded = lh_score + rh_score + lf_score + rf_score
    
    # 奖励：接地质量越接近4.0越好
    # 使用 total/4 作为奖励，这样每个肢端的贡献是均匀的
    reward = total_grounded / 4.0
    
    # 应用高度mask
    reward = reward * height_mask
    
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    return reward

def base_heading_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    朝向保持奖励: 惩罚Yaw角偏离初始方向
    
    确保机器人是"向前趴下"而不是侧向摔倒或原地打转。
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
    
    Returns:
        朝向偏差惩罚值（越小越好）
    """
    robot = env.scene[asset_cfg.name]
    root_quat = robot.data.root_quat_w
    
    # 提取Yaw角（绕Z轴旋转）
    # quat = [w, x, y, z]，Yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # 惩罚Yaw角偏离0（初始朝向）
    yaw_penalty = torch.square(yaw)
    
    # 数值保护
    yaw_penalty = torch.clamp(yaw_penalty, min=0.0, max=10.0)
    return torch.nan_to_num(yaw_penalty, nan=0.0, posinf=10.0, neginf=0.0)

def base_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    躯干接触惩罚: 惩罚base_link接触地面（但不终止）
    
    在训练初期，将base_contact从终止条件改为惩罚，
    避免机器人产生"恐地症"而不敢完全趴下。
    
    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        threshold: 判定接触的力阈值（N）
    
    Returns:
        接触惩罚值
    """
    try:
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, 0, :, :]
        max_force = torch.norm(forces, dim=-1).max(dim=-1).values
        
        # 超过阈值则惩罚
        penalty = (max_force > threshold).float()
        
        return torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=0.0)
    except:
        return torch.zeros(env.num_envs, device=env.device)

def trunk_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    躯干全范围接触惩罚（v4新增）：惩罚躯干任何部位触地
    
    使用专用的躯干接触传感器（contact_forces_trunk），
    检测 base_link + Waist + Waist_2 的接触力。
    
    v3只检测base_link，导致机器人用头（Waist_2）顶地来支撑。
    v4扩大检测范围到整个躯干。
    
    V3 link结构：
      base_link → Waist → Waist_2 → 肩膀/髋关节
    
    四足支撑时，只有四肢末端（手掌LARMAy/RARMAy、脚底LANKLEy/RANKLEy）
    应该接触地面，躯干必须悬空。
    
    使用连续值惩罚（力的大小），而非二值惩罚：
    - 轻微碰触（<threshold）不惩罚
    - 力越大惩罚越大（tanh归一化）
    
    Args:
        sensor_cfg: 躯干专用接触传感器配置（contact_forces_trunk）
        threshold: 判定接触的最小力阈值（N）
    
    Returns:
        躯干接触惩罚值 [0, 1]
    """
    try:
        sensor = env.scene.sensors[sensor_cfg.name]
        
        # net_forces_w_history: (num_envs, history_len, num_bodies, 3)
        forces = sensor.data.net_forces_w_history[:, 0, :, :]  # (num_envs, num_bodies, 3)
        force_norms = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
        
        # 取所有躯干body中最大的接触力
        max_trunk_force = force_norms.max(dim=-1).values  # (num_envs,)
        
        # 超过阈值的部分进行惩罚
        excess = torch.clamp(max_trunk_force - threshold, min=0.0)
        # 使用tanh归一化到[0,1]，50N时penalty≈0.76，100N时≈0.96
        penalty = torch.tanh(excess / 50.0)
        
        return torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        return torch.zeros(env.num_envs, device=env.device)

def base_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    躯干稳定性奖励: 惩罚侧向速度和角速度
    
    关键修复（v2）：
    - 只惩罚X方向（侧向）速度，完全放开Y方向（前后）速度
    - 从站立到四足支撑，质心需要向前移动（Y方向），惩罚Y速度会导致后仰
    - 允许垂直方向（Z）的速度（下蹲需要）
    - 角速度仍然全方向惩罚（防止打转/翻滚）
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
    
    Returns:
        稳定性惩罚值（越小越好）
    """
    robot = env.scene[asset_cfg.name]
    
    # 只惩罚X方向（侧向）速度
    # V3坐标系：X右，Y前，Z上
    # Y方向（前后）速度是趴下动作的必要运动，不惩罚
    # Z方向（上下）速度是下蹲的必要运动，不惩罚
    lin_vel = robot.data.root_lin_vel_b
    lateral_vel_penalty = torch.square(torch.clamp(lin_vel[:, 0], min=-10.0, max=10.0))
    
    # 惩罚角速度（所有方向），但降低pitch（绕X轴）的惩罚
    # 因为前倾需要pitch方向的角速度
    ang_vel = robot.data.root_ang_vel_b
    # X轴角速度（pitch/前倾）用较小系数，Y和Z轴正常惩罚
    ang_vel_penalty = 0.3 * torch.square(torch.clamp(ang_vel[:, 0], min=-10.0, max=10.0)) + \
                      torch.square(torch.clamp(ang_vel[:, 1], min=-10.0, max=10.0)) + \
                      torch.square(torch.clamp(ang_vel[:, 2], min=-10.0, max=10.0))
    
    reward = lateral_vel_penalty + ang_vel_penalty
    # 数值保护：裁剪到合理范围
    reward = torch.clamp(reward, min=0.0, max=100.0)
    return torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=0.0)

def anti_backward_fall_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    forward_axis: int = 1,
) -> torch.Tensor:
    """
    反后仰奖励：强力惩罚后仰，鼓励前倾
    
    重力投影方向推导（V3坐标系 Y=前方）：
    - 站立时：gravity_b ≈ (0, 0, -1)
    - 前倾时：身体向Y正方向倾斜 → gravity_b.y 变**正**（重力相对本体有Y正分量）
    - 后仰时：身体向Y负方向倾斜 → gravity_b.y 变**负**（重力相对本体有Y负分量）
    
    因此：
    - gravity_b[forward_axis] > 0 → 前倾 ✅
    - gravity_b[forward_axis] < 0 → 后仰 ❌ → 惩罚
    
    Args:
        forward_axis: 前方轴索引（0=X, 1=Y）
    
    Returns:
        后仰惩罚值（用于负权重），越大表示后仰越严重
    """
    robot = env.scene[asset_cfg.name]
    
    # 使用 Isaac Lab 内置的 projected_gravity_b
    gravity_body = robot.data.projected_gravity_b
    
    # 后仰检测：gravity_body[forward_axis] < 0 表示后仰
    # 取负值的绝对值作为惩罚（只惩罚负值=后仰，正值=前倾不惩罚）
    backward_component = torch.clamp(-gravity_body[:, forward_axis], min=0.0)
    
    # 平方惩罚：轻微后仰小惩罚，严重后仰大惩罚
    penalty = torch.square(backward_component)
    
    penalty = torch.clamp(penalty, min=0.0, max=10.0)
    return torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)

# 诊断计数器：控制打印频率
_forward_lean_debug_counter = {"count": 0}

def forward_lean_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    forward_axis: int = 1,
    target_lean: float = 0.42,
    lean_tolerance: float = 0.15,
) -> torch.Tensor:
    """
    前倾引导奖励（钟形版）：只奖励适度前倾，过度前倾（扑倒）反而衰减
    
    重力投影方向推导（V3坐标系 Y=前方）：
    - 站立时：gravity_b ≈ (0, 0, -1)，gravity_b.y ≈ 0
    - 前倾25°时：gravity_b.y ≈ +0.42（目标四足姿态）
    - 完全扑倒时：gravity_b.y ≈ +1.0（90°前倾，不是我们要的！）
    
    关键修复（v3）：之前用单调递增奖励（tanh），导致"越趴越好"，
    策略发现直接扑倒能拿满分。改为钟形奖励：
    - gravity_b.y = 0（站立）→ reward ≈ 0
    - gravity_b.y = 0.42（目标前倾）→ reward = 1.0（满分）
    - gravity_b.y = 1.0（完全扑倒）→ reward ≈ 0.05（几乎为0）
    
    Args:
        forward_axis: 前方轴索引（0=X, 1=Y），默认1（Y轴）
        target_lean: 目标前倾角度对应的gravity_b分量，+0.42≈sin(25°)
        lean_tolerance: 容差范围，在 [target-tol, target+tol] 内给满分
    
    Returns:
        前倾奖励值 [0, 1]，在目标前倾角度时最大
    """
    robot = env.scene[asset_cfg.name]
    
    # 使用 Isaac Lab 内置的 projected_gravity_b（更可靠）
    gravity_body = robot.data.projected_gravity_b  # (num_envs, 3)
    
    # === 诊断打印：每200步打印一次 ===
    _forward_lean_debug_counter["count"] += 1
    if _forward_lean_debug_counter["count"] % 200 == 1:
        gb = gravity_body[0]
        height = robot.data.root_pos_w[0, 2]
        print(f"[DEBUG forward_lean] step={_forward_lean_debug_counter['count']} "
              f"height={height:.3f} "
              f"gravity_b=({gb[0]:.3f}, {gb[1]:.3f}, {gb[2]:.3f}) "
              f"using axis={forward_axis}")
    
    # 前倾程度
    lean_value = gravity_body[:, forward_axis]
    
    # 钟形奖励：以 target_lean 为中心的高斯分布
    # 在容差范围内给满分，超出范围快速衰减
    error = torch.abs(lean_value - target_lean)
    outside = torch.clamp(error - lean_tolerance, min=0.0)
    
    # sigma=0.15: 偏离容差0.15→reward≈0.5, 偏离0.3→reward≈0.07
    # 这意味着：
    # - 站立(lean=0): error=0.42, outside=0.27, reward≈exp(-3.24)≈0.04
    # - 目标(lean=0.42): error=0, outside=0, reward=1.0
    # - 扑倒(lean=1.0): error=0.58, outside=0.43, reward≈exp(-8.22)≈0.0003
    reward = torch.exp(-torch.square(outside) / 0.02)
    
    # 额外：后仰（lean < 0）时奖励为0
    reward = reward * (lean_value > 0.0).float()
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


# ============================================================
# v5新增: 关节角度引导奖励（替代方向引导）
# ============================================================

# 缓存关节索引
_joint_indices_cache: dict = {}

def joint_pose_target_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_joints: dict = None,
) -> torch.Tensor:
    """
    关节角度目标奖励（v5核心）：直接引导关键关节趋向四足姿态角度
    
    与"方向引导"(forward_lean)的本质区别：
    - forward_lean 只告诉"身体应该前倾" → 策略找到扑倒/头顶地等捷径
    - joint_pose_target 告诉"每个关节应该在什么角度" → 精确定义目标姿态
    
    V3 四足姿态的关键关节角度（从URDF运动学推算）：
    - RHIPp: -1.2 (右髋前屈，axis=-X，负值=前屈)
    - LHIPp: +1.2 (左髋前屈，axis=+X，正值=前屈)
    - RKNEEP: +1.0 (右膝弯曲，axis=-X，正值=弯曲)
    - LKNEEp: -1.0 (左膝弯曲，axis=+X，负值=弯曲)
    - RSDp: +1.2 (右肩前伸，axis=+X，正值=前伸)
    - LSDp: +1.2 (左肩前伸，axis≈+X，正值=前伸)
    - RARMp: -0.5 (右肘弯曲，axis=+X，负值=弯曲)
    - LARMp: +0.5 (左肘弯曲，axis=-X，正值=弯曲)
    
    奖励计算：对每个目标关节，计算角度误差的高斯奖励
    reward = mean(exp(-error²/sigma²))
    
    无高度门控！从站立开始就有梯度信号。
    
    Args:
        asset_cfg: 机器人配置
        target_joints: {关节名: 目标角度(rad)} 字典
    
    Returns:
        关节角度匹配奖励 [0, 1]
    """
    global _joint_indices_cache
    robot = env.scene[asset_cfg.name]
    
    if target_joints is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 缓存关节索引
    cache_key = f"{asset_cfg.name}_target_joints"
    if cache_key not in _joint_indices_cache:
        joint_names = list(target_joints.keys())
        indices, names = robot.find_joints(joint_names)
        _joint_indices_cache[cache_key] = {
            "indices": indices,
            "names": names,
            "targets": torch.tensor(
                [target_joints[n] for n in joint_names],
                device=env.device, dtype=torch.float32
            ),
        }
        print(f"[joint_pose_target] Tracking joints: {names} at indices {indices}")
        print(f"[joint_pose_target] Target angles: {[target_joints[n] for n in joint_names]}")
    
    cached = _joint_indices_cache[cache_key]
    indices = cached["indices"]
    targets = cached["targets"]
    
    # 获取当前关节角度
    current_pos = robot.data.joint_pos  # (num_envs, num_joints)
    
    # 提取目标关节的当前角度
    # 注意：find_joints() 返回的 indices 可能是 list[int] 或 tensor，
    # 直接用切片索引更安全且高效
    if isinstance(indices, (list, tuple)):
        idx_tensor = torch.tensor(indices, device=env.device, dtype=torch.long)
    else:
        idx_tensor = indices
    joint_angles = current_pos[:, idx_tensor]  # (num_envs, num_target_joints)
    
    # 计算误差
    errors = joint_angles - targets.unsqueeze(0)  # (num_envs, num_target_joints)
    
    # 高斯奖励：sigma=0.5rad(≈29°)，误差越小奖励越高
    # error=0 → reward=1.0
    # error=0.5rad → reward=0.61
    # error=1.0rad → reward=0.14
    # error=1.5rad → reward=0.01
    sigma = 0.5
    per_joint_reward = torch.exp(-torch.square(errors) / (2 * sigma ** 2))
    
    # 取所有目标关节的平均奖励
    reward = per_joint_reward.mean(dim=1)
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


def height_linear_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.19,
    standing_height: float = 0.302,
    min_height: float = 0.10,
) -> torch.Tensor:
    """
    线性高度奖励（v5替代指数衰减版）：从站立到目标高度有持续梯度
    
    v4的问题：用 exp(-outside²/0.002) 衰减，站立时reward≈0.015，几乎没有梯度。
    
    v5改为分段线性：
    - height >= standing_height(0.302m) → reward = 0（站着不动没奖励）
    - height = target_height(0.19m) → reward = 1.0（目标高度满分）
    - height = min_height(0.10m) → reward = 0（太低也没奖励）
    - height < min_height → reward = 负值（惩罚扑倒）
    
    关键：站立(0.302m)到目标(0.19m)之间是线性增长的！
    - 站立(0.302m): reward = 0
    - 下蹲(0.25m): reward = (0.302-0.25)/(0.302-0.19) = 0.464
    - 目标(0.19m): reward = 1.0
    
    这意味着每降低1cm高度，奖励增加约0.089，策略有明确的梯度信号。
    
    Args:
        target_height: 目标高度（四足姿态）
        standing_height: 站立高度
        min_height: 最低允许高度
    """
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    current_height = base_pos[:, 2]
    
    reward = torch.zeros_like(current_height)
    
    # 区间1: target_height <= h <= standing_height → 线性从0到1
    upper_mask = (current_height >= target_height) & (current_height <= standing_height)
    reward = torch.where(
        upper_mask,
        (standing_height - current_height) / (standing_height - target_height),
        reward
    )
    
    # 区间2: min_height <= h < target_height → 线性从0到1（从下往上）
    lower_mask = (current_height >= min_height) & (current_height < target_height)
    reward = torch.where(
        lower_mask,
        (current_height - min_height) / (target_height - min_height),
        reward
    )
    
    # 区间3: h < min_height → 负值惩罚（线性递减）
    too_low_mask = current_height < min_height
    reward = torch.where(
        too_low_mask,
        (current_height - min_height) / min_height,  # 负值，h=0时reward=-1
        reward
    )
    
    # 区间4: h > standing_height → 0（不应该发生）
    
    return torch.nan_to_num(torch.clamp(reward, min=-1.0, max=1.0), nan=0.0)


# 缓存头部body索引
_head_body_index_cache: dict = {}

def trunk_level_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    head_body_name: str = "Waist_2",
    target_height: float = 0.19,
    sigma_level: float = 0.03,
    sigma_height: float = 0.03,
) -> torch.Tensor:
    """
    躯干水平+目标高度奖励（v5增强版）：
    
    双重约束：
    1. 头部(Waist_2)高度 ≈ base_link高度（躯干水平/平行）
    2. 头部高度 ≈ 目标高度0.19m（和base_link一起在目标高度）
    
    reward = level_reward × head_height_reward
    - level_reward: 头和base同高 → 1.0
    - head_height_reward: 头在0.19m → 1.0
    
    站立时：头比base高16cm → level≈0, 头在0.46m → height≈0 → reward≈0
    目标时：头≈base≈0.19m → level≈1.0, height≈1.0 → reward≈1.0
    
    Args:
        asset_cfg: 机器人配置
        head_body_name: 头部body名称（默认Waist_2）
        target_height: 目标高度（m），头部和base_link都应该在这个高度
        sigma_level: 水平约束的高斯宽度（m）
        sigma_height: 高度约束的高斯宽度（m）
    
    Returns:
        躯干水平+高度奖励 [0, 1]
    """
    global _head_body_index_cache
    robot = env.scene[asset_cfg.name]
    
    # 缓存头部body索引
    cache_key = f"{asset_cfg.name}_head"
    if cache_key not in _head_body_index_cache:
        indices, names = robot.find_bodies([head_body_name])
        if len(indices) == 0:
            print(f"[trunk_level] WARNING: body '{head_body_name}' not found!")
            _head_body_index_cache[cache_key] = -1
        else:
            _head_body_index_cache[cache_key] = indices[0]
            print(f"[trunk_level] Found head body: {names[0]} at index {indices[0]}")
    
    head_idx = _head_body_index_cache[cache_key]
    if head_idx < 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # base_link Z坐标（root_pos_w 就是 base_link 的世界坐标）
    base_z = robot.data.root_pos_w[:, 2]
    
    # Waist_2 Z坐标
    head_z = robot.data.body_pos_w[:, head_idx, 2]
    
    # 约束1：头部和base_link同高（躯干水平/平行）
    z_diff = torch.abs(head_z - base_z)
    level_reward = torch.exp(-torch.square(z_diff) / (2 * sigma_level ** 2))
    
    # 约束2：头部高度在目标高度0.19m
    head_height_err = torch.abs(head_z - target_height)
    height_reward = torch.exp(-torch.square(head_height_err) / (2 * sigma_height ** 2))
    
    # 两个约束相乘：只有同时满足才给高分
    reward = level_reward * height_reward
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


# 缓存肢端body索引（用于朝向检测）
_limb_orient_cache: dict = {}

def limb_flat_on_ground_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    hand_body_names: list = None,
    foot_body_names: list = None,
) -> torch.Tensor:
    """
    肢端掌面朝下奖励（v5.1新增）：确保手掌/脚掌完全踩在地面上
    
    问题：当前 quadruped_stance 只检查接触力和Z坐标，
    但机器人可能用关节侧面或背面接触地面，而不是掌面朝下。
    
    解决：检查每个肢端body的局部-Z轴在世界坐标系中的方向。
    当掌面完全朝下时，肢端的局部-Z轴应该指向世界-Z方向（即朝地面）。
    
    具体来说：
    - 从 body_quat_w 提取每个肢端的四元数
    - 计算局部-Z轴在世界坐标系中的方向向量
    - 该向量的世界Z分量应该为负值（指向地面）
    - reward = 该Z分量的负值（越负越好，即越朝下越好）
    
    四元数旋转局部Z轴 [0,0,1] 到世界坐标系：
    world_z = quat_rotate([0,0,1], quat)
    对于 quat = [w, x, y, z]：
    world_z_x = 2*(x*z + w*y)
    world_z_y = 2*(y*z - w*x)
    world_z_z = 1 - 2*(x*x + y*y)
    
    掌面朝下时 world_z_z ≈ -1（局部Z轴指向世界-Z）
    
    Args:
        asset_cfg: 机器人配置
        hand_body_names: 手部末端body名称列表
        foot_body_names: 脚部末端body名称列表
    
    Returns:
        肢端朝向奖励 [0, 1]，所有肢端掌面朝下时最大
    """
    global _limb_orient_cache
    robot = env.scene[asset_cfg.name]
    
    if hand_body_names is None:
        hand_body_names = ["LARMAy", "RARMAy"]
    if foot_body_names is None:
        foot_body_names = ["LANKLEy", "RANKLEy"]
    
    # 缓存肢端body索引
    cache_key = f"{asset_cfg.name}_limb_orient"
    if cache_key not in _limb_orient_cache:
        all_names = hand_body_names + foot_body_names
        indices, names = robot.find_bodies(all_names)
        _limb_orient_cache[cache_key] = indices
        print(f"[limb_flat] Found limb bodies for orientation: {names} at indices {indices}")
    
    limb_indices = _limb_orient_cache[cache_key]
    if len(limb_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 获取肢端四元数 (num_envs, num_bodies, 4) -> 提取目标body
    body_quats = robot.data.body_quat_w  # (num_envs, num_bodies, 4)
    
    total_score = torch.zeros(env.num_envs, device=env.device)
    
    for idx in limb_indices:
        if idx < 0 or idx >= body_quats.shape[1]:
            continue
        
        # 提取该body的四元数 [w, x, y, z]
        quat = body_quats[:, idx, :]  # (num_envs, 4)
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        
        # 计算局部Z轴在世界坐标系中的Z分量
        # 对于四元数 [w,x,y,z]，旋转 [0,0,1] 后的Z分量为：
        # world_z_z = 1 - 2*(x² + y²) = w² - x² - y² + z²
        world_z_z = 1.0 - 2.0 * (x * x + y * y)
        
        # 掌面朝下时 world_z_z ≈ -1
        # 转换为奖励：(-world_z_z + 1) / 2 → 朝下=1.0, 朝上=0.0, 水平=0.5
        # 但我们用更严格的高斯：以 world_z_z = -1 为目标
        orient_err = world_z_z + 1.0  # 目标是-1，所以误差 = world_z_z - (-1)
        score = torch.exp(-torch.square(orient_err) / (2 * 0.3 ** 2))
        # sigma=0.3: err=0→1.0, err=0.3→0.61, err=0.5→0.33, err=1.0→0.004
        
        total_score = total_score + score
    
    # 平均所有肢端的朝向分数
    num_limbs = len(limb_indices)
    reward = total_score / max(num_limbs, 1)
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


def foot_flat_on_ground_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg = None,
    right_foot_sensor_cfg: SceneEntityCfg = None,
    foot_body_names: list = None,
    force_threshold: float = 5.0,
    height_threshold: float = 0.25,
) -> torch.Tensor:
    """
    脚掌平踩地面奖励（v5.2新增）：综合接触力+方向约束
    
    问题：当前 limb_flat_on_ground_reward 只检查方向，不检查是否真的在接触地面。
    而 quadruped_stance 虽然检查接触力，但对脚掌方向约束不够。
    
    这个奖励综合两者：
    1. 脚掌必须有接触力（确实踩在地上）
    2. 脚掌的局部Z轴必须指向世界-Z（掌面朝下，完全踩平）
    
    只有同时满足才给高分：reward = contact_score × orientation_score
    
    高度门控：只有 height < height_threshold 时才激活
    
    Args:
        asset_cfg: 机器人配置
        left_foot_sensor_cfg: 左脚接触传感器
        right_foot_sensor_cfg: 右脚接触传感器
        foot_body_names: 脚部末端body名称列表
        force_threshold: 判定接触的最小力阈值（N）
        height_threshold: 激活奖励的质心高度阈值（m）
    
    Returns:
        脚掌平踩奖励 [0, 1]
    """
    robot = env.scene[asset_cfg.name]
    
    if foot_body_names is None:
        foot_body_names = ["LANKLEy", "RANKLEy"]
    
    current_height = robot.data.root_pos_w[:, 2]
    height_mask = (current_height < height_threshold).float()
    
    # 获取脚部body索引（使用全局缓存）
    cache_key = f"{asset_cfg.name}_foot_flat"
    if cache_key not in _limb_orient_cache:
        indices, names = robot.find_bodies(foot_body_names)
        _limb_orient_cache[cache_key] = indices
        print(f"[foot_flat] Found foot bodies: {names} at indices {indices}")
    
    foot_indices = _limb_orient_cache[cache_key]
    if len(foot_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # === 接触力检测 ===
    sensor_cfgs = [left_foot_sensor_cfg, right_foot_sensor_cfg]
    contact_scores = []
    for i, sensor_cfg in enumerate(sensor_cfgs):
        if sensor_cfg is None:
            contact_scores.append(torch.ones(env.num_envs, device=env.device))
            continue
        try:
            sensor = env.scene.sensors[sensor_cfg.name]
            forces = sensor.data.net_forces_w_history[:, 0, :, :]
            max_force = torch.norm(forces, dim=-1).max(dim=-1).values
            # 连续接触分数：force越大分数越高
            score = torch.tanh(max_force / 30.0)  # 30N时score≈0.96
            contact_scores.append(score)
        except Exception:
            contact_scores.append(torch.zeros(env.num_envs, device=env.device))
    
    # === 方向检测：脚掌局部Z轴应指向世界-Z ===
    body_quats = robot.data.body_quat_w
    orient_scores = []
    
    for idx in foot_indices:
        if idx < 0 or idx >= body_quats.shape[1]:
            orient_scores.append(torch.zeros(env.num_envs, device=env.device))
            continue
        
        quat = body_quats[:, idx, :]  # (num_envs, 4) [w, x, y, z]
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        
        # 局部Z轴在世界坐标系中的Z分量
        world_z_z = 1.0 - 2.0 * (x * x + y * y)
        
        # 掌面朝下时 world_z_z ≈ -1
        # 使用更严格的高斯：sigma=0.2（比limb_flat的0.3更严格）
        orient_err = world_z_z + 1.0  # 目标是-1
        score = torch.exp(-torch.square(orient_err) / (2 * 0.2 ** 2))
        orient_scores.append(score)
    
    # === 综合评分 ===
    total_reward = torch.zeros(env.num_envs, device=env.device)
    num_feet = min(len(contact_scores), len(orient_scores))
    
    for i in range(num_feet):
        # 接触 × 方向 = 综合分数
        combined = contact_scores[i] * orient_scores[i]
        total_reward = total_reward + combined
    
    reward = total_reward / max(num_feet, 1)
    reward = reward * height_mask
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


def hand_ground_contact_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hand_sensor_cfg: SceneEntityCfg = None,
    right_hand_sensor_cfg: SceneEntityCfg = None,
    hand_body_names: list = None,
    force_threshold: float = 5.0,
    height_threshold: float = 0.30,
) -> torch.Tensor:
    """
    手掌撑地奖励（v5.2新增）：确保双手向前撑住地面
    
    问题：当前 hand_forward_reach 只奖励手在前方，不检查是否真的撑在地上。
    quadruped_stance 虽然检查接触力，但四肢平均分配权重，手的贡献被稀释。
    
    这个奖励专门强调手掌必须：
    1. 有接触力（确实撑在地上）
    2. 手掌高度贴地（Z接近0）
    3. 手掌在质心前方（Y > com_Y）
    4. 手掌方向正确（掌面朝下）
    
    每只手独立评分，两只手的分数相乘（而非相加），
    这样只有一只手撑地时奖励很低，必须双手都撑地才能拿高分。
    
    Args:
        asset_cfg: 机器人配置
        left_hand_sensor_cfg: 左手接触传感器
        right_hand_sensor_cfg: 右手接触传感器
        hand_body_names: 手部末端body名称列表
        force_threshold: 判定接触的最小力阈值（N）
        height_threshold: 激活奖励的质心高度阈值（m）
    
    Returns:
        手掌撑地奖励 [0, 1]
    """
    robot = env.scene[asset_cfg.name]
    
    if hand_body_names is None:
        hand_body_names = ["LARMAy", "RARMAy"]
    
    current_height = robot.data.root_pos_w[:, 2]
    height_mask = (current_height < height_threshold).float()
    
    # 获取手部body索引
    cache_key = f"{asset_cfg.name}_hand_ground"
    if cache_key not in _limb_orient_cache:
        indices, names = robot.find_bodies(hand_body_names)
        _limb_orient_cache[cache_key] = indices
        print(f"[hand_ground] Found hand bodies: {names} at indices {indices}")
    
    hand_indices = _limb_orient_cache[cache_key]
    if len(hand_indices) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    
    com_y = robot.data.root_pos_w[:, 1]
    body_pos = robot.data.body_pos_w
    body_quats = robot.data.body_quat_w
    
    sensor_cfgs = [left_hand_sensor_cfg, right_hand_sensor_cfg]
    hand_scores = []
    
    for i, (idx, sensor_cfg) in enumerate(zip(hand_indices, sensor_cfgs)):
        if idx < 0 or idx >= body_pos.shape[1]:
            hand_scores.append(torch.zeros(env.num_envs, device=env.device))
            continue
        
        hand_y = body_pos[:, idx, 1]
        hand_z = body_pos[:, idx, 2]
        
        # 1. 接触力分数
        contact_score = torch.zeros(env.num_envs, device=env.device)
        if sensor_cfg is not None:
            try:
                sensor = env.scene.sensors[sensor_cfg.name]
                forces = sensor.data.net_forces_w_history[:, 0, :, :]
                max_force = torch.norm(forces, dim=-1).max(dim=-1).values
                contact_score = torch.tanh(max_force / 20.0)  # 20N时≈0.96
            except Exception:
                pass
        
        # 2. 贴地分数：Z越接近0越好
        height_score = torch.exp(-torch.square(torch.clamp(hand_z, min=0.0)) / (0.03 ** 2))
        
        # 3. 前方位置分数：手在质心前方
        forward_score = torch.sigmoid((hand_y - com_y) * 10.0)  # 在前方0.1m时≈0.73
        
        # 4. 方向分数：掌面朝下
        quat = body_quats[:, idx, :]
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z_q = quat[:, 3]
        world_z_z = 1.0 - 2.0 * (x * x + y * y)
        orient_err = world_z_z + 1.0
        orient_score = torch.exp(-torch.square(orient_err) / (2 * 0.3 ** 2))
        
        # 综合分数 = 接触 × 贴地 × 前方 × 方向
        score = contact_score * height_score * forward_score * orient_score
        hand_scores.append(torch.clamp(score, min=0.0, max=1.0))
    
    # 双手分数相乘：必须双手都撑地才能拿高分
    # sqrt使得梯度更平滑：如果一只手0.8另一只0.0，reward=0
    # 如果两只手都0.8，reward=0.8
    if len(hand_scores) >= 2:
        reward = torch.sqrt(hand_scores[0] * hand_scores[1] + 1e-8)
    else:
        reward = hand_scores[0] if hand_scores else torch.zeros(env.num_envs, device=env.device)
    
    reward = reward * height_mask
    
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)


# ============================================================
# 奖励配置: 站立 -> 趴下 (v5: 无门控 + 关节引导)
# ============================================================
@configclass
class V3StandToProneRewardsCfg:
    """
    v5.2 奖励配置：无门控 + 关节引导 + 躯干水平约束 + 四肢接触强化
    
    v5.2 核心改进（解决手掌不撑地、脚掌不平踩问题）：
    1. joint_pose_target 新增踝关节角度（RANKLEp/LANKLEp=-0.2），确保脚掌平踩
    2. 新增 hand_ground_contact 奖励：双手必须同时撑地（接触力×贴地×前方×方向）
    3. 新增 foot_flat_on_ground 奖励：脚掌必须有接触力且掌面朝下
    4. limb_flat 权重从2.0→3.0，增强掌面朝下约束
    
    设计原则：
    1. 所有奖励从站立开始就有梯度信号（无高度门控）
    2. 用关节角度目标精确定义四足姿态（替代方向引导）
    3. 下蹲和手臂前伸同时进行（手臂前伸无门控）
    4. 头部(Waist_2)和base_link同高（躯干水平约束）
    5. 四肢必须全部接触地面且掌面朝下（v5.2强化）
    
    目标姿态时奖励分解（预估）：
      height_linear:      +2.0 × 1.0   = +2.0   (h=0.19m满分)
      joint_pose_target:  +3.0 × ~0.80 = +2.40  (关节接近目标，含踝关节)
      trunk_level:        +3.0 × ~0.90 = +2.70  (头≈base≈0.19m)
      quadruped_stance:   +3.0 × ~0.8  = +2.4   (四肢撑地)
      limb_flat:          +3.0 × ~0.85 = +2.55  (掌面朝下)
      foot_flat_ground:   +2.5 × ~0.8  = +2.0   (脚掌平踩+接触力)
      hand_ground:        +3.0 × ~0.7  = +2.1   (双手撑地)
      hand_forward_reach: +1.5 × ~0.8  = +1.2   (手在前方支撑)
      orientation_prone:  -0.3 × ~0.05 = -0.015 (姿态接近目标)
      anti_backward:      -1.0 × 0.0   = 0.0    (前倾，不后仰)
      trunk_contact_pen:  -0.5 × 0.0   = 0.0    (躯干悬空)
      alive_bonus:        +0.5 × 1.0   = +0.5
      ─────────────────────────────────────────
      净奖励 ≈ +17.8/step（比v5.1的11.3更高，四肢接触贡献显著）
    """
    
    # ================================================================
    # 1. 核心驱动力: 线性高度奖励（替代v4的指数衰减版）
    # ================================================================
    # v5改进：分段线性，站立到目标之间有持续均匀梯度
    # - 站立(0.302m): reward=0.0
    # - 每降低1cm: reward增加约0.089
    # - 目标(0.19m): reward=1.0
    # - 过低(<0.10m): reward为负值（内置惩罚，不需要单独的height_floor）
    height_linear = RewTerm(
        func=height_linear_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.19,
            "standing_height": 0.302,
            "min_height": 0.10,
        },
    )
    
    # ================================================================
    # 2. 关节角度目标引导（v5核心：替代方向引导）
    # ================================================================
    # 精确定义四足姿态的每个关键关节角度
    # 无高度门控！站立时关节全在0，离目标远，reward≈0.14
    # 每个关节向目标移动都会增加奖励，提供持续梯度
    #
    # V3 四足姿态关节角度（从URDF运动学推算）：
    # - 髋关节前屈：RHIPp=-1.2(axis=-X), LHIPp=+1.2(axis≈+X)
    # - 膝关节弯曲：RKNEEP=+1.0(axis=-X), LKNEEp=-1.0(axis=+X)
    # - 踝关节补偿：RANKLEp/LANKLEp(axis=-X)，补偿膝弯曲使脚掌平踩
    #   四足姿态下小腿后倾，踝关节需要约+0.2rad补偿使脚底朝下
    #   RANKLEp限位[-0.43, 0.06]，LANKLEp限位[-0.43, 0.06]
    #   axis=-X: 正值=背屈(脚尖抬起)，负值=跖屈(脚尖下压)
    #   四足时需要轻微背屈来补偿小腿角度: 约-0.2rad
    # - 肩关节前伸：RSDp=+1.2(axis=+X), LSDp=+1.2(axis≈+X)
    # - 肘关节弯曲：RARMp=-0.5(axis=+X), LARMp=+0.5(axis=-X)
    # - 躯干水平：Waist_2≈-1.57（头部从竖直翻转到水平前伸）
    joint_pose_target = RewTerm(
        func=joint_pose_target_reward,
        weight=3.0,  # 最强信号：精确定义目标姿态
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_joints": {
                "RHIPp": -1.2,    # 右髋前屈
                "LHIPp": 1.2,     # 左髋前屈
                "RKNEEP": 1.0,    # 右膝弯曲
                "LKNEEp": -1.0,   # 左膝弯曲
                "RANKLEp": -0.2,  # v5.2新增：右踝补偿，脚掌平踩地面
                "LANKLEp": -0.2,  # v5.2新增：左踝补偿，脚掌平踩地面
                "RSDp": 1.2,      # 右肩前伸
                "LSDp": 1.2,      # 左肩前伸
                "RARMp": -0.5,    # 右肘弯曲
                "LARMp": 0.5,     # 左肘弯曲
                "Waist_2": -1.57, # 头部水平前伸（从竖直→水平）
            },
        },
    )
    
    # ================================================================
    # 3. 躯干水平+目标高度约束（v5增强版）
    # ================================================================
    # 双重约束：
    # 1. Waist_2(头)高度 ≈ base_link高度（躯干平行）
    # 2. Waist_2(头)高度 ≈ 0.19m（和base_link一起在目标高度）
    # reward = level_reward × height_reward，只有同时满足才给高分
    trunk_level = RewTerm(
        func=trunk_level_reward,
        weight=3.0,  # v5.1增强：从1.5→3.0，躯干水平是核心目标
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "head_body_name": "Waist_2",
            "target_height": 0.19,     # 头部目标高度=0.19m
            "sigma_level": 0.03,       # 水平约束：3cm容差
            "sigma_height": 0.03,      # 高度约束：3cm容差
        },
    )
    
    # ================================================================
    # 4. 四足支撑奖励（移除高度门控！增强权重！）
    # ================================================================
    # v5.1增强：权重从2.0→3.0，更强调四肢（特别是双手）撑地
    # 双臂必须支撑在地面上，不能悬空
    quadruped_stance = RewTerm(
        func=quadruped_stance_reward,
        weight=3.0,  # v5.1增强：从2.0→3.0，四肢撑地是核心
        params={
            "left_hand_sensor_cfg": SceneEntityCfg("contact_forces_left_hand"),
            "right_hand_sensor_cfg": SceneEntityCfg("contact_forces_right_hand"),
            "left_foot_sensor_cfg": SceneEntityCfg("contact_forces_left_foot"),
            "right_foot_sensor_cfg": SceneEntityCfg("contact_forces_right_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "force_threshold": 10.0,
            "height_threshold": 999.0,  # v5: 无门控！
            "hand_body_names": ["LARMAy", "RARMAy"],
            "foot_body_names": ["LANKLEy", "RANKLEy"],
            "ground_z_threshold": 0.03,
            "weight_force_threshold": 20.0,
        },
    )

    # ================================================================
    # 4b. 肢端掌面朝下约束（v5.1新增，v5.2增强！）
    # ================================================================
    # 确保手掌/脚掌的掌面朝下，完全踩在地面上
    # 而不是用关节侧面或背面接触地面
    # 检查每个肢端body的局部Z轴方向：掌面朝下时 world_z_z ≈ -1
    limb_flat = RewTerm(
        func=limb_flat_on_ground_reward,
        weight=3.0,  # v5.2增强：从2.0→3.0，掌面朝下是核心约束
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hand_body_names": ["LARMAy", "RARMAy"],
            "foot_body_names": ["LANKLEy", "RANKLEy"],
        },
    )

    # ================================================================
    # 4c. 脚掌平踩地面奖励（v5.2新增！）
    # ================================================================
    # 综合接触力+方向约束：脚掌必须有力且掌面朝下
    # 与 limb_flat 的区别：这里同时要求接触力，确保脚真的踩在地上
    # 而不只是方向正确但悬空
    foot_flat_ground = RewTerm(
        func=foot_flat_on_ground_reward,
        weight=2.5,  # 强约束：脚掌必须平踩
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "left_foot_sensor_cfg": SceneEntityCfg("contact_forces_left_foot"),
            "right_foot_sensor_cfg": SceneEntityCfg("contact_forces_right_foot"),
            "foot_body_names": ["LANKLEy", "RANKLEy"],
            "force_threshold": 5.0,
            "height_threshold": 0.25,  # 低高度时才激活
        },
    )

    # ================================================================
    # 4d. 手掌撑地奖励（v5.2新增！）
    # ================================================================
    # 专门强调双手必须向前撑住地面
    # 综合：接触力 × 贴地高度 × 前方位置 × 掌面朝下
    # 双手分数相乘：必须双手都撑地才能拿高分
    hand_ground = RewTerm(
        func=hand_ground_contact_reward,
        weight=3.0,  # 最强约束：双手必须撑地！
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "left_hand_sensor_cfg": SceneEntityCfg("contact_forces_left_hand"),
            "right_hand_sensor_cfg": SceneEntityCfg("contact_forces_right_hand"),
            "hand_body_names": ["LARMAy", "RARMAy"],
            "force_threshold": 5.0,
            "height_threshold": 0.30,  # 下蹲后就激活
        },
    )

    # ================================================================
    # 5. 手臂前伸引导（移除高度门控！增强权重！）
    # ================================================================
    # v5.1增强：权重从0.5→1.5，双臂前伸支撑是关键动作
    # 手臂必须向前伸出并支撑在地面上
    hand_forward_reach = RewTerm(
        func=hand_forward_reach_reward,
        weight=1.5,  # v5.1增强：从0.5→1.5，手臂支撑是关键
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hand_body_names": ["LARMAy", "RARMAy"],
            "height_threshold": 999.0,  # v5: 无门控！
        },
    )

    # ================================================================
    # 6. 姿态约束（保留高度门控！）
    # ================================================================
    # v5修正：orientation_prone 必须保留高度门控！
    # 原因：站立时 gravity_b.z=-1，z_err=0.85，惩罚=0.72
    # 这个惩罚驱动"让z趋向0"，但前倒和后倒都能做到
    # 策略选择后倒（更容易），导致向后摔倒
    #
    # 解决：姿态约束只在接近目标高度时激活（h<0.25m）
    # 在高处靠 joint_pose_target 和 height_linear 引导方向
    orientation_prone = RewTerm(
        func=orientation_prone_measure,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_pitch": 0.42,
            "z_tol": 0.15,
            "y_tol": 0.15,
            "height_threshold": 0.25,  # 保留门控：只在低高度时约束姿态
        },
    )
    
    # 反后仰保护（v5新增）：轻量级，只惩罚后仰不奖励前倾
    # gravity_b.y < 0 表示后仰 → 惩罚
    # gravity_b.y >= 0 表示前倾或站立 → 不惩罚
    # 与之前的 forward_lean 不同：这里只是"防护栏"，不是"方向引导"
    # 不会产生"越前倾越好"的捷径
    anti_backward = RewTerm(
        func=anti_backward_fall_reward,
        weight=-1.0,  # 中等惩罚：后仰是错误方向
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "forward_axis": 1,  # Y轴=前方
        },
    )

    # ================================================================
    # 7. 稳定性与安全约束
    # ================================================================
    
    # 朝向保持（防止侧向摔倒/打转）
    base_heading = RewTerm(
        func=base_heading_reward,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 躯干接触惩罚（v5降权：从-2.0→-0.5，允许探索）
    # v4的-2.0太大（站立正奖励才+0.115），策略极度恐惧探索
    # v5正奖励约+0.92，-0.5的惩罚比例合理
    trunk_contact_pen = RewTerm(
        func=trunk_contact_penalty,
        weight=-0.5,  # v5降低：让策略敢于探索
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_trunk"),
            "threshold": 5.0,
        },
    )
    
    # 软着陆（保持不变）
    soft_landing = RewTerm(
        func=soft_landing_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 200.0},
    )
    
    # 躯干稳定性（保持不变）
    base_stability = RewTerm(
        func=base_stability_reward,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 动作平滑（保持不变）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    
    # 活着奖励（v5增大：从0.1→0.5，鼓励存活探索）
    alive_bonus = RewTerm(func=mdp.is_alive, weight=0.5)


# ============================================================
# 终止条件配置
# ============================================================

@configclass
class V3StandToProneTerminationsCfg:
    """终止条件配置"""
    
    # 超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 注释掉base_contact终止条件，改为奖励中的惩罚
    # 这样机器人不会因为腹部蹭地而立刻重置，避免"恐地症"
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    # )

# ============================================================
# 场景配置: 初始站立
# ============================================================

@configclass
class V3StandToProneSceneCfg(InteractiveSceneCfg):
    # 地面
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
             static_friction=1.0, friction_combine_mode="average",
        ),
    )
    
    # 机器人: 初始设置为标准站立
    robot: ArticulationCfg = V3_HUMANOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.302), # V3站立质心高度0.302m（实测数据）
            rot=(1.0, 0.0, 0.0, 0.0), # 直立
            joint_pos={
                # 所有关节使用默认姿态（从URDF读取）
                # 如果默认姿态不是站立，需要在V3_HUMANOID_CFG中设置
                # ".*": 0.0,  # 不要强制所有关节为0，使用URDF默认值
            },
        ),
    )
    
    # 传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3
    )
    
    # 肢端接触传感器：只检测手掌末端和脚底末端
    # 关键修复：收紧传感器范围，防止用膝盖/前臂"凑够四点接触"
    #
    # V3 URDF link 结构：
    #   右臂: RSDp→RSDy→RSDr→RARMp→RARMAP→RARMAy (末端)
    #   左臂: LSDp→LSDy→LSDr→LARMp→LARMAp→LARMAy (末端)
    #   右腿: RHIPp→RHIPy→RHIPr→RKNEEP→RANKLEp→RANKLEy (末端)
    #   左腿: LHIPp→LHIPy→LHIPr→LKNEEp→LANKLEp→LANKLEy (末端)
    #
    # 手部传感器：只检测手腕末端 (LARMAy/RARMAy) + 前一节 (LARMAp/RARMAP)
    contact_forces_left_hand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LARMA(y|p)",  # 只检测左手末端两节
        history_length=3,
        track_air_time=True,
    )
    contact_forces_right_hand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RARMA(y|P)",  # 只检测右手末端两节 (注意P大写)
        history_length=3,
        track_air_time=True,
    )
    
    # 脚部传感器：只检测脚踝末端 (LANKLEy/RANKLEy) + 前一节 (LANKLEp/RANKLEp)
    contact_forces_left_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LANKLE(y|p)",  # 只检测左脚末端两节
        history_length=3,
        track_air_time=True,
    )
    contact_forces_right_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RANKLE(y|p)",  # 只检测右脚末端两节
        history_length=3,
        track_air_time=True,
    )
    
    # 躯干接触传感器（v4新增）：检测 base_link + Waist + Waist_2
    # 四足支撑时，躯干任何部位都不应该触地
    # 之前只检测base_link，导致机器人用头（Waist_2）顶地
    contact_forces_trunk = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base_link|Waist|Waist_2)",
        history_length=3,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


# ============================================================
# 观测与动作 (保持标准)
# ============================================================

@configclass
class V3ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        # 增加: 当前高度信息，让它知道自己离地多远
        base_height = ObsTerm(func=base_height_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class V3ActionsCfg:
    """
    动作配置：平衡运动范围与控制稳定性
    
    关键修复：scale从0.25增大到0.75，允许关节移动±43度
    - 0.75 rad ≈ 43°，足够完成趴下动作（髋关节需要弯曲60-90度）
    - 相比1.0（57°）更保守，避免过度激进的动作导致不稳定
    - 如果发现运动范围不足，可以增大到1.0
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.75,  # 平衡运动范围与稳定性
        use_default_offset=True
    )

# ============================================================
# 环境入口
# ============================================================

@configclass
class V3StandToProneEnvCfg(ManagerBasedRLEnvCfg):
    """
    站立到趴下 (Stand-to-Prone) 训练环境
    """
    sim = SimulationCfg(dt=0.005, render_interval=4)
    scene: V3StandToProneSceneCfg = V3StandToProneSceneCfg(num_envs=4096, env_spacing=2.5)
    
    observations: V3ObservationsCfg = V3ObservationsCfg()
    actions: V3ActionsCfg = V3ActionsCfg()
    rewards: V3StandToProneRewardsCfg = V3StandToProneRewardsCfg()
    
    # 终止条件：防止机器人飞出边界或翻倒
    terminations: V3StandToProneTerminationsCfg = V3StandToProneTerminationsCfg()
    
    # 这是一个一次性动作任务，不需要 continuous locomotion command
    # 但为了让它不动，我们给零指令或者干脆不通过 Command 驱动，完全靠 Reward 驱动
    # 这里也可以留空 commands
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0 # 短时间任务，5秒足够趴下
        self.decimation = 4

@configclass
class V3StandToProneEnvCfg_PLAY(V3StandToProneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


