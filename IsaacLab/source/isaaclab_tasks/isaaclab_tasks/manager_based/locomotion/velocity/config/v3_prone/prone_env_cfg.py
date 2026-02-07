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
    target_pitch: float = -0.6,
    z_tol: float = 0.15,
    y_tol: float = 0.15,
    height_threshold: float = 0.25,
) -> torch.Tensor:
    """
    四足姿态引导奖励函数（带高度门控）
    
    关键修复：
    1. 只有当高度 < height_threshold 时才激活姿态约束
    2. 避免站立阶段因"没趴平"而被过度惩罚，导致策略梯度不稳定
    
    V3机器人坐标系（从URDF和Isaac Sim确认）：
    - X轴：指向右侧（红色箭头）
    - Y轴：指向前方（绿色箭头）
    - Z轴：指向上方（蓝色箭头）
    
    站立时：X右，Y前，Z上
    趴下时：X右，Y前下倾，Z水平
    
    引导机器人从双足站立变为四足姿态，需要：
    1. 躯干Z轴水平（垂直于重力方向）- gravity_body.z ≈ 0
    2. 躯干Y轴前倾约30-45度 - gravity_body.y ≈ -0.6
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
        target_pitch: 目标前倾角度对应的重力分量，默认-0.6（约37度前倾）
                     -0.5对应30度，-0.7对应45度，-1.0对应90度（完全趴平）
        z_tol: Z轴容差（允许偏差范围）
        y_tol: Y轴容差（允许偏差范围）
        height_threshold: 高度门控阈值，只有低于此高度才激活姿态约束
    
    Returns:
        姿态偏差惩罚值，越小越好（站立阶段返回0）
    """
    robot = env.scene[asset_cfg.name]
    root_quat = robot.data.root_quat_w
    root_pos = robot.data.root_pos_w
    
    # 高度门控：只有蹲下来了，才开始在意是不是趴平了
    height = root_pos[:, 2]
    mask = (height < height_threshold).float()
    
    # 获取世界坐标系下的重力向量
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device)
    gravity_world = gravity_world.unsqueeze(0).expand(root_quat.shape[0], -1)
    
    # 转换到机器人本体坐标系
    from isaaclab.utils.math import quat_apply_inverse
    gravity_body = quat_apply_inverse(root_quat, gravity_world)
    
    # 使用"区间内不惩罚，区间外再惩罚"的 hinge 形式，训练更稳定
    #
    # 约束逻辑（V3坐标系：X右，Y前，Z上）：
    # 1. Z轴应该水平（gravity_body.z ≈ 0）
    z_err = torch.clamp(torch.abs(gravity_body[:, 2]) - z_tol, min=0.0)
    
    # 2. Y轴应该前倾（gravity_body.y ≈ target_pitch）
    #    Y轴指向前方，前倾时gravity_body.y为负值
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
    """
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    current_height = base_pos[:, 2]
    error = torch.abs(current_height - target_height)
    outside = torch.clamp(error - tol, min=0.0, max=5.0)
    reward = torch.where(error <= tol, torch.ones_like(error), torch.exp(-torch.square(outside) / 0.002))
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

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

def base_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    躯干稳定性奖励: 惩罚质心水平速度和角速度
    
    关键修复：
    - 去掉前向速度的额外惩罚（导致后仰）
    - 只惩罚水平速度（X和Y），允许垂直方向（Z）的速度（下蹲需要）
    - 各方向对称惩罚，不偏向任何方向
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
    
    Returns:
        稳定性惩罚值（越小越好）
    """
    robot = env.scene[asset_cfg.name]
    
    # 只惩罚水平速度（X和Y），允许垂直方向（Z）的速度
    # V3坐标系：X右，Y前，Z上
    # 下蹲时Z轴速度是正常的，不应该惩罚
    lin_vel = robot.data.root_lin_vel_b
    horizontal_vel_penalty = torch.square(torch.clamp(lin_vel[:, 0], min=-10.0, max=10.0)) + \
                             torch.square(torch.clamp(lin_vel[:, 1], min=-10.0, max=10.0))
    
    # 惩罚角速度（所有方向）
    ang_vel = robot.data.root_ang_vel_b
    ang_vel_penalty = torch.sum(torch.square(torch.clamp(ang_vel, min=-10.0, max=10.0)), dim=1)
    
    reward = horizontal_vel_penalty + ang_vel_penalty
    # 数值保护：裁剪到合理范围
    reward = torch.clamp(reward, min=0.0, max=100.0)
    return torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=0.0)

# ============================================================
# 奖励配置: 站立 -> 趴下
# ============================================================

@configclass
class V3StandToProneRewardsCfg:
    
    # --- 1. 核心驱动力: 下降 ---
    # 引导质心高度降至 0.19m (四足姿态高度)
    # 实测数据：站立质心0.302m，四足姿态约为站立的60-65%，即0.18-0.20m
    #
    # 关键修复：收紧容差从0.05到0.02，避免"站桩"问题
    # - 站立误差0.112m，如果tol=0.05且衰减慢，站着也能拿高分
    # - 收紧后，站立时奖励≈exp(-(0.112-0.02)²/0.002)≈exp(-4.23)≈0.015（几乎为0）
    reach_target_height = RewTerm(
        func=height_progress_reward,
        weight=0.5,  # 降低权重，避免梯度过大
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.19, "tol": 0.02},
    )
    
    # --- 2. 姿态引导: 变平（放宽高度门控） ---
    # 关键修复：放宽高度门控到0.35m（接近站立高度0.302m）
    # 这样机器人在站立阶段就能感受到"向前倾"的引导信号
    # 避免后仰摔倒（因为没有方向引导）
    orientation_prone = RewTerm(
        func=orientation_prone_measure,
        weight=-0.15,  # 适中权重
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_pitch": -0.6,
            "z_tol": 0.15,
            "y_tol": 0.15,  # V3坐标系：Y轴指向前方（绿色箭头）
            "height_threshold": 0.35,  # 放宽门控：站立阶段就开始引导前倾
        },
    )

    # --- 3. 动作引导: 四足支撑（完全接地版） ---
    # 不仅检测四肢是否"碰到"地面，还要求"完全接地"：
    # - 肢端Z坐标贴地（< 0.03m）
    # - 接触力足够大（承重 > 20N）
    # - 手在质心前方，脚在质心后方
    # 每个肢端的接地质量是连续值[0,1]，总分/4作为奖励
    quadruped_stance = RewTerm(
        func=quadruped_stance_reward,
        weight=1.5,  # 强化四足支撑引导
        params={
            "left_hand_sensor_cfg": SceneEntityCfg("contact_forces_left_hand"),
            "right_hand_sensor_cfg": SceneEntityCfg("contact_forces_right_hand"),
            "left_foot_sensor_cfg": SceneEntityCfg("contact_forces_left_foot"),
            "right_foot_sensor_cfg": SceneEntityCfg("contact_forces_right_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "force_threshold": 10.0,       # 最小接触力阈值（N）
            "height_threshold": 0.25,      # 质心高度门控（m）
            "hand_body_names": ["LARMAy", "RARMAy"],   # 手掌末端body
            "foot_body_names": ["LANKLEy", "RANKLEy"], # 脚底末端body
            "ground_z_threshold": 0.03,    # 贴地Z阈值（m），越小要求越严格
            "weight_force_threshold": 20.0, # 承重力阈值（N）
        },
    )

    # --- 3b. 手臂前伸引导 ---
    # 根据参考轨迹，双足→四足的关键动作是手臂向前伸展
    # 手需要从身体侧面向前方伸出0.4-0.6m，然后着地形成前支撑
    # 这个奖励引导手部Y坐标超过质心Y坐标（即手在身体前方）
    hand_forward_reach = RewTerm(
        func=hand_forward_reach_reward,
        weight=0.3,  # 正奖励，鼓励手臂前伸
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hand_body_names": ["LARMAy", "RARMAy"],  # V3手臂末端body名称
            "height_threshold": 0.28,  # 略低于站立高度0.302m，开始下蹲就激活
        },
    )

    # --- 4. 稳定性与风格 ---
    # 朝向保持（防止侧向摔倒）
    base_heading = RewTerm(
        func=base_heading_reward,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 躯干接触惩罚（改为惩罚而非终止）
    base_contact_pen = RewTerm(
        func=base_contact_penalty,
        weight=-1.0,  # 较大惩罚，但不终止
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    
    # 软着陆
    soft_landing = RewTerm(
        func=soft_landing_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 200.0},
    )
    
    # 躯干稳定性（修正版）
    # 关键修复：
    # - 去掉前向速度额外惩罚（导致后仰）
    # - 只惩罚水平速度，允许垂直下蹲
    # - 权重适中，不要过度抑制运动
    base_stability = RewTerm(
        func=base_stability_reward,
        weight=-0.15,  # 适中权重，不要过度抑制运动
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # 动作平滑
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    
    # 活着
    alive_bonus = RewTerm(func=mdp.is_alive, weight=0.1)


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


