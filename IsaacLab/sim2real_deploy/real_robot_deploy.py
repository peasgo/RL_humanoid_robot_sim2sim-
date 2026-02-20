#!/usr/bin/env python3
"""
V6 Humanoid Sim2Real 部署脚本

观测空间严格匹配 IsaacLab flat_env_cfg.py 的 ObservationsCfg:
  obs[0:3]   = base_ang_vel * 0.2        (body frame 角速度)
  obs[3:6]   = projected_gravity          (body frame 重力投影)
  obs[6:9]   = velocity_commands          [vx, vy, yaw_rate]
  obs[9:22]  = (joint_pos - default) * 1.0 (关节位置偏差, PhysX BFS 顺序)
  obs[22:35] = joint_vel * 0.05           (关节速度, PhysX BFS 顺序)
  obs[35:48] = last_action                (上一步动作)

动作应用:
  target_pos = action * 0.25 + default_angles

参考: IsaacLab/mujoco_deploy_v6_humanoid/run_v6_humanoid.py (已验证的 sim2sim 逻辑)
"""

import time
import numpy as np
import torch
import yaml
import os
import argparse
import signal
import sys


# ============================================================
# 重力投影计算
# 将世界坐标系重力 [0, 0, -1] 投影到 body frame
# 公式: g_body = R^T @ [0, 0, -1]
# 与 run_v6_humanoid.py 中的实现完全一致
# ============================================================
def get_gravity_orientation(quaternion):
    """计算 body frame 中的重力投影向量。

    Args:
        quaternion: [w, x, y, z] 格式的四元数 (body-to-world 旋转)

    Returns:
        [gx, gy, gz] 重力在 body frame 中的投影

    Note:
        V6 机器人站直时 (init_state.rot = [0.7071068, 0, 0, 0.7071068]):
        由于 URDF Y-up → IsaacLab Z-up 的旋转，重力投影应为 [0, -1, 0]
    """
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz], dtype=np.float32)


# ============================================================
# 硬件通信接口 (抽象基类)
# ⚠️ 用户需要根据实际硬件实现具体的子类
# ============================================================
class HardwareInterface:
    """硬件通信抽象接口。

    用户需要继承此类并实现以下方法:
    - connect(): 建立硬件连接
    - disconnect(): 断开连接
    - read_imu(): 读取 IMU 数据 (四元数 + 角速度)
    - read_joint_states(): 读取关节状态 (位置 + 速度)
    - send_joint_commands(): 发送关节位置/力矩指令
    """

    def connect(self):
        raise NotImplementedError("请实现 connect() 方法")

    def disconnect(self):
        raise NotImplementedError("请实现 disconnect() 方法")

    def read_imu(self):
        """读取 IMU 数据。

        Returns:
            quaternion: np.array [w, x, y, z] — body-to-world 旋转四元数
            angular_velocity: np.array [wx, wy, wz] — body frame 角速度 (rad/s)
        """
        raise NotImplementedError("请实现 read_imu() 方法")

    def read_joint_states(self):
        """读取所有关节状态。

        Returns:
            positions: np.array (num_motors,) — 各电机当前角度 (rad), 按 motor_id 顺序
            velocities: np.array (num_motors,) — 各电机当前角速度 (rad/s), 按 motor_id 顺序
        """
        raise NotImplementedError("请实现 read_joint_states() 方法")

    def send_joint_commands(self, motor_ids, positions, kps, kds, torques=None):
        """发送关节位置指令 (PD 控制)。

        Args:
            motor_ids: list[int] — 目标电机 CAN ID 列表
            positions: np.array — 目标位置 (rad)
            kps: np.array — 位置增益
            kds: np.array — 速度增益
            torques: np.array or None — 前馈力矩 (可选)
        """
        raise NotImplementedError("请实现 send_joint_commands() 方法")


class DummyHardware(HardwareInterface):
    """虚拟硬件接口，用于测试部署逻辑（不连接实际硬件）。"""

    def __init__(self, num_joints):
        self.num_joints = num_joints
        self._joint_pos = np.zeros(num_joints, dtype=np.float32)
        self._joint_vel = np.zeros(num_joints, dtype=np.float32)
        # 默认站直姿态的四元数 (与 IsaacLab init_state 一致)
        self._quat = np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
        self._ang_vel = np.zeros(3, dtype=np.float32)

    def connect(self):
        print("[DummyHardware] 虚拟硬件已连接 (测试模式)")

    def disconnect(self):
        print("[DummyHardware] 虚拟硬件已断开")

    def read_imu(self):
        return self._quat.copy(), self._ang_vel.copy()

    def read_joint_states(self):
        return self._joint_pos.copy(), self._joint_vel.copy()

    def send_joint_commands(self, motor_ids, positions, kps, kds, torques=None):
        # 简单模拟: 关节位置直接跟踪目标
        for i, mid in enumerate(motor_ids):
            if mid < self.num_joints:
                self._joint_pos[mid] = positions[i]


# ============================================================
# 主部署类
# ============================================================
class V6HumanoidDeployer:
    """V6 人形机器人 sim2real 部署控制器。

    观测构建逻辑严格匹配 IsaacLab flat_env_cfg.py 的 ObservationsCfg,
    与 MuJoCo sim2sim (run_v6_humanoid.py) 完全一致。
    """

    def __init__(self, config_path, hardware=None, dry_run=False):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dry_run = dry_run

        # --- 策略维度 ---
        self.num_actions = self.config["num_actions"]  # 13
        self.num_obs = self.config["num_obs"]          # 48

        # --- 关节顺序 (PhysX BFS) ---
        self.isaac_joint_order = self.config["isaac_joint_order"]
        assert len(self.isaac_joint_order) == self.num_actions, \
            f"isaac_joint_order 长度 {len(self.isaac_joint_order)} != num_actions {self.num_actions}"

        # --- 默认关节角度 (PhysX BFS 顺序) ---
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        assert len(self.default_angles) == self.num_actions

        # --- 缩放参数 ---
        self.action_scale = float(self.config["action_scale"])      # 0.25
        self.ang_vel_scale = float(self.config["ang_vel_scale"])    # 0.2
        self.dof_pos_scale = float(self.config["dof_pos_scale"])    # 1.0
        self.dof_vel_scale = float(self.config["dof_vel_scale"])    # 0.05

        # --- PD 增益 (PhysX BFS 顺序) ---
        self.kp = np.array(self.config["kp"], dtype=np.float32)
        self.kd = np.array(self.config["kd"], dtype=np.float32)
        assert len(self.kp) == self.num_actions
        assert len(self.kd) == self.num_actions

        # --- 电机映射 ---
        self.motor_ids = self.config["motor_ids"]       # isaac[i] → CAN ID
        self.joint_sign = np.array(self.config["joint_sign"], dtype=np.float32)
        assert len(self.motor_ids) == self.num_actions
        assert len(self.joint_sign) == self.num_actions

        # --- IMU 坐标变换矩阵 ---
        imu_rot_flat = self.config["imu_rotation"]
        self.imu_rotation = np.array(imu_rot_flat, dtype=np.float32).reshape(3, 3)

        # --- 安全限制 ---
        self.torque_limit = float(self.config["torque_limit"])
        self.velocity_limit = float(self.config["velocity_limit"])
        self.clip_obs = float(self.config.get("clip_obs", 100.0))
        self.clip_actions = float(self.config.get("clip_actions", 100.0))

        # --- 控制频率 ---
        self.control_dt = float(self.config["control_dt"])  # 0.02s = 50Hz

        # --- 速度指令 ---
        self.cmd = np.array(self.config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

        # --- 状态变量 ---
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        # --- 构建 motor_id → isaac_index 的反向映射 ---
        # motor_ids[isaac_idx] = can_id
        # 我们需要: 从 CAN ID 读回的数据 → PhysX BFS 顺序
        self.motor_id_to_isaac_idx = {}
        for isaac_idx, can_id in enumerate(self.motor_ids):
            self.motor_id_to_isaac_idx[can_id] = isaac_idx

        # --- 加载策略 ---
        policy_path = self.config["policy_path"]
        if policy_path == "FILL_IN_YOUR_POLICY_PATH":
            print("⚠️  警告: policy_path 未设置，将以无策略模式运行 (PD 保持默认姿态)")
            self.policy = None
        elif os.path.exists(policy_path):
            print(f"加载策略: {policy_path}")
            self.policy = torch.jit.load(policy_path)
            print("策略加载成功!")
        else:
            print(f"⚠️  警告: 策略文件不存在 {policy_path}，将以无策略模式运行")
            self.policy = None

        # --- 硬件接口 ---
        if hardware is not None:
            self.hw = hardware
        elif dry_run:
            self.hw = DummyHardware(self.num_actions)
        else:
            # 用户需要在这里实例化实际的硬件接口
            raise RuntimeError(
                "未提供硬件接口! 请实现 HardwareInterface 子类并传入，"
                "或使用 --dry-run 进行测试。"
            )

    def safety_check_imu(self, gravity_vec):
        """启动时检查 IMU 重力向量是否合理。

        机器人站直时，重力投影应接近 [0, -1, 0]（V6 URDF Y-up 特性）。
        如果偏差过大，说明 IMU 安装方向或 imu_rotation 配置有误。
        """
        expected = np.array([0.0, -1.0, 0.0])
        error = np.linalg.norm(gravity_vec - expected)
        magnitude = np.linalg.norm(gravity_vec)

        print(f"  IMU 重力投影: {gravity_vec}")
        print(f"  重力向量模长: {magnitude:.4f} (应接近 1.0)")
        print(f"  与期望值 [0,-1,0] 的偏差: {error:.4f}")

        if abs(magnitude - 1.0) > 0.1:
            print("  ❌ 重力向量模长异常! 请检查 IMU 数据。")
            return False

        if error > 0.5:
            print("  ⚠️  重力投影与期望值偏差较大!")
            print("     可能原因: 1) 机器人未站直  2) imu_rotation 配置错误  3) IMU 安装方向不对")
            print("     请确认后继续...")
            return True  # 警告但不阻止

        print("  ✅ IMU 重力检查通过")
        return True

    def transform_imu_data(self, quat_imu, ang_vel_imu):
        """将 IMU 数据从传感器坐标系变换到 URDF body frame。

        Args:
            quat_imu: [w, x, y, z] IMU 原始四元数
            ang_vel_imu: [wx, wy, wz] IMU 原始角速度

        Returns:
            quat_body: [w, x, y, z] body frame 四元数
            ang_vel_body: [wx, wy, wz] body frame 角速度
        """
        R = self.imu_rotation  # IMU frame → body frame

        # 如果 imu_rotation 是单位矩阵，直接返回
        if np.allclose(R, np.eye(3)):
            return quat_imu, ang_vel_imu

        # 变换角速度: omega_body = R @ omega_imu
        ang_vel_body = R @ ang_vel_imu

        # 变换四元数: 需要将 IMU 的旋转矩阵左乘 R
        # q_body = q_R * q_imu (四元数乘法)
        # 这里简化处理: 先转旋转矩阵，变换后再转回四元数
        from scipy.spatial.transform import Rotation
        R_imu = Rotation.from_quat([quat_imu[1], quat_imu[2], quat_imu[3], quat_imu[0]])  # scipy: [x,y,z,w]
        R_transform = Rotation.from_matrix(R)
        R_body = R_transform * R_imu
        q_scipy = R_body.as_quat()  # [x, y, z, w]
        quat_body = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=np.float32)  # [w,x,y,z]

        return quat_body, ang_vel_body.astype(np.float32)

    def read_robot_state(self):
        """从硬件读取机器人状态，并转换到 PhysX BFS 顺序。

        Returns:
            quat: [w,x,y,z] body frame 四元数
            ang_vel: [wx,wy,wz] body frame 角速度 (rad/s)
            joint_pos: (13,) PhysX BFS 顺序的关节位置 (rad)
            joint_vel: (13,) PhysX BFS 顺序的关节速度 (rad/s)
        """
        # 读取 IMU
        quat_raw, ang_vel_raw = self.hw.read_imu()
        quat, ang_vel = self.transform_imu_data(quat_raw, ang_vel_raw)

        # 读取关节状态 (按 motor_id 顺序返回)
        motor_pos, motor_vel = self.hw.read_joint_states()

        # 将电机数据映射到 PhysX BFS 顺序，并应用 joint_sign
        joint_pos = np.zeros(self.num_actions, dtype=np.float32)
        joint_vel = np.zeros(self.num_actions, dtype=np.float32)

        for isaac_idx in range(self.num_actions):
            can_id = self.motor_ids[isaac_idx]
            sign = self.joint_sign[isaac_idx]
            joint_pos[isaac_idx] = motor_pos[can_id] * sign
            joint_vel[isaac_idx] = motor_vel[can_id] * sign

        return quat, ang_vel, joint_pos, joint_vel

    def build_observation(self, ang_vel, gravity, cmd, joint_pos, joint_vel):
        """构建观测向量，严格匹配 IsaacLab ObservationsCfg。

        观测布局 (48 dims):
          [0:3]   base_ang_vel * ang_vel_scale
          [3:6]   projected_gravity
          [6:9]   velocity_commands [vx, vy, yaw_rate]
          [9:22]  (joint_pos - default_angles) * dof_pos_scale
          [22:35] joint_vel * dof_vel_scale
          [35:48] last_action

        与 run_v6_humanoid.py 的观测构建完全一致。
        """
        obs = self.obs

        # obs[0:3]: base_ang_vel * scale
        # IsaacLab: ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        obs[0:3] = ang_vel * self.ang_vel_scale

        # obs[3:6]: projected_gravity (body frame)
        # IsaacLab: ObsTerm(func=mdp.projected_gravity)
        obs[3:6] = gravity

        # obs[6:9]: velocity_commands
        # IsaacLab: ObsTerm(func=mdp.generated_commands)
        obs[6:9] = cmd

        # obs[9:22]: joint_pos_rel = (joint_pos - default) * dof_pos_scale
        # IsaacLab: ObsTerm(func=mdp.joint_pos_rel)
        obs[9:22] = (joint_pos - self.default_angles) * self.dof_pos_scale

        # obs[22:35]: joint_vel * dof_vel_scale
        # IsaacLab: ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        obs[22:35] = joint_vel * self.dof_vel_scale

        # obs[35:48]: last_action (策略原始输出，未缩放)
        # IsaacLab: ObsTerm(func=mdp.last_action)
        obs[35:48] = self.last_action

        # 裁剪观测 (rsl_rl 默认行为)
        np.clip(obs, -self.clip_obs, self.clip_obs, out=obs)

        return obs

    def apply_action(self, action):
        """将策略输出转换为电机指令并发送。

        Args:
            action: (13,) 策略原始输出 (PhysX BFS 顺序)
        """
        # 裁剪动作 (rsl_rl 默认行为)
        action = np.clip(action, -self.clip_actions, self.clip_actions)

        # 保存为 last_action (下一步观测需要)
        self.last_action[:] = action

        # 计算目标关节角度: target = action * scale + default
        # 与 IsaacLab JointPositionActionCfg(scale=0.25, use_default_offset=True) 一致
        target_pos_isaac = action * self.action_scale + self.default_angles

        # 转换到电机指令: 应用 joint_sign 并通过 motor_ids 映射
        motor_targets = np.zeros(self.num_actions, dtype=np.float32)
        motor_kps = np.zeros(self.num_actions, dtype=np.float32)
        motor_kds = np.zeros(self.num_actions, dtype=np.float32)

        for isaac_idx in range(self.num_actions):
            sign = self.joint_sign[isaac_idx]
            motor_targets[isaac_idx] = target_pos_isaac[isaac_idx] * sign
            motor_kps[isaac_idx] = self.kp[isaac_idx]
            motor_kds[isaac_idx] = self.kd[isaac_idx]

        # 安全检查: 限制速度
        # (实际部署时应在硬件层面也做限制)

        # 发送指令
        self.hw.send_joint_commands(
            motor_ids=self.motor_ids,
            positions=motor_targets,
            kps=motor_kps,
            kds=motor_kds,
        )

    def run(self):
        """主控制循环。"""
        print("\n" + "=" * 60)
        print(f"V6 Humanoid Sim2Real 部署")
        print(f"  num_actions:   {self.num_actions}")
        print(f"  num_obs:       {self.num_obs}")
        print(f"  action_scale:  {self.action_scale}")
        print(f"  ang_vel_scale: {self.ang_vel_scale}")
        print(f"  dof_pos_scale: {self.dof_pos_scale}")
        print(f"  dof_vel_scale: {self.dof_vel_scale}")
        print(f"  control_dt:    {self.control_dt}s ({1.0/self.control_dt:.0f}Hz)")
        print(f"  policy:        {'已加载' if self.policy else '无 (PD 保持)'}")
        print(f"  dry_run:       {self.dry_run}")
        print(f"  cmd:           {self.cmd}")
        print("=" * 60)

        # 连接硬件
        self.hw.connect()

        # --- 启动安全检查 ---
        print("\n--- 启动安全检查 ---")
        try:
            quat, ang_vel, joint_pos, joint_vel = self.read_robot_state()
            gravity = get_gravity_orientation(quat)
            if not self.safety_check_imu(gravity):
                print("IMU 检查失败，退出!")
                self.hw.disconnect()
                return

            print(f"\n  当前关节位置 (PhysX BFS 顺序):")
            for i, name in enumerate(self.isaac_joint_order):
                print(f"    [{i:2d}] {name:14s}  pos={joint_pos[i]:+.4f}  "
                      f"default={self.default_angles[i]:+.4f}  "
                      f"diff={joint_pos[i]-self.default_angles[i]:+.4f}")
        except Exception as e:
            print(f"启动检查失败: {e}")
            self.hw.disconnect()
            return

        print("\n--- 安全检查完成，开始控制循环 ---")
        print("按 Ctrl+C 停止\n")

        # --- 主循环 ---
        running = True
        step_count = 0

        def signal_handler(sig, frame):
            nonlocal running
            running = False
            print("\n收到停止信号，正在安全退出...")

        signal.signal(signal.SIGINT, signal_handler)

        try:
            while running:
                loop_start = time.time()

                # 1. 读取机器人状态
                quat, ang_vel, joint_pos, joint_vel = self.read_robot_state()

                # 2. 计算重力投影
                gravity = get_gravity_orientation(quat)

                # 3. 构建观测
                obs = self.build_observation(
                    ang_vel=ang_vel,
                    gravity=gravity,
                    cmd=self.cmd,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                )

                # 4. 策略推理或 PD 保持
                if self.policy is not None:
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    with torch.no_grad():
                        action = self.policy(obs_tensor).detach().numpy().squeeze()
                else:
                    # 无策略: 输出零动作 → 目标 = default_angles
                    action = np.zeros(self.num_actions, dtype=np.float32)

                # 5. 应用动作
                self.apply_action(action)

                # 6. 周期性状态打印
                if step_count % 50 == 0:  # 每秒打印一次 (50Hz)
                    t = step_count * self.control_dt
                    print(f"[t={t:6.1f}s] gravity={gravity} "
                          f"cmd=({self.cmd[0]:+.2f},{self.cmd[1]:+.2f},{self.cmd[2]:+.2f}) "
                          f"act_max={np.max(np.abs(self.last_action)):.3f}")

                step_count += 1

                # 7. 控制频率同步
                elapsed = time.time() - loop_start
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif step_count % 100 == 0:
                    print(f"  ⚠️  控制循环超时: {elapsed*1000:.1f}ms > {self.control_dt*1000:.1f}ms")

        except Exception as e:
            print(f"\n控制循环异常: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 安全停止: 发送默认姿态指令
            print("\n正在发送默认姿态指令...")
            try:
                zero_action = np.zeros(self.num_actions, dtype=np.float32)
                self.apply_action(zero_action)
                time.sleep(0.1)
            except Exception:
                pass

            self.hw.disconnect()
            print(f"部署结束，共运行 {step_count} 步 ({step_count * self.control_dt:.1f}s)")


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="V6 Humanoid Sim2Real 部署")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_robot_config.yaml"),
        help="配置文件路径 (默认: 同目录下的 real_robot_config.yaml)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="干跑模式: 使用虚拟硬件测试部署逻辑"
    )
    parser.add_argument(
        "--cmd", type=float, nargs=3, default=None,
        metavar=("VX", "VY", "YAW"),
        help="初始速度指令 [vx, vy, yaw_rate]"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)

    deployer = V6HumanoidDeployer(
        config_path=args.config,
        dry_run=args.dry_run,
    )

    if args.cmd is not None:
        deployer.cmd[:] = args.cmd

    deployer.run()


if __name__ == "__main__":
    main()
