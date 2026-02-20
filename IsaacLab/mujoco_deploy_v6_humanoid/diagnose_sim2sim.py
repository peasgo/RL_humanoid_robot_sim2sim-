"""
V6 Humanoid Sim2Sim 诊断脚本
用途: 不加载policy, 只检查MuJoCo模型的基本行为和观测构建是否正确

运行: cd IsaacLab/mujoco_deploy_v6_humanoid && python diagnose_sim2sim.py v6_robot.yaml
"""
import mujoco
import numpy as np
import yaml
import os
import sys


def get_gravity_orientation(quaternion):
    """Convert quaternion [w,x,y,z] to gravity vector in body frame."""
    w, x, y, z = quaternion
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.array([gx, gy, gz])


def quat_to_rotmat(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix (body-to-world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "v6_robot.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    init_height = float(config.get("init_height", 0.55))
    isaac_joint_order = config.get("isaac_joint_order")
    action_joint_order = config.get("action_joint_order")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    # ================================================================
    # 1. 打印MuJoCo关节信息
    # ================================================================
    mj_joint_names = []
    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            mj_joint_names.append(jname)

    print("=" * 70)
    print("1. MuJoCo 关节顺序 (DFS, XML中的顺序)")
    print("=" * 70)
    for i, name in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        limited = m.jnt_limited[jid]
        lo, hi = m.jnt_range[jid] if limited else (0, 0)
        print(f"  MJ[{i:2d}] {name:14s}  default={default_angles[i]:+.4f}  range=[{lo:+.2f}, {hi:+.2f}]")

    # ================================================================
    # 2. 打印Isaac观测关节顺序
    # ================================================================
    print(f"\n{'=' * 70}")
    print("2. Isaac 观测关节顺序 (PhysX BFS)")
    print("=" * 70)
    isaac_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in isaac_joint_order],
        dtype=np.int32
    )
    default_angles_isaac = default_angles[isaac_to_mujoco].copy()

    for i, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i]
        print(f"  Isaac[{i:2d}] {jname:14s} -> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]:14s}  default={default_angles_isaac[i]:+.4f}")

    # ================================================================
    # 3. 打印Isaac动作关节顺序 (YAML中定义的, 但代码中未使用)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("3. Isaac 动作关节顺序 (YAML中定义, 代码中action_to_mujoco)")
    print("=" * 70)
    action_to_mujoco = np.array(
        [mj_joint_names.index(jname) for jname in action_joint_order],
        dtype=np.int32
    )
    for i, jname in enumerate(action_joint_order):
        mj_idx = action_to_mujoco[i]
        print(f"  Action[{i:2d}] {jname:14s} -> MJ[{mj_idx:2d}] {mj_joint_names[mj_idx]:14s}")

    # ================================================================
    # 4. 关键检查: 动作映射是否正确
    # ================================================================
    print(f"\n{'=' * 70}")
    print("4. 关键检查: run_v6_humanoid.py 中动作映射使用的是 isaac_to_mujoco (观测顺序)")
    print("   而不是 action_to_mujoco (动作顺序)")
    print("=" * 70)
    print(f"  isaac_joint_order == action_joint_order ? {isaac_joint_order == action_joint_order}")
    if isaac_joint_order != action_joint_order:
        print(f"\n  *** 警告: 两个顺序不同! ***")
        print(f"  代码中 action 使用 isaac_to_mujoco 映射 (观测顺序)")
        print(f"  如果 policy 输出是 PhysX BFS 顺序 (preserve_order=False), 这是正确的")
        print(f"  因为 preserve_order=False 时, action 和 obs 都是 PhysX BFS 顺序")
        print(f"\n  对比:")
        for i in range(len(isaac_joint_order)):
            obs_j = isaac_joint_order[i]
            act_j = action_joint_order[i]
            match = "✓" if obs_j == act_j else "✗ MISMATCH"
            print(f"    [{i:2d}] obs={obs_j:14s}  act={act_j:14s}  {match}")

    # ================================================================
    # 5. 检查初始姿态和重力
    # ================================================================
    print(f"\n{'=' * 70}")
    print("5. 初始姿态和重力检查")
    print("=" * 70)

    d.qpos[2] = init_height
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0

    mujoco.mj_forward(m, d)

    quat = d.qpos[3:7]
    gravity = get_gravity_orientation(quat)
    R = quat_to_rotmat(quat)

    print(f"  初始四元数 (w,x,y,z): {quat}")
    print(f"  旋转矩阵 R (body->world):")
    print(f"    {R[0]}")
    print(f"    {R[1]}")
    print(f"    {R[2]}")
    print(f"  重力向量 (body frame): {gravity}")
    print(f"  期望值 (站立): [0, 0, -1]")
    print(f"  误差: {np.linalg.norm(gravity - np.array([0, 0, -1])):.6f}")

    # 检查body frame的轴方向
    print(f"\n  Body frame 轴在世界坐标中的方向:")
    print(f"    body X -> world {R[:, 0]}  (应该是水平方向)")
    print(f"    body Y -> world {R[:, 1]}  (应该是水平方向)")
    print(f"    body Z -> world {R[:, 2]}  (应该是 [0,0,1] 即向上)")

    # ================================================================
    # 6. 检查角速度帧
    # ================================================================
    print(f"\n{'=' * 70}")
    print("6. 角速度帧检查 (MuJoCo qvel[3:6] 是 body frame 还是 world frame)")
    print("=" * 70)

    d.qvel[3:6] = [0.5, 0.3, 0.1]
    mujoco.mj_forward(m, d)

    # cvel[body_id] = [ang_vel_world(3), lin_vel_world(3)]
    # body 0 = world, body 1 = base_link
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cvel = d.cvel[base_body_id].copy()
    omega_world_from_cvel = cvel[0:3]
    omega_world_if_body = R @ d.qvel[3:6]

    print(f"  设置 qvel[3:6] = {d.qvel[3:6]}")
    print(f"  cvel omega (world frame): {omega_world_from_cvel}")
    print(f"  R @ qvel (如果是body frame): {omega_world_if_body}")
    print(f"  误差 (body frame假设): {np.linalg.norm(omega_world_from_cvel - omega_world_if_body):.6f}")
    print(f"  误差 (world frame假设): {np.linalg.norm(omega_world_from_cvel - d.qvel[3:6]):.6f}")

    if np.linalg.norm(omega_world_from_cvel - omega_world_if_body) < 1e-4:
        print(f"  结论: qvel[3:6] 是 BODY frame ✓ (直接使用, 匹配 IsaacLab base_ang_vel)")
    else:
        print(f"  结论: qvel[3:6] 是 WORLD frame ✗ (需要 R^T @ qvel 转换!)")

    d.qvel[:] = 0

    # ================================================================
    # 7. 检查PD控制器行为 - 默认姿态保持
    # ================================================================
    print(f"\n{'=' * 70}")
    print("7. PD控制器测试 - 默认姿态保持 (5秒)")
    print("=" * 70)

    d.qpos[0:3] = [0, 0, init_height]
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0

    actuator_to_joint = []
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pd_index = mj_joint_names.index(joint_name)
        actuator_to_joint.append(pd_index)
    actuator_to_joint = np.array(actuator_to_joint, dtype=np.int32)

    target_dof_pos = default_angles.copy()

    for step in range(int(5.0 / m.opt.timestep)):
        d.ctrl[:] = target_dof_pos[actuator_to_joint]
        mujoco.mj_step(m, d)

    height_after = d.qpos[2]
    quat_after = d.qpos[3:7]
    gravity_after = get_gravity_orientation(quat_after)
    qj_after = d.qpos[7:]
    qj_error = np.max(np.abs(qj_after - default_angles))

    print(f"  初始高度: {init_height:.4f}m")
    print(f"  5秒后高度: {height_after:.4f}m  (变化: {height_after - init_height:+.4f}m)")
    print(f"  5秒后四元数: {quat_after}")
    print(f"  5秒后重力: {gravity_after}  (期望 [0,0,-1])")
    print(f"  关节最大偏差: {qj_error:.6f} rad")
    print(f"  接触数: {d.ncon}")

    if height_after < 0.1:
        print(f"\n  *** 严重问题: 机器人倒了! 高度只有 {height_after:.4f}m ***")
        print(f"  可能原因:")
        print(f"    1. MuJoCo XML 模型有问题 (关节轴/body位置)")
        print(f"    2. 默认角度不对")
        print(f"    3. PD增益太低")
    elif height_after < init_height - 0.1:
        print(f"\n  *** 警告: 机器人下沉明显 ({height_after - init_height:+.4f}m) ***")
    else:
        print(f"\n  ✓ 机器人在默认姿态下能站稳")

    # 打印每个关节的位置偏差
    print(f"\n  关节位置偏差 (MuJoCo顺序):")
    for i, jname in enumerate(mj_joint_names):
        print(f"    [{i:2d}] {jname:14s}  target={default_angles[i]:+.4f}  actual={qj_after[i]:+.4f}  error={qj_after[i]-default_angles[i]:+.6f}")

    # ================================================================
    # 8. 检查脚底接触
    # ================================================================
    print(f"\n{'=' * 70}")
    print("8. 脚底接触检查")
    print("=" * 70)

    foot_bodies = ["RANKLEy", "LANKLEy"]
    for fb in foot_bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fb)
        if bid >= 0:
            pos = d.xpos[bid]
            print(f"  {fb}: world pos = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
        else:
            print(f"  {fb}: NOT FOUND in model!")

    print(f"\n  活跃接触点 ({d.ncon} 个):")
    for ci in range(d.ncon):
        c = d.contact[ci]
        geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
        geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
        print(f"    contact[{ci}]: {geom1_name} <-> {geom2_name}  pos=({c.pos[0]:+.4f}, {c.pos[1]:+.4f}, {c.pos[2]:.4f})  depth={c.dist:.6f}")

    # ================================================================
    # 9. 检查总质量和质心
    # ================================================================
    print(f"\n{'=' * 70}")
    print("9. 质量和质心")
    print("=" * 70)
    total_mass = sum(m.body_mass[i] for i in range(m.nbody))
    print(f"  总质量: {total_mass:.4f} kg")
    for bi in range(m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bi) or f"body{bi}"
        print(f"    [{bi:2d}] {bname:14s}  mass={m.body_mass[bi]:.4f} kg")

    com = d.subtree_com[0]
    print(f"\n  整体质心 (world): ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")

    # ================================================================
    # 10. 构建一帧观测并打印
    # ================================================================
    print(f"\n{'=' * 70}")
    print("10. 第一帧观测向量 (应与 IsaacLab step 0 对比)")
    print("=" * 70)

    # Reset to initial state
    d.qpos[0:3] = [0, 0, init_height]
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    quat = d.qpos[3:7]
    omega_body = d.qvel[3:6].copy()
    qj_mujoco = d.qpos[7:].copy()
    dqj_mujoco = d.qvel[6:].copy()

    qj_isaac = qj_mujoco[isaac_to_mujoco]
    dqj_isaac = dqj_mujoco[isaac_to_mujoco]

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    omega_obs = omega_body * ang_vel_scale
    gravity_obs = get_gravity_orientation(quat)
    cmd_obs = cmd * cmd_scale
    qj_rel = (qj_isaac - default_angles_isaac) * dof_pos_scale
    dqj_obs = dqj_isaac * dof_vel_scale
    last_act = np.zeros(config["num_actions"], dtype=np.float32)

    obs = np.concatenate([omega_obs, gravity_obs, cmd_obs, qj_rel, dqj_obs, last_act])

    print(f"  obs shape: {obs.shape}")
    print(f"  obs[0:3]   ang_vel:  {omega_obs}")
    print(f"  obs[3:6]   gravity:  {gravity_obs}")
    print(f"  obs[6:9]   cmd:      {cmd_obs}")
    print(f"  obs[9:22]  qj_rel:   {qj_rel}")
    print(f"  obs[22:35] dqj:      {dqj_obs}")
    print(f"  obs[35:48] last_act: {last_act}")
    print(f"\n  完整 obs = {np.array2string(obs, precision=6, separator=', ', max_line_width=120)}")

    print(f"\n  *** 请用 play_forward_backward.py 在 IsaacLab 中运行, 对比 step 0 的观测 ***")
    print(f"  *** 如果观测不匹配, 说明关节映射或坐标系有问题 ***")

    # ================================================================
    # 11. 检查 actuator 力矩限制是否合理
    # ================================================================
    print(f"\n{'=' * 70}")
    print("11. Actuator 力矩限制")
    print("=" * 70)
    for i in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_id = m.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        kp = m.actuator_gainprm[i, 0]
        kv = m.actuator_biasprm[i, 2]  # for position actuator, biasprm[2] = -kv
        frange = m.actuator_forcerange[i]
        print(f"  [{i:2d}] {aname:14s} -> {jname:14s}  kp={kp:.0f}  kv={abs(kv):.0f}  forcerange=[{frange[0]:.0f}, {frange[1]:.0f}]")

    print(f"\n{'=' * 70}")
    print("诊断完成")
    print("=" * 70)
