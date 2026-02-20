"""
Visual joint axis test for V6 Humanoid.
1. Sequentially rotates each joint to verify axis direction
2. Then applies IsaacLab Step 0 actions and lets it run
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "v6_robot.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
action_scale = config["action_scale"]
num_actions = config["num_actions"]
simulation_dt = config["simulation_dt"]
decimation = config["control_decimation"]
default_angles = np.array(config["default_angles"], dtype=np.float32)
isaac_joint_order = config["isaac_joint_order"]

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# MuJoCo joint names
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

isaac_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac_joint_order], dtype=np.int32)

# Actuator mapping
actuator_to_joint_indices = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    pd_index = mj_joint_names.index(joint_name)
    actuator_to_joint_indices.append(pd_index)
actuator_to_joint_indices = np.array(actuator_to_joint_indices, dtype=np.int32)

# IsaacLab Step 0 actions (Isaac order)
isaac_actions_step0 = np.array([
    -0.030991,  0.331074, -0.490743,  0.285347,  0.051951,
    -0.121331, -0.109601, -0.636742,  0.114688, -0.328023,
    -0.851936, -0.966040,  0.682626
], dtype=np.float32)

# IsaacLab Step 0 joint positions (Isaac order)
isaac_jpos_step0 = np.array([
    -0.055546, -0.257883, -0.166365, -0.082370, -0.118554,
    -0.080793, -0.088018, -0.534720, +0.314056, +0.268519,
    -0.060209, -0.091471, -0.017295
], dtype=np.float32)

def reset_robot(d, default_angles):
    d.qpos[0:3] = [0, 0, 0.55]
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0

print("=" * 60)
print("V6 Humanoid Joint Axis Visual Test")
print("=" * 60)
print("\nPhase 1: Sequentially rotate each joint (+0.5 rad)")
print("Phase 2: Apply IsaacLab Step 0 actions")
print("Phase 3: Free run with IsaacLab actions")
print()

with mujoco.viewer.launch_passive(m, d) as viewer:
    # ============================================================
    # Phase 1: Test each joint axis one by one
    # ============================================================
    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        
        # Reset to default
        reset_robot(d, default_angles)
        target = np.array(default_angles, dtype=np.float32)
        
        # Hold default for 0.5s
        for _ in range(int(0.5 / simulation_dt)):
            d.ctrl[:] = target[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
        viewer.sync()
        
        # Rotate this joint by +0.5 rad
        test_angle = 0.5
        target[mj_idx] = default_angles[mj_idx] + test_angle
        
        print(f"  [{i_isaac:2d}] {jname:14s}  MJ[{mj_idx}]  "
              f"default={default_angles[mj_idx]:+.2f} -> target={target[mj_idx]:+.2f}  "
              f"(+{test_angle} rad)")
        
        # Animate to target over 1.5s
        for step in range(int(1.5 / simulation_dt)):
            d.ctrl[:] = target[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if step % 10 == 0:
                viewer.sync()
        
        # Hold for 1s so user can see
        for _ in range(int(1.0 / simulation_dt)):
            d.ctrl[:] = target[actuator_to_joint_indices]
            mujoco.mj_step(m, d)
            if _ % 10 == 0:
                viewer.sync()
        
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print("Phase 2: Apply IsaacLab Step 0 pose")
    print("=" * 60)
    
    # Set to IsaacLab Step 0 joint positions
    reset_robot(d, default_angles)
    # Set actual joint positions from IsaacLab
    for i_isaac in range(num_actions):
        mj_idx = isaac_to_mujoco[i_isaac]
        d.qpos[7 + mj_idx] = isaac_jpos_step0[i_isaac]
    d.qvel[:] = 0
    
    # Compute PD targets from Step 0 actions
    target = np.array(default_angles, dtype=np.float32)
    for i_isaac in range(num_actions):
        mj_idx = isaac_to_mujoco[i_isaac]
        target[mj_idx] = isaac_actions_step0[i_isaac] * action_scale + default_angles[mj_idx]
    
    print("\nIsaacLab Step 0 pose set. Applying actions...")
    print("Joint targets (Isaac order):")
    for i_isaac, jname in enumerate(isaac_joint_order):
        mj_idx = isaac_to_mujoco[i_isaac]
        print(f"  [{i_isaac:2d}] {jname:14s}  pos={isaac_jpos_step0[i_isaac]:+.4f}  "
              f"action={isaac_actions_step0[i_isaac]:+.4f}  "
              f"target={target[mj_idx]:+.4f}")
    
    # Run for 5 seconds with these targets
    for step in range(int(5.0 / simulation_dt)):
        d.ctrl[:] = target[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        if step % 10 == 0:
            viewer.sync()
    
    print("\n" + "=" * 60)
    print("Phase 3: Free view - press Ctrl+C to exit")
    print("=" * 60)
    
    # Keep viewer open
    while viewer.is_running():
        d.ctrl[:] = target[actuator_to_joint_indices]
        mujoco.mj_step(m, d)
        if int(d.time / simulation_dt) % 10 == 0:
            viewer.sync()
        time_until_next = simulation_dt - (time.time() % simulation_dt)
        if time_until_next > 0:
            time.sleep(time_until_next * 0.5)

print("Done.")
