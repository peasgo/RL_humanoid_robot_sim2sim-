#!/usr/bin/env python3
"""Check action-to-joint mapping correctness.

The policy outputs 16 actions in Isaac16 order.
These need to be mapped to MuJoCo actuators correctly.

Also check: does the PD controller in MuJoCo match IsaacLab's?
IsaacLab uses: torque = kp * (target - current) - kd * velocity
MuJoCo position actuator: torque = kp * (target - current) - kd * velocity

Also check: are the actuator gains correct?
"""
import numpy as np
import mujoco
import yaml
import os

np.set_printoptions(precision=6, suppress=True, linewidth=150)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
CONFIG_YAML = os.path.join(SCRIPT_DIR, "v4_robot.yaml")

ISAAC17 = ['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2','LSDp','RSDp',
           'LKNEEp','RKNEEP','LSDy','RSDy','LANKLEp','RANKLEp',
           'LARMp','RARMp','LARMAp','RARMAP']
ISAAC16 = [n for n in ISAAC17 if n != 'Waist_2']

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

default_angles = np.array(cfg["default_angles"], dtype=np.float64)
kps = np.array(cfg["kps"], dtype=np.float64)
kds = np.array(cfg["kds"], dtype=np.float64)

# Get MuJoCo joint names
mj_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_names.append(jname)

print("="*100)
print("MuJoCo Joint Order (17 joints):")
print("="*100)
for i, name in enumerate(mj_names):
    print(f"  [{i:2d}] {name:12s}  default={default_angles[i]:+.4f}  kp={kps[i]:.0f}  kd={kds[i]:.0f}")

print("\n" + "="*100)
print("Isaac17 Order:")
print("="*100)
i17_to_mj = []
for i, name in enumerate(ISAAC17):
    mj_idx = mj_names.index(name)
    i17_to_mj.append(mj_idx)
    print(f"  Isaac17[{i:2d}] = {name:12s} → MuJoCo[{mj_idx:2d}]  default={default_angles[mj_idx]:+.4f}")

print("\n" + "="*100)
print("Isaac16 (Action) Order:")
print("="*100)
i16_to_mj = []
for i, name in enumerate(ISAAC16):
    mj_idx = mj_names.index(name)
    i16_to_mj.append(mj_idx)
    print(f"  Isaac16[{i:2d}] = {name:12s} → MuJoCo[{mj_idx:2d}]  default={default_angles[mj_idx]:+.4f}")

print("\n" + "="*100)
print("MuJoCo Actuator Mapping:")
print("="*100)
act_to_joint = []
for i in range(m.nu):
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    mj_idx = mj_names.index(joint_name)
    act_to_joint.append(mj_idx)
    
    # Get actuator gains from model
    act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    gainprm = m.actuator_gainprm[i]
    biasprm = m.actuator_biasprm[i]
    
    print(f"  Actuator[{i:2d}] = {act_name:20s} → Joint[{mj_idx:2d}] {joint_name:12s}")
    print(f"    gainprm: {gainprm[:3]}")
    print(f"    biasprm: {biasprm[:3]}")

# Check the action application logic
print("\n" + "="*100)
print("Action Application Test:")
print("="*100)
print("\nTest: apply action=[1,0,0,...,0] (only first Isaac16 joint)")
print(f"  Isaac16[0] = {ISAAC16[0]} → MuJoCo[{i16_to_mj[0]}]")
print(f"  target = action[0] * action_scale + default = 1.0 * 0.25 + {default_angles[i16_to_mj[0]]:.4f} = {1.0 * 0.25 + default_angles[i16_to_mj[0]]:.4f}")

# Now check: in the run_v4_robot.py, how is the action applied?
# target_dof_pos[mj_idx] = action_isaac16[i16] * action_scale + default_angles[mj_idx]
# d.ctrl[:] = target_dof_pos[act_to_joint]

# The ctrl is indexed by ACTUATOR, and maps to JOINT via act_to_joint
# So ctrl[actuator_i] = target_dof_pos[act_to_joint[actuator_i]]

print("\n" + "="*100)
print("Verify ctrl assignment:")
print("="*100)
print("\nFor each actuator, what joint does it control?")
for i in range(m.nu):
    act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_idx = act_to_joint[i]
    joint_name = mj_names[joint_idx]
    print(f"  ctrl[{i:2d}] → target_dof_pos[{joint_idx:2d}] ({joint_name:12s})")

# Check if actuator order matches joint order
print("\n  Actuator order == Joint order?", act_to_joint == list(range(len(mj_names))))

# Now let's check the training config for PD gains
print("\n" + "="*100)
print("Training PD Gains (from flat_env_cfg.py):")
print("="*100)

# From the training config, the stiffness and damping are set per joint
# Let me check the env.yaml
env_yaml_path = os.path.join(SCRIPT_DIR,
    "../logs/rsl_rl/v4_quadruped_flat/2026-02-11_12-17-13/params/env.yaml")
with open(env_yaml_path) as f:
    env_cfg = yaml.safe_load(f)

# Find actuator config
if 'scene' in env_cfg and 'robot' in env_cfg['scene']:
    robot_cfg = env_cfg['scene']['robot']
    if 'actuators' in robot_cfg:
        for act_name, act_cfg in robot_cfg['actuators'].items():
            print(f"\n  {act_name}:")
            if 'stiffness' in act_cfg:
                print(f"    stiffness: {act_cfg['stiffness']}")
            if 'damping' in act_cfg:
                print(f"    damping: {act_cfg['damping']}")
            if 'joint_names_expr' in act_cfg:
                print(f"    joints: {act_cfg['joint_names_expr']}")
