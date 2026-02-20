"""Dump V6 humanoid PhysX BFS joint order and default positions.

Run: cd IsaacLab && python scripts/dump_v6_isaac_obs.py --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.v6_humanoid import V6_HUMANOID_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))

cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)

robot_cfg = V6_HUMANOID_CFG.replace(prim_path="/World/envs/env_0/Robot")
robot = Articulation(cfg=robot_cfg)

sim.reset()
robot.update(sim.cfg.dt)

# ================================================================
# 1. Print PhysX BFS joint order
# ================================================================
print("\n" + "=" * 70)
print("V6 Humanoid - PhysX BFS Joint Order (robot.joint_names)")
print("=" * 70)
default_pos = robot.data.default_joint_pos[0].cpu().numpy()
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name:20s}  default={default_pos[i]:+.4f}")
print(f"\nTotal joints: {robot.num_joints}")

# ================================================================
# 2. Test find_joints with ActionsCfg joint_names
# ================================================================
action_joint_names = [
    "pelvis_link",
    "RHIPp", "RHIPy", "RHIPr", "RKNEEp", "RANKLEp", "RANKLEy",
    "LHIPp", "LHIPy", "LHIPr", "LKNEEp", "LANKLEp", "LANKLEy",
]

ids_false, names_false = robot.find_joints(action_joint_names, preserve_order=False)
ids_true, names_true = robot.find_joints(action_joint_names, preserve_order=True)

print(f"\n{'='*70}")
print("find_joints(preserve_order=False) — ACTUAL policy action/obs order:")
print("(This is what the trained policy uses)")
print("=" * 70)
for i, (jid, jname) in enumerate(zip(ids_false, names_false)):
    print(f"  policy[{i:2d}] -> joint_id={jid:2d} name={jname:20s} default={default_pos[jid]:+.4f}")

print(f"\n{'='*70}")
print("find_joints(preserve_order=True) — config list order:")
print("=" * 70)
for i, (jid, jname) in enumerate(zip(ids_true, names_true)):
    print(f"  config[{i:2d}] -> joint_id={jid:2d} name={jname:20s} default={default_pos[jid]:+.4f}")

# ================================================================
# 3. Compare with current YAML isaac_joint_order
# ================================================================
yaml_isaac_order = [
    'pelvis_link',
    'LHIPp', 'RHIPp',
    'LHIPy', 'RHIPy',
    'LHIPr', 'RHIPr',
    'LKNEEp', 'RKNEEp',
    'LANKLEp', 'RANKLEp',
    'LANKLEy', 'RANKLEy',
]

print(f"\n{'='*70}")
print("COMPARISON: YAML isaac_joint_order vs actual PhysX BFS")
print("=" * 70)
mismatch = False
for i, (yaml_name, actual_name) in enumerate(zip(yaml_isaac_order, names_false)):
    match = "✓" if yaml_name == actual_name else "✗ MISMATCH!"
    if yaml_name != actual_name:
        mismatch = True
    print(f"  [{i:2d}] YAML: {yaml_name:20s}  Actual: {actual_name:20s}  {match}")

if mismatch:
    print("\n*** MISMATCH DETECTED! ***")
    print("The YAML isaac_joint_order does NOT match the actual PhysX BFS order.")
    print("This is likely the cause of sim2sim failure!")
    print("\nCorrect isaac_joint_order for YAML:")
    print("isaac_joint_order:")
    for name in names_false:
        print(f"  - {name}")
    print("\naction_joint_order (same, since preserve_order=False):")
    print("action_joint_order:")
    for name in names_false:
        print(f"  - {name}")
else:
    print("\n✓ YAML isaac_joint_order matches actual PhysX BFS order.")

# ================================================================
# 4. Print MuJoCo-compatible mapping
# ================================================================
mj_order = [
    'pelvis_link',
    'RHIPp', 'RHIPy', 'RHIPr', 'RKNEEp', 'RANKLEp', 'RANKLEy',
    'LHIPp', 'LHIPy', 'LHIPr', 'LKNEEp', 'LANKLEp', 'LANKLEy',
]

print(f"\n{'='*70}")
print("isaac_to_mujoco mapping (for run_v6_humanoid.py):")
print("=" * 70)
for i_isaac, jname in enumerate(names_false):
    mj_idx = mj_order.index(jname)
    print(f"  isaac[{i_isaac:2d}] {jname:20s} -> mujoco[{mj_idx:2d}] {mj_order[mj_idx]}")

print(f"\n{'='*70}")
simulation_app.close()
