"""Print Isaac Lab joint order for V6 humanoid robot.
Run: cd IsaacLab && python scripts/print_v6_joint_order.py --headless
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

print("\n" + "=" * 60)
print("V6 Humanoid - Isaac Lab Joint Names (PhysX DOF order):")
print("=" * 60)
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")
print(f"\nTotal joints: {robot.num_joints}")
print(f"Default joint pos: {robot.data.default_joint_pos[0].cpu().numpy()}")

# Also test what find_joints returns with the ActionsCfg joint_names
action_joint_names = [
    "pelvis_link",
    "RHIPp", "RHIPy", "RHIPr", "RKNEEp", "RANKLEp", "RANKLEy",
    "LHIPp", "LHIPy", "LHIPr", "LKNEEp", "LANKLEp", "LANKLEy",
]
ids, names = robot.find_joints(action_joint_names, preserve_order=False)
print(f"\nfind_joints(preserve_order=False) result:")
for i, (jid, jname) in enumerate(zip(ids, names)):
    print(f"  action[{i:2d}] -> joint_id={jid:2d} name={jname}")

ids2, names2 = robot.find_joints(action_joint_names, preserve_order=True)
print(f"\nfind_joints(preserve_order=True) result:")
for i, (jid, jname) in enumerate(zip(ids2, names2)):
    print(f"  action[{i:2d}] -> joint_id={jid:2d} name={jname}")

# Print the isaac_joint_order that MuJoCo code should use
print(f"\n{'='*60}")
print("For MuJoCo deployment, isaac_joint_order should be:")
print("(This is the order policy sees for both actions and observations)")
print("=" * 60)
print("isaac_joint_order = [")
for name in names:
    print(f"    '{name}',")
print("]")
print("=" * 60)

simulation_app.close()
