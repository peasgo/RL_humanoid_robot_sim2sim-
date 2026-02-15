"""Minimal script to print Isaac Lab joint order for V4 robot.
Run: python scripts/print_joint_order.py --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))

# Spawn ground plane
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)

robot_cfg = V4_QUADRUPED_CFG.replace(prim_path="/World/envs/env_0/Robot")
robot_cfg.init_state.pos = (0.0, 0.0, 0.35)
robot = Articulation(cfg=robot_cfg)

sim.reset()
robot.update(sim.cfg.dt)

print("\n" + "=" * 60)
print("Isaac Lab Joint Names (PhysX DOF order):")
print("=" * 60)
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")
print(f"\nTotal joints: {robot.num_joints}")
print(f"Default joint pos: {robot.data.default_joint_pos[0].cpu().numpy()}")
print("=" * 60)

simulation_app.close()
