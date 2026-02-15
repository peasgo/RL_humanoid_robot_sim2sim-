"""Print Isaac Lab joint order for the V4 robot."""
import argparse
import sys
import os

# Add Isaac Lab to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab_assets.robots.v4_quadruped import V4_QUADRUPED_CFG

# Create a simple scene
sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)

# Spawn robot
robot_cfg: ArticulationCfg = V4_QUADRUPED_CFG.replace(prim_path="/World/Robot")
robot_cfg.init_state.pos = (0.0, 0.0, 0.35)
robot = Articulation(cfg=robot_cfg)

sim_utils.build_simulation_scene()
sim.reset()
robot.update(sim.cfg.dt)

print("\n" + "="*60)
print("Isaac Lab Joint Order:")
print("="*60)
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

print(f"\nTotal joints: {robot.num_joints}")
print(f"\nDefault joint pos: {robot.data.default_joint_pos[0]}")

# Also check action joint order
from isaaclab_tasks.manager_based.locomotion.velocity.config.v4_quadruped.flat_env_cfg import ActionsCfg
print(f"\nAction joint names (from config): {ActionsCfg.joint_pos.joint_names}")

# Find action joint indices
action_joint_ids, action_joint_names = robot.find_joints(ActionsCfg.joint_pos.joint_names)
print(f"\nAction joint IDs: {action_joint_ids}")
print(f"Action joint names (resolved): {action_joint_names}")

simulation_app.close()
