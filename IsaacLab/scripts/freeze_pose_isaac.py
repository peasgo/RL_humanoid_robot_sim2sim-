"""Set V6 humanoid to exact IsaacLab Step 1 joint positions and freeze the viewer.

Usage:
  cd /home/rl/RL-human_robot/IsaacLab
  ./isaaclab.sh -p scripts/freeze_pose_isaac.py
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Freeze V6 humanoid at a specific pose in IsaacLab viewer.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Import robot config
from isaaclab_assets.robots.v6_humanoid import V6_HUMANOID_CFG

# ================================================================
# IsaacLab Step 1 joint positions (Isaac BFS order)
# From obs_step1[9:22] + default_angles
# ================================================================
# Isaac BFS order: pelvis_link, LHIPp, RHIPp, LHIPy, RHIPy, LHIPr, RHIPr,
#                  LKNEEp, RKNEEp, LANKLEp, RANKLEp, LANKLEy, RANKLEy
isaac_default = np.array([
    0.0,    # pelvis_link
    -0.2,   # LHIPp
    -0.2,   # RHIPp
    0.0,    # LHIPy
    0.0,    # RHIPy
    0.0,    # LHIPr
    0.0,    # RHIPr
    -0.4,   # LKNEEp
    0.4,    # RKNEEp
    0.2,    # LANKLEp
    -0.2,   # RANKLEp
    0.0,    # LANKLEy
    0.0,    # RANKLEy
], dtype=np.float32)

isaac_jpos_rel_step1 = np.array([
    -0.0438, -0.0435,  0.0053, -0.0576, -0.0920, -0.0718,
    -0.0708, -0.1467, -0.0751,  0.0455,  0.0816, -0.1083,
     0.0174
], dtype=np.float32)

isaac_jpos_step1 = isaac_jpos_rel_step1 + isaac_default

joint_names_bfs = [
    "pelvis_link", "LHIPp", "RHIPp", "LHIPy", "RHIPy", "LHIPr", "RHIPr",
    "LKNEEp", "RKNEEp", "LANKLEp", "RANKLEp", "LANKLEy", "RANKLEy"
]

print("=" * 60)
print("Target pose (IsaacLab Step 1):")
print("=" * 60)
for i, jname in enumerate(joint_names_bfs):
    print(f"  [{i:2d}] {jname:14s}  pos={isaac_jpos_step1[i]:+.6f}")

# ================================================================
# Setup simulation
# ================================================================
sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device or "cuda:0")
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=[2.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])

# Spawn ground plane
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)

# Spawn robot
robot_cfg = V6_HUMANOID_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Play sim to initialize
sim.reset()

# Get joint name to index mapping
robot_joint_names = robot.joint_names
print(f"\nRobot joint names: {robot_joint_names}")

# Build target joint positions tensor
target_pos = torch.zeros(1, len(robot_joint_names), device=sim.device)
for i, jname in enumerate(joint_names_bfs):
    if jname in robot_joint_names:
        idx = robot_joint_names.index(jname)
        target_pos[0, idx] = float(isaac_jpos_step1[i])
        print(f"  Setting {jname} (idx={idx}) = {isaac_jpos_step1[i]:+.6f}")

# Set joint positions
robot.write_joint_state_to_sim(target_pos, torch.zeros_like(target_pos))
robot.update(sim.cfg.dt)

# Step once to render
sim.step()
robot.update(sim.cfg.dt)

print(f"\n{'='*60}")
print(f"Robot frozen at IsaacLab Step 1 pose.")
print(f"Use mouse to zoom/rotate in the viewer.")
print(f"Close the window to exit.")
print(f"{'='*60}\n")

# Keep viewer alive
while simulation_app.is_running():
    # Don't step physics, just keep rendering
    sim.render()
    time.sleep(0.05)

simulation_app.close()
