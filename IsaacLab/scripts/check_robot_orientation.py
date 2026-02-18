"""Quick script to check V6 robot orientation in Isaac Sim.
Run: isaaclab -p scripts/check_robot_orientation.py
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.math import quat_apply, yaw_quat
from isaaclab_assets.robots.v6_humanoid import V6_HUMANOID_CFG

sim_cfg = SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)

scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)
scene_cfg.robot = V6_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
scene_cfg.ground = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(),
)

scene = InteractiveScene(scene_cfg)
sim.reset()
scene.reset()

# Step a few times to let physics settle
for _ in range(100):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_cfg.dt)

robot = scene["robot"]
quat = robot.data.root_quat_w[0]  # (w, x, y, z)
pos = robot.data.root_pos_w[0]

# Get the forward direction (X axis of body frame in world)
forward_body = torch.tensor([[1.0, 0.0, 0.0]], device=quat.device)
forward_world = quat_apply(quat.unsqueeze(0), forward_body)

# Get the left direction (Y axis of body frame in world)
left_body = torch.tensor([[0.0, 1.0, 0.0]], device=quat.device)
left_world = quat_apply(quat.unsqueeze(0), left_body)

# Get the up direction (Z axis of body frame in world)
up_body = torch.tensor([[0.0, 0.0, 1.0]], device=quat.device)
up_world = quat_apply(quat.unsqueeze(0), up_body)

print(f"\n{'='*60}")
print(f"V6 Robot Orientation Check")
print(f"{'='*60}")
print(f"Root position (world): {pos.cpu().numpy()}")
print(f"Root quaternion (w,x,y,z): {quat.cpu().numpy()}")
print(f"")
print(f"Body X axis in world: {forward_world[0].cpu().numpy()}")
print(f"Body Y axis in world: {left_world[0].cpu().numpy()}")
print(f"Body Z axis in world: {up_world[0].cpu().numpy()}")
print(f"")
print(f"track_lin_vel_xy_yaw_frame_exp uses yaw_quat to extract heading.")
print(f"In heading frame: vel[:,0] = along body X, vel[:,1] = along body Y")
print(f"")
print(f"So command.lin_vel_x (forward) maps to body X direction in world: {forward_world[0].cpu().numpy()}")
print(f"And command.lin_vel_y (lateral) maps to body Y direction in world: {left_world[0].cpu().numpy()}")
print(f"")

# Check which direction the robot is visually facing
# If body X points along world +X, forward command = world +X
# If body X points along world +Y, forward command = world +Y
fx, fy, fz = forward_world[0].cpu().numpy()
if abs(fx) > abs(fy):
    print(f"Robot body X axis is mostly along world {'+'if fx>0 else '-'}X")
else:
    print(f"Robot body X axis is mostly along world {'+'if fy>0 else '-'}Y")

print(f"")
print(f"If the robot faces -Y in URDF and we rotated +90 around Z:")
print(f"  URDF -Y -> world +X (forward command should move robot in this direction)")
print(f"  URDF +X -> world +Y (lateral command should move robot in this direction)")
print(f"{'='*60}")

sim.stop()
