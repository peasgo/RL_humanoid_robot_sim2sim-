"""
Render IsaacLab Step 1 pose in MuJoCo offscreen and save as image.
No viewer needed - saves PNG files from multiple angles.
"""
import mujoco
import numpy as np
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "v6_robot.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = os.path.join(current_dir, config["xml_path"])
default_angles = np.array(config["default_angles"], dtype=np.float32)
isaac_joint_order = config["isaac_joint_order"]

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = config["simulation_dt"]

# Joint mappings
mj_joint_names = []
for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        mj_joint_names.append(jname)

isaac_to_mujoco = np.array([mj_joint_names.index(j) for j in isaac_joint_order], dtype=np.int32)
default_angles_isaac = default_angles[isaac_to_mujoco].copy()

# IsaacLab Step 1 joint positions
isaac_jpos_rel_step1 = np.array([
    -0.0438, -0.0435,  0.0053, -0.0576, -0.0920, -0.0718,
    -0.0708, -0.1467, -0.0751,  0.0455,  0.0816, -0.1083,
     0.0174
], dtype=np.float32)
isaac_jpos_step1 = isaac_jpos_rel_step1 + default_angles_isaac

# Set robot state
d.qpos[0:3] = [0, 0, 0.55]
d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
for i_isaac in range(len(isaac_jpos_step1)):
    mj_idx = isaac_to_mujoco[i_isaac]
    d.qpos[7 + mj_idx] = isaac_jpos_step1[i_isaac]
d.qvel[:] = 0

mujoco.mj_forward(m, d)

# Offscreen rendering - use framebuffer-safe size
width, height = 640, 480
renderer = mujoco.Renderer(m, height, width)

# Camera views
views = [
    ("front", [1.5, 0.0, 0.8], [0, 0, 0.4]),
    ("side_right", [0.0, -1.5, 0.8], [0, 0, 0.4]),
    ("side_left", [0.0, 1.5, 0.8], [0, 0, 0.4]),
    ("back", [-1.5, 0.0, 0.8], [0, 0, 0.4]),
    ("top", [0.0, 0.0, 2.5], [0, 0, 0.0]),
    ("front_close", [0.8, 0.3, 0.6], [0, 0, 0.35]),
]

output_dir = os.path.join(current_dir, "pose_screenshots")
os.makedirs(output_dir, exist_ok=True)

for name, eye, target in views:
    cam = mujoco.MjvCamera()
    # Compute lookat and distance
    eye = np.array(eye)
    target = np.array(target)
    diff = eye - target
    dist = np.linalg.norm(diff)
    
    cam.lookat[:] = target
    cam.distance = dist
    cam.azimuth = np.degrees(np.arctan2(diff[1], diff[0]))
    cam.elevation = np.degrees(np.arcsin(diff[2] / dist))
    
    renderer.update_scene(d, cam)
    img = renderer.render()
    
    filepath = os.path.join(output_dir, f"step1_{name}.png")
    
    # Save using PIL or raw
    try:
        from PIL import Image
        Image.fromarray(img).save(filepath)
    except ImportError:
        # Fallback: save raw with matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.imsave(filepath, img)
        except ImportError:
            # Last resort: save as raw numpy
            np.save(filepath.replace('.png', '.npy'), img)
            filepath = filepath.replace('.png', '.npy')
    
    print(f"  Saved: {filepath}")

renderer.close()

print(f"\nAll screenshots saved to: {output_dir}/")
print("Compare these with the IsaacLab viewer pose.")
