#!/usr/bin/env python3
"""Headless sim2sim test: run policy for 20 steps and print diagnostics."""
import mujoco, numpy as np, yaml, torch, sys

m = mujoco.MjModel.from_xml_path('v6_scene.xml')
d = mujoco.MjData(m)

with open('v6_robot.yaml') as f:
    config = yaml.safe_load(f)

default_angles = np.array(config['default_angles'], dtype=np.float64)
isaac_joint_order = config['isaac_joint_order']
action_scale = config['action_scale']
ang_vel_scale = config['ang_vel_scale']
dof_pos_scale = config['dof_pos_scale']
dof_vel_scale = config['dof_vel_scale']
clip_actions = config['clip_actions']
clip_obs = 100.0
control_decimation = config['control_decimation']
action_smoothing = config.get('action_smoothing', 0.0)

mj_joint_names = []
for i in range(m.njnt):
    if m.jnt_type[i] == 3:
        mj_joint_names.append(m.joint(i).name)

isaac_to_mujoco = [mj_joint_names.index(n) for n in isaac_joint_order]
default_angles_isaac = np.array([default_angles[isaac_to_mujoco[i]] for i in range(13)])

actuator_to_joint = []
for i in range(m.nu):
    trnid = m.actuator_trnid[i, 0]
    jname = m.joint(trnid).name
    actuator_to_joint.append(mj_joint_names.index(jname))
actuator_to_joint = np.array(actuator_to_joint)

policy = torch.jit.load(config['policy_path'])
policy.eval()

mujoco.mj_resetData(m, d)
for i in range(len(default_angles)):
    d.qpos[7 + i] = default_angles[i]
mujoco.mj_forward(m, d)

target_dof_pos = default_angles.copy()
action_raw = np.zeros(13, dtype=np.float32)
action_applied = np.zeros(13, dtype=np.float32)

# Warmup
for _ in range(200):
    d.ctrl[:] = target_dof_pos[actuator_to_joint]
    mujoco.mj_step(m, d)

print(f'After warmup: h={d.qpos[2]:.4f}  quat={d.qpos[3:7]}')

for step in range(20):
    quat = d.qpos[3:7].copy()
    omega_body = d.qvel[3:6].copy()
    
    w, x, y, z = quat
    gx = -2*(x*z - w*y)
    gy = -2*(y*z + w*x)
    gz = -(1 - 2*(x*x + y*y))
    
    qj_isaac = np.array([d.qpos[7 + isaac_to_mujoco[i]] for i in range(13)])
    dqj_isaac = np.array([d.qvel[6 + isaac_to_mujoco[i]] for i in range(13)])
    
    obs = np.zeros(48, dtype=np.float32)
    obs[0:3] = omega_body * ang_vel_scale
    obs[3:6] = [gx, gy, gz]
    obs[6:9] = [0.0, 0.0, 0.0]
    obs[9:22] = (qj_isaac - default_angles_isaac) * dof_pos_scale
    obs[22:35] = dqj_isaac * dof_vel_scale
    obs[35:48] = action_raw
    obs = np.clip(obs, -clip_obs, clip_obs)
    
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    action_raw = policy(obs_tensor).detach().numpy().squeeze()
    action_raw = np.clip(action_raw, -clip_actions, clip_actions)
    
    if action_smoothing > 0:
        action_applied = (1 - action_smoothing) * action_raw + action_smoothing * action_applied
    else:
        action_applied = action_raw.copy()
    
    for i_isaac in range(13):
        mj_idx = isaac_to_mujoco[i_isaac]
        target_dof_pos[mj_idx] = action_applied[i_isaac] * action_scale + default_angles[mj_idx]
    
    print(f'Step {step:2d}: h={d.qpos[2]:.4f}  grav=[{gx:+.3f},{gy:+.3f},{gz:+.3f}]  act_max={np.max(np.abs(action_raw)):.3f}  ncon={d.ncon}')
    if step < 3:
        print(f'  obs[0:6]={obs[0:6]}')
        print(f'  obs[9:22]={obs[9:22]}')
        act_str = np.array2string(action_raw, precision=3, separator=', ')
        print(f'  action={act_str}')
    
    for _ in range(control_decimation):
        d.ctrl[:] = target_dof_pos[actuator_to_joint]
        mujoco.mj_step(m, d)

print(f'\nFinal: h={d.qpos[2]:.4f}')
