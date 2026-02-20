"""Headless test of V6 humanoid MuJoCo deploy — dump first N policy steps' actions."""
import numpy as np
import torch
import mujoco
import yaml
import os
import sys
import argparse
from collections import deque

class TermHistory:
    def __init__(self, max_len, term_dim):
        self.max_len = max_len
        self.term_dim = term_dim
        self._buf = deque(maxlen=max_len)
        for _ in range(max_len):
            self._buf.append(np.zeros(term_dim, dtype=np.float32))
    def append(self, data):
        self._buf.append(data.astype(np.float32))
    def flatten(self):
        return np.concatenate(list(self._buf))

def get_gravity_orientation(quat):
    w, x, y, z = quat
    gx = -2*(x*z - w*y)
    gy = -2*(y*z + w*x)
    gz = -(1 - 2*(x*x + y*y))
    return np.array([gx, gy, gz], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "v6_robot.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(current_dir, xml_path)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    obs_history_len = config["obs_history_length"]
    obs_single_dim = config["obs_single_frame_dim"]
    control_decimation = config["control_decimation"]
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    clip_actions = config["clip_actions"]
    clip_obs = 100.0
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Force zero command
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    isaac_joint_order = config["isaac_joint_order"]
    num_joints = len(isaac_joint_order)
    mj_joint_names = []
    for i in range(m.njnt):
        if m.jnt_type[i] == 3:  # hinge
            mj_joint_names.append(m.joint(i).name)

    isaac_to_mujoco = np.array([mj_joint_names.index(n) for n in isaac_joint_order])
    default_angles_isaac = default_angles[isaac_to_mujoco]

    actuator_to_joint = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        for j, jname in enumerate(mj_joint_names):
            if m.joint(jname).id == jid:
                actuator_to_joint.append(j)
                break
    actuator_to_joint = np.array(actuator_to_joint)

    policy_path = config["policy_path"]
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    term_dims = [3, 3, 3, num_joints, num_joints, num_actions]
    obs_term_histories = [TermHistory(obs_history_len, td) for td in term_dims]

    obs = np.zeros(num_obs, dtype=np.float32)
    action_raw = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # Init pose — match IsaacLab reset exactly (no warmup)
    d.qpos[2] = config["init_height"]
    d.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    print(f"Init (no warmup): h={d.qpos[2]:.4f}m, ncon={d.ncon}")

    # Fill history with initial obs
    quat = d.qpos[3:7].copy()
    omega_body = d.qvel[3:6].copy()
    qj_mujoco = d.qpos[7:].astype(np.float32)
    dqj_mujoco = d.qvel[6:].astype(np.float32)
    qj_isaac = qj_mujoco[isaac_to_mujoco]
    dqj_isaac = dqj_mujoco[isaac_to_mujoco]

    init_terms = [
        omega_body.astype(np.float32) * ang_vel_scale,
        get_gravity_orientation(quat),
        cmd * cmd_scale,
        (qj_isaac - default_angles_isaac) * dof_pos_scale,
        dqj_isaac * dof_vel_scale,
        np.zeros(num_actions, dtype=np.float32),
    ]
    for hist_buf, term_data in zip(obs_term_histories, init_terms):
        for _ in range(obs_history_len):
            hist_buf.append(term_data)

    # Header
    print(f"\nMuJoCo V6 Humanoid — {args.steps} policy steps, cmd=(0,0,0)")
    print(f"num_obs={num_obs}, obs_history_len={obs_history_len}, num_actions={num_actions}")
    print(f"Isaac joint order: {isaac_joint_order}")
    print(f"\n{'step':>4s}  {'height':>7s}  action_raw (13 joints, Isaac order)")
    print("-" * 120)

    counter = 0
    for policy_step in range(args.steps):
        for _ in range(control_decimation):
            d.ctrl[:] = target_dof_pos[actuator_to_joint]
            mujoco.mj_step(m, d)
            counter += 1

        quat = d.qpos[3:7].copy()
        omega_body = d.qvel[3:6].copy()
        qj_mujoco = d.qpos[7:].astype(np.float32)
        dqj_mujoco = d.qvel[6:].astype(np.float32)
        qj_isaac = qj_mujoco[isaac_to_mujoco]
        dqj_isaac = dqj_mujoco[isaac_to_mujoco]

        omega_obs = omega_body.astype(np.float32) * ang_vel_scale
        gravity_obs = get_gravity_orientation(quat)
        cmd_obs = cmd * cmd_scale
        qj_rel = (qj_isaac - default_angles_isaac) * dof_pos_scale
        dqj_obs = dqj_isaac * dof_vel_scale
        last_act = action_raw.copy()

        current_terms = [omega_obs, gravity_obs, cmd_obs, qj_rel, dqj_obs, last_act]
        for hist_buf, term_data in zip(obs_term_histories, current_terms):
            hist_buf.append(term_data)

        offset = 0
        for hist_buf in obs_term_histories:
            flat = hist_buf.flatten()
            obs[offset:offset + len(flat)] = flat
            offset += len(flat)

        obs = np.clip(obs, -clip_obs, clip_obs)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            action_raw = policy(obs_tensor).detach().numpy().squeeze()
        action_raw = np.clip(action_raw, -clip_actions, clip_actions)

        for i_isaac in range(num_actions):
            mj_idx = isaac_to_mujoco[i_isaac]
            target_dof_pos[mj_idx] = action_raw[i_isaac] * action_scale + default_angles[mj_idx]

        act_str = np.array2string(action_raw, precision=4, separator=', ', max_line_width=200)
        print(f"{policy_step:4d}  {d.qpos[2]:7.4f}  {act_str}")

        if args.verbose and policy_step < 5:
            print(f"       obs: {np.array2string(obs, precision=4, separator=', ', max_line_width=200)}")

if __name__ == "__main__":
    main()
