"""
Compare Isaac Lab and MuJoCo obs dumps side by side.

Usage:
  python compare_obs.py <isaac_dump.npz> <mujoco_dump.npz>
"""
import numpy as np
import sys


def load_dump(path):
    data = np.load(path, allow_pickle=True)
    return list(data['data'])


def compare(isaac_path, mujoco_path, max_steps=20):
    isaac_data = load_dump(isaac_path)
    mujoco_data = load_dump(mujoco_path)

    n = min(len(isaac_data), len(mujoco_data), max_steps)
    print(f"Comparing {n} steps (Isaac: {len(isaac_data)}, MuJoCo: {len(mujoco_data)})")

    obs_labels = [
        ("ang_vel",   0,  3),
        ("gravity",   3,  6),
        ("cmd",       6,  9),
        ("jpos_rel",  9, 22),
        ("jvel_rel", 22, 35),
        ("last_act", 35, 47),
    ]

    for step in range(n):
        i = isaac_data[step]
        m = mujoco_data[step]

        i_obs = i['obs']
        m_obs = m['obs']

        diff = np.abs(i_obs - m_obs)
        max_diff = np.max(diff)

        if step < 5 or step % 10 == 0 or max_diff > 0.1:
            print(f"\n{'='*70}")
            print(f"Step {step}  (max obs diff: {max_diff:.6f})")
            print(f"{'='*70}")

            for label, s, e in obs_labels:
                i_slice = i_obs[s:e]
                m_slice = m_obs[s:e]
                d_slice = diff[s:e]
                max_d = np.max(d_slice)
                flag = " *** MISMATCH ***" if max_d > 0.05 else ""
                print(f"\n  {label} [{s}:{e}] (max_diff={max_d:.6f}){flag}")
                print(f"    Isaac:  {np.array2string(i_slice, precision=5, suppress_small=True)}")
                print(f"    MuJoCo: {np.array2string(m_slice, precision=5, suppress_small=True)}")
                if max_d > 0.01:
                    print(f"    Diff:   {np.array2string(d_slice, precision=5, suppress_small=True)}")

            # Compare raw states if available
            if 'root_quat_w' in i and 'root_quat' in m:
                print(f"\n  root_quat:")
                print(f"    Isaac:  {i['root_quat_w']}")
                print(f"    MuJoCo: {m['root_quat']}")

            if 'joint_pos' in i and 'qj_mujoco' in m:
                jdiff = np.abs(i['joint_pos'] - m['qj_mujoco'])
                print(f"\n  joint_pos (raw, max_diff={np.max(jdiff):.6f}):")
                print(f"    Isaac:  {np.array2string(i['joint_pos'], precision=5)}")
                print(f"    MuJoCo: {np.array2string(m['qj_mujoco'], precision=5)}")

            if 'actions' in i and 'actions' in m:
                adiff = np.abs(i['actions'] - m['actions'])
                print(f"\n  actions (max_diff={np.max(adiff):.6f}):")
                print(f"    Isaac:  {np.array2string(i['actions'], precision=5)}")
                print(f"    MuJoCo: {np.array2string(m['actions'], precision=5)}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Per-component max absolute difference over all steps")
    print(f"{'='*70}")
    all_diffs = []
    for step in range(n):
        d = np.abs(isaac_data[step]['obs'] - mujoco_data[step]['obs'])
        all_diffs.append(d)
    all_diffs = np.array(all_diffs)
    max_per_dim = np.max(all_diffs, axis=0)

    for label, s, e in obs_labels:
        print(f"  {label:12s} [{s:2d}:{e:2d}]: max={np.max(max_per_dim[s:e]):.6f}  mean={np.mean(max_per_dim[s:e]):.6f}")
    print(f"  {'TOTAL':12s}        : max={np.max(max_per_dim):.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_obs.py <isaac_dump.npz> <mujoco_dump.npz>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
