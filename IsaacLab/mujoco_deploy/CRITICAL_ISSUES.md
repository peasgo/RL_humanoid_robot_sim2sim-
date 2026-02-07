# Critical Issues: IsaacLab → Mujoco Deployment

## 🔴 CRITICAL: Joint Order Mismatch

**This is the PRIMARY reason your robot cannot stand.**

### IsaacLab Joint Order (from trajectory recording):
```
['LHipP', 'RHipP', 'LHipY', 'RHipY', 'LHipR', 'RHipR',
 'LKneeP', 'RKneeP', 'LAnkleP', 'RAankleP', 'LAnkleR', 'RAnkleR']
```

### Mujoco Joint Order (from robot.xml):
```
['RHipP', 'RHipY', 'RHipR', 'RKneeP', 'RAankleP', 'RAnkleR',
 'LHipP', 'LHipY', 'LHipR', 'LKneeP', 'LAnkleP', 'LAnkleR']
```

### Impact:
When the policy outputs action[0] for LHipP, Mujoco applies it to RHipP instead.
This causes completely wrong joint movements, making the robot unable to balance.

### Required Mapping:
```
Mujoco[0] RHipP     ← IsaacLab[1]
Mujoco[1] RHipY     ← IsaacLab[3]
Mujoco[2] RHipR     ← IsaacLab[5]
Mujoco[3] RKneeP    ← IsaacLab[7]
Mujoco[4] RAankleP  ← IsaacLab[9]
Mujoco[5] RAnkleR   ← IsaacLab[11]
Mujoco[6] LHipP     ← IsaacLab[0]
Mujoco[7] LHipY     ← IsaacLab[2]
Mujoco[8] LHipR     ← IsaacLab[4]
Mujoco[9] LKneeP    ← IsaacLab[6]
Mujoco[10] LAnkleP  ← IsaacLab[8]
Mujoco[11] LAnkleR  ← IsaacLab[10]
```

## Other Issues

### 1. Control Frequency Mismatch
- **IsaacLab**: dt=0.005s, decimation=4 → control_dt=0.02s
- **Mujoco config**: dt=0.005s, control_decimation=4 → control_dt=0.02s
- **Status**: ✓ Correct in config, but run_robot.py has logic issues

### 2. Initial Height
- **IsaacLab**: 0.35m (from env.yaml)
- **Mujoco**: 0.29m (FIXED_BASE_HEIGHT in run_robot.py)
- **Trajectory**: starts at 0.3475m
- **Status**: ⚠️ Mismatch may cause initial instability

### 3. Base Suspension Logic
- run_robot.py forces base position for first 1000 steps
- This prevents natural dynamics and may cause issues when released

## Fix Priority

1. **IMMEDIATE**: Fix joint order mapping in run_robot.py
2. **HIGH**: Adjust initial height to match IsaacLab (0.35m)
3. **MEDIUM**: Review base suspension logic
4. **LOW**: Verify observation scaling matches exactly
