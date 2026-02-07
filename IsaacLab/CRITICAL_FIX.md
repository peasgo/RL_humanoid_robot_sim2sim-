# MuJoCo部署关键修复

## 问题根源

您的修改导致了3个严重问题：

### 1. 力矩限制过小 (run_robot.py:267)
```python
tau = np.clip(tau, -10, 10)  # ❌ 太小，无法支撑机器人
```
**正确值**: `tau = np.clip(tau, -1000, 1000)`

### 2. 动作限制过小 (run_robot.py:362)
```python
action = np.clip(action, -10.0, 10.0)  # ❌ 太小
```
**正确值**: `action = np.clip(action, -1000.0, 1000.0)`

### 3. 初始高度错误 (run_robot.py:220)
```python
FIXED_BASE_HEIGHT = 0.3  # ❌ 训练时是0.4
```
**正确值**: `FIXED_BASE_HEIGHT = 0.4`

### 4. 速度坐标系错误 (run_robot.py:336-337)
```python
base_lin_vel = d.qvel[0:3]  # ❌ 世界坐标系
omega = d.qvel[3:6]          # ❌ 世界坐标系
```
**需要改为机体坐标系**

## 立即修复

恢复以下参数到原始值：
- 力矩限制: -1000 到 1000
- 动作限制: -1000.0 到 1000.0
- 初始高度: 0.4m
- 速度坐标系: 机体坐标系
