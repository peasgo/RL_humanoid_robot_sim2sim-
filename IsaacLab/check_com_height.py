#!/usr/bin/env python3
import torch
from isaaclab_assets.robots.parallelhuman import PARALLELHUMAN_CFG
from isaaclab_assets.robots.softfinger import SOFTFINGER_CFG

# ParallelHuman
ph_init_pos = PARALLELHUMAN_CFG.init_state.pos
print("=" * 60)
print("ParallelHuman:")
print(f"  Root link初始位置: {ph_init_pos}")
print(f"  Root link初始高度(Z): {ph_init_pos[2]} m")

# SoftFinger
sf_init_pos = SOFTFINGER_CFG.init_state.pos
print("\n" + "=" * 60)
print("SoftFinger:")
print(f"  Root link初始位置: {sf_init_pos}")
print(f"  Root link初始高度(Z): {sf_init_pos[2]} m")

print("\n" + "=" * 60)
print("注意:")
print("  - 以上是root link（基座链接）的高度")
print("  - 质心高度 = root link高度 + body_com_pos_b（质心偏移）")
print("  - body_com_pos_b由USD模型的质量分布决定")
print("  - 需要运行仿真才能获取实际质心高度")
