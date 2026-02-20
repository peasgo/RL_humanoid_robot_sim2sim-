"""诊断脚底碰撞 v2：验证新增 box geom 位置 + contact 稳定性"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("v6_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("=" * 60)
print("1. 有碰撞属性的 geom（contype>0 或 conaffinity>0）")
print("=" * 60)
type_names = {0:"plane",1:"hfield",2:"sphere",3:"capsule",4:"ellipsoid",5:"cylinder",6:"box",7:"mesh"}
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
    ct = model.geom_contype[i]
    ca = model.geom_conaffinity[i]
    if ct > 0 or ca > 0:
        gtype = type_names.get(model.geom_type[i], "?")
        pos = data.geom_xpos[i]
        fric = model.geom_friction[i]
        print(f"  {name:30s} type={gtype:6s} ct={ct} ca={ca} pos=[{pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}] fric={fric}")

print()
print("=" * 60)
print("2. 脚底 box geom 世界坐标（初始姿态）")
print("=" * 60)
for name in ["R_foot_sole", "L_foot_sole"]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid >= 0:
        pos = data.geom_xpos[gid]
        size = model.geom_size[gid]
        lowest_z = pos[2] - size[2]
        print(f"  {name}: center={pos}, size={size}, lowest_z={lowest_z:.4f}")
    else:
        print(f"  {name}: NOT FOUND!")

print()
print("=" * 60)
print("3. 模拟 500 步（2.5s），每 100 步打印 ncon 和 contact")
print("=" * 60)
for step in range(500):
    mujoco.mj_step(model, data)
    if (step + 1) % 100 == 0:
        t = (step + 1) * model.opt.timestep
        print(f"\n  --- t={t:.2f}s (step {step+1}), ncon={data.ncon} ---")
        foot_contacts = 0
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
            if "foot_sole" in g1 or "foot_sole" in g2 or "floor" in g1 or "floor" in g2:
                foot_contacts += 1
                print(f"    {g1} <-> {g2}  pos=[{c.pos[0]:+.4f},{c.pos[1]:+.4f},{c.pos[2]:+.4f}] dist={c.dist:.6f}")
        if foot_contacts == 0:
            print(f"    ⚠ 没有脚-地面 contact!")
        
        # 打印 base_link 高度
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        print(f"    base_link z = {data.xpos[base_id][2]:.4f}")
