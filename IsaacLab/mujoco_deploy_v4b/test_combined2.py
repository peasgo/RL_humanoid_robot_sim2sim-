#!/usr/bin/env python3
"""简化组合测试：只测试3个最关键的配置"""
import numpy as np
import mujoco
import torch
import yaml
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "v4_scene.xml")
with open(os.path.join(SCRIPT_DIR, "v4_robot.yaml")) as f:
    cfg = yaml.safe_load(f)
policy = torch.jit.load(cfg["policy_path"], map_location="cpu")
policy.eval()

def quat_to_rotmat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def w2b(v,q): return quat_to_rotmat(q).T @ v
def get_grav(q):
    w,x,y,z=q; return np.array([-2*(x*z-w*y),-2*(y*z+w*x),-(1-2*(x*x+y*y))])
def remap_lin(v): return np.array([v[2],v[0],v[1]])
def remap_ang(v): return np.array([v[0],v[2],v[1]])
def remap_grav(v): return np.array([v[2],v[0],v[1]])
def quat_to_yaw(q):
    w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

def run(cmd_vec, ang_mode="double_rot", of=0.0, ramp=0, h=0.26, af=0.3, N=500):
    m=mujoco.MjModel.from_xml_path(SCENE_XML); d=mujoco.MjData(m)
    m.opt.timestep=cfg["simulation_dt"]
    da=np.array(cfg["default_angles"],dtype=np.float64)
    asc=cfg["action_scale"]; dec=cfg["control_decimation"]
    mj_n=[]
    for j in range(m.njnt):
        jn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
        if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE: mj_n.append(jn)
    i17=['LHIPp','RHIPp','LHIPy','RHIPy','Waist_2','LSDp','RSDp','LKNEEp','RKNEEP','LSDy','RSDy','LANKLEp','RANKLEp','LARMp','RARMp','LARMAp','RARMAP']
    i16=[j for j in i17 if j!='Waist_2']
    i17m=np.array([mj_n.index(j) for j in i17])
    i16m=np.array([mj_n.index(j) for j in i16])
    wm=mj_n.index('Waist_2')
    a2j=np.array([mj_n.index(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,m.actuator_trnid[i,0])) for i in range(m.nu)])
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"base_link")
    mujoco.mj_resetData(m,d)
    d.qpos[2]=h; d.qpos[3:7]=[.7071068,.7071068,0,0]; d.qpos[7:]=da; d.qvel[:]=0
    tgt=da.copy(); a16r=np.zeros(16,np.float32); a16f=np.zeros(16,np.float32)
    obs=np.zeros(62,np.float32); pobs=np.zeros(62,np.float32)
    cmd=np.array(cmd_vec,np.float32)
    ip=d.qpos[0:3].copy(); iy=quat_to_yaw(d.qpos[3:7]); cnt=0; ps=0
    for s in range(N*dec):
        d.ctrl[:]=tgt[a2j]; mujoco.mj_step(m,d); cnt+=1
        if cnt%dec==0:
            q=d.qpos[3:7]
            lv=w2b(d.qvel[0:3].copy(),q)
            if ang_mode=="double_rot": om=w2b(d.qvel[3:6].copy(),q)
            elif ang_mode=="direct": om=d.qvel[3:6].copy()
            else:
                v6=np.zeros(6); mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,v6,1); om=v6[0:3]
            qj=d.qpos[7:].copy()[i17m]; dqj=d.qvel[6:].copy()[i17m]; di=da[i17m]; g=get_grav(q)
            obs[0:3]=remap_lin(lv); obs[3:6]=remap_ang(om); obs[6:9]=remap_grav(g)
            obs[9:12]=cmd; obs[12:29]=(qj-di).astype(np.float32); obs[29:46]=dqj.astype(np.float32)
            obs[46:62]=a16r.astype(np.float32)
            if of>0 and ps>0:
                obs[0:6]=of*pobs[0:6]+(1-of)*obs[0:6]; obs[29:46]=of*pobs[29:46]+(1-of)*obs[29:46]
            pobs[:]=obs
            with torch.no_grad(): out=policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            a16r[:]=np.clip(out,-5,5)
            eff=a16r.copy()
            if ramp>0 and ps<ramp: eff*=float(ps)/float(ramp)
            if af>0: a16f[:]=af*a16f+(1-af)*eff
            else: a16f[:]=eff
            ps+=1
            tgt[wm]=da[wm]
            for i in range(16): tgt[i16m[i]]=a16f[i]*asc+da[i16m[i]]
    fp=d.qpos[0:3]
    return -(fp[1]-ip[1]), fp[0]-ip[0], np.degrees(quat_to_yaw(d.qpos[3:7])-iy), fp[2]

cmds=[("FWD 0.5",[.5,0,0]),("FWD 0.3",[.3,0,0]),("STAND",[0,0,0]),
      ("BWD 0.3",[-.3,0,0]),("BWD 0.5",[-.5,0,0]),("LEFT 0.3",[0,.3,0]),
      ("TURN_L",[0,0,.5]),("TURN_R",[0,0,-.5])]

configs=[
    ("CLEAN+AF double_rot", "double_rot", 0.0, 0, 0.26, 0.3),
    ("CLEAN+AF objvel",     "objvel",     0.0, 0, 0.26, 0.3),
    ("CLEAN+AF direct",     "direct",     0.0, 0, 0.26, 0.3),
    ("BASELINE double_rot", "double_rot", 0.3, 50, 0.22, 0.0),
]

for cn,am,of,rp,h,af in configs:
    print(f"\n{'='*80}\n  {cn}\n{'='*80}")
    print(f"  {'cmd':12s} | {'fwd':>7s} | {'lat':>7s} | {'yaw':>7s} | {'h':>5s}")
    print(f"  {'-'*50}")
    for name,cv in cmds:
        f,l,y,ht=run(cv,ang_mode=am,of=of,ramp=rp,h=h,af=af)
        print(f"  {name:12s} | {f:+7.3f} | {l:+7.3f} | {y:+7.1f} | {ht:.3f}")
