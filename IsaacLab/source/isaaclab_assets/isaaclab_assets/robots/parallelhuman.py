import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# ParallelHuman (Bipedal Humanoid) Configuration
##

PARALLELHUMAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 路径请根据你的 Linux 实际存放位置修改
        #usd_path="/home/rl/桌面/WholeAssembleV2/urdf/WholeAssembleV2.usd",
        usd_path="/home/rl/桌面/WholeAssembleV2_2/urdf/WholeAssembleV2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),  # 初始高度下调至 0.28m，避免悬空
        rot=(1.0, 0.0, 0.0, 0.0), # w, x, y, z
        joint_pos={
            # --- 右腿 (根据图片关节名) ---
            "RHipP": 0.0,    # 髋部微屈
            "RHipY": 0.0,
            "RHipR": 0.0,
            "RKneeP": 0.0,    # 膝盖微屈 (-0.3 rad 约 17度)
            "RAankleP": -0, # 注意图片中是两个 'a': RAankleP
            "RAnkleR": 0.0,
            
            # --- 左腿 ---
            "LHipP": 0.0,    # 髋部微屈
            "LHipY": 0.0,
            "LHipR": 0.0,
            "LKneeP": 0.0,    # 膝盖微屈
            "LAnkleP": -0,
            "LAnkleR": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["RHipP", "RHipY", "RHipR", "RKneeP", "LHipP", "LHipY", "LHipR", "LKneeP"],
            # 1. 降低力矩上限，防止瞬时爆发过力
            effort_limit_sim=150.0, 
            # 2. 降低刚度（原先 150-200 可能太硬了）
            stiffness={
                ".*Hip.*": 80.0,   # 髋部通常需要一定支撑力，但 80 足够了
                ".*Knee.*": 120.0, # 膝盖可以稍高，维持站立高度
            },
            # 3. 显著加大阻尼，抑制速度
            damping={
                ".*Hip.*": 10.0,   # 增加到 10 以上
                ".*Knee.*": 10.0,
            },
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["RAankleP", "RAnkleR", "LAnkleP", "LAnkleR"],
            effort_limit_sim=50.0,
            # 踝关节通常需要更软，以便脚掌贴地
            stiffness=20.0,
            damping=4.0, # 阻尼通常设为刚度的 1/5 到 1/10
        ),
    },
)