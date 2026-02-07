# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
V3 Humanoid Robot Configuration
===============================

V3 是一个全身人形机器人，包含：
- 腰部关节 (2 DOF): Waist, Waist_2
- 双臂 (12 DOF): 左右各 6 个关节
- 双腿 (12 DOF): 左右各 6 个关节

总计 26 个自由度

关节命名规则：
- 右腿: RHIPp, RHIPy, RHIPr, RKNEEP, RANKLEp, RANKLEy
- 左腿: LHIPp, LHIPy, LHIPr, LKNEEp, LANKLEp, LANKLEy
- 右臂: RSDp, RSDy, RSDr, RARMp, RARMAP, RARMAy
- 左臂: LSDp, LSDy, LSDr, LARMp, LARMAp, LARMAy
- 腰部: Waist, Waist_2
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


##
# V3 Humanoid Configuration
##

V3_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # USD 文件路径
        usd_path="/home/rl/RL-human_robot/V3/urdf/V3_v3.usd",
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
        pos=(0.0, 0.0, 0.30),  # 初始高度
        rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z (无旋转)
        joint_pos={
            # === 腰部关节 ===
            "Waist": 0.0,
            "Waist_2": 0.0,
            
            # === 右腿关节 ===
            "RHIPp": 0.0,      
            "RHIPy": 0.0,      
            "RHIPr": 0.0,      
            "RKNEEP": 0.0,     
            "RANKLEp": 0.0,    
            "RANKLEy": 0.0,    
            
            # === 左腿关节 ===
            "LHIPp": 0.0,      
            "LHIPy": 0.0,      
            "LHIPr": 0.0,      
            "LKNEEp": 0.0,     
            "LANKLEp": 0.0,    
            "LANKLEy": 0.0,    
            
            # === 右臂关节 ===
            "RSDp": 0.0,       
            "RSDy": 0.0,       
            "RSDr": 0.0,       
            "RARMp": 0.0,      
            "RARMAP": 0.0,     
            "RARMAy": 0.0,     
            
            # === 左臂关节 ===
            "LSDp": 0.0,       
            "LSDy": 0.0,       
            "LSDr": 0.0,       
            "LARMp": 0.0,      
            "LARMAp": 0.0,     
            "LARMAy": 0.0,     
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # === 腿部执行器（主要用于行走） ===
        "legs_hip": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP.*"],  # 匹配所有髋部关节
            effort_limit_sim=150.0,
            stiffness={
                ".*HIPp": 100.0,   # 髋部俯仰需要较大刚度
                ".*HIPy": 80.0,    # 髋部偏航
                ".*HIPr": 80.0,    # 髋部横滚
            },
            damping={
                ".*HIPp": 10.0,
                ".*HIPy": 8.0,
                ".*HIPr": 8.0,
            },
        ),
        "legs_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*KNEE.*"],  # 匹配所有膝盖关节
            effort_limit_sim=150.0,
            stiffness=120.0,
            damping=10.0,
        ),
        "legs_ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],  # 匹配所有踝关节
            effort_limit_sim=50.0,
            stiffness=25.0,
            damping=5.0,
        ),
        
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist.*"],
            effort_limit_sim=100.0,
            stiffness=60.0,
            damping=8.0,
        ),
        
        "arms_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[".*SD.*"],  # 肩部关节
            effort_limit_sim=50.0,
            stiffness=40.0,
            damping=5.0,
        ),
        "arms_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*ARM.*"],  # 手臂关节
            effort_limit_sim=30.0,
            stiffness=30.0,
            damping=4.0,
        ),
    },
)


