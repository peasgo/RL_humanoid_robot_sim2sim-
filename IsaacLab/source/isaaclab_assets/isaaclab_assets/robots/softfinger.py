# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# SoftFinger articulation configuration (loads your USD for training)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ===(1) 改成你的USD路径：=====================================================
# 备注：如果你后续把 usd 放到工程内，请把以下路径改为相对路径或环境变量。


SOFTFINGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/rl/桌面/softfinger/FigureAssmURDFR/urdf/11_1_usdfolder/softfinger_convert_urdf.usd",
        usd_path="/home/rl/桌面/softfinger/FigureAssmURDFR/urdf/FigureAssmURDFR.usd",
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
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1,
        ),
    ),
    # ===(2) 初始位姿/关节：如果你的USD里关节名不同，不用逐一列出，直接用正则".*"清零即可
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),        # 把软指放高一点避免初始穿模
        rot=(1.0, 0.0, 1.57, 0.0),    # w,x,y,z
        # joint_pos={".*": 0.0},

        joint_pos = {
            "F1B_F1P": -1.10,
            "F1M_F1D": -0.59,
            "F1P_F1M": -0.24,
            "F2B_F1B": -1.40,
            "F1B_FT2T": -1.80,
            "F2B_F2P": -0.82,
            "F2M_F2D":  -0.65,
            "F2P_F2M": -0.57,
            "FT2T_FTB": 1.40,
            "F2B_F3B": -0.97,
            "F3B_F3P":  0.80,
            "F3M_F3D":  0.69,
            "F3P_F3M":  0.79,
            "F3B_F4B":  1.00,
            "F4B_F4P": -0.83,
            "F4M_F4D": -0.36,
            "F4P_F4M": -0.67,
            "FTB_FTM": -0.62,
            "FTM_FTD": -0.93,
        },

        joint_vel={".*": 0.0},
    ),
    # ===(3) 关节限位缓冲
    soft_joint_pos_limit_factor=0.98,
    # ===(4) 执行器：通配所有关节，用隐式PD；后续可按关节分别细化
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # joint_names_expr=[
            #     "F2B_F3B",
            #     "F3B_F4B",
            #     "F2B_F1B",
            #     "F1B_F1P",
            #     "FTB_FTM",
            #     "FT2T_FTB",
            #     "F1B_FT2T",
            #     "F2B_F2P",
            #     "F3B_F3P",
            #     "F4B_F4P",

            # ],
            stiffness=40.0,     # 先用温和刚度，利于稳定
            damping=2.0,
            effort_limit=2000.0, # 足够宽松，避免早期饱和
            velocity_limit=20.0,
            friction=0.0,
        ),
    },
)
