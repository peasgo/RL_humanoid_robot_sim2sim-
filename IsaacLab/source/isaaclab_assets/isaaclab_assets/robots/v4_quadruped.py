
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

V4_QUADRUPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rl/RL-human_robot/DOG_V5/urdf/DOG_V5.usd",
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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        rot=(0.7071, 0.7071, 0.0, 0.0),
        joint_pos={
            "Waist_2": 3.14159,  # 上半身完全前倾 ≈π
            # 前腿（手臂）
            "RSDp": 0.6,
            "RSDy": 0.0,
            "RARMp": -1.4,
            "LSDp": 0.6,
            "LSDy": 0.0,
            "LARMp": 1.4,
            # 后腿
            "RHIPp": 0.78,
            "RHIPy": 0.0,
            "RKNEEP": 1.0,
            "LHIPp": -0.78,
            "LHIPy": 0.0,
            "LKNEEp": -1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist_2"],
            effort_limit_sim=1000.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*SD.*", ".*ARM.*", ".*HIP.*", ".*KNEE.*"],
            effort_limit_sim=150,
            stiffness=150,
            damping=15,
        ),
    },
)
