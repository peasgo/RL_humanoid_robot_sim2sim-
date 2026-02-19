import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

V6_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rl/RL-human_robot/URDF_Humanoid_legs_V6/urdf/URDF_Humanoid_legs_V6.usd",
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
        pos=(0.0, 0.0, 0.55),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "pelvis_link": 0.0,
            "RHIPp": -0.2,
            "RHIPy": 0.0,
            "RHIPr": 0.0,
            "RKNEEp": 0.4,
            "RANKLEp": -0.2,
            "RANKLEy": 0.0,
            "LHIPp": -0.2,
            "LHIPy": 0.0,
            "LHIPr": 0.0,
            "LKNEEp": -0.4,
            "LANKLEp": 0.2,
            "LANKLEy": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "pelvis": ImplicitActuatorCfg(
            joint_names_expr=["pelvis_link"],
            effort_limit_sim=200.0,
            stiffness=150.0,
            damping=15.0,
        ),
        "hip": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP.*"],
            effort_limit_sim=150.0,
            stiffness=120.0,
            damping=12.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*KNEE.*"],
            effort_limit_sim=150.0,
            stiffness=120.0,
            damping=12.0,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            effort_limit_sim=80.0,
            stiffness=80.0,
            damping=8.0,
        ),
    },
)
