
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

V4_QUADRUPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rl/RL-human_robot/V4.SLDASM/urdf/V4_new.usd",
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
        pos=(0.0, 0.0, 0.23),
        rot=(0.7071, 0.7071, 0.0, 0.0),
        joint_pos={
            "Waist_2": 3.14159,
            "RSDp": 0.5235987756,
            "RSDy": 0.0,
            "RARMp": -1.5708,
            "RARMAP": 0.7854,
            "LSDp": 0.7854,
            "LSDy": 0.0,
            "LARMp": 1.5708,
            "LARMAp": 0.7854,
            "RHIPp": 0.7854,
            "RHIPy": 0.0,
            "RKNEEP": 1.5708,
            "RANKLEp": -0.7854,
            "LHIPp": -0.7854,
            "LHIPy": 0.0,
            "LKNEEp": -1.5708,
            "LANKLEp": -0.7854,
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
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            effort_limit_sim=120,
            stiffness=120,
            damping=12,
        ),
    },
)
