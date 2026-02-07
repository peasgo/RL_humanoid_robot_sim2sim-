# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# ----------------------------------------------------------------------------------------
# Per‑joint equal scaling map (MCP/PIP/DIP equal; thumb CMC/MP/IP equal)
# ----------------------------------------------------------------------------------------

ACTION_SCALE_POS = {
    # Index (F1*)
    "F1B_F1P": 0.7, "F1P_F1M": 0.7, "F1M_F1D": 0.7,
    # Middle (F2*)
    "F2B_F2P": 0.7, "F2P_F2M": 0.7, "F2M_F2D": 0.7,
    # Ring (F3*)
    "F3B_F3P": 0.7, "F3P_F3M": 0.7, "F3M_F3D": 0.7,
    # Little (F4*)
    "F4B_F4P": 0.7, "F4P_F4M": 0.7, "F4M_F4D": 0.7,
    # Thumb (FT*) — carpometacarpal/MP/IP equal as well
    "FT2T_FTB": 0.7, "FTB_FTM": 0.7, "FTM_FTD": 0.7,
    # Palm spreads (ab/adduction) — smaller amplitude
    "F2B_F3B": 0.3, "F3B_F4B": 0.3, "F2B_F1B": 0.3,
}

# Deterministic joint order for action post‑processing (matches the dict above)
ACTION_JOINT_ORDER = [
    "F1B_F1P", "F1P_F1M", "F1M_F1D",
    "F2B_F2P", "F2P_F2M", "F2M_F2D",
    "F3B_F3P", "F3P_F3M", "F3M_F3D",
    "F4B_F4P", "F4P_F4M", "F4M_F4D",
    "FT2T_FTB", "FTB_FTM", "FTM_FTD",
    "F2B_F3B", "F3B_F4B", "F2B_F1B",
]


# ----------------------------------------------------------------------------------------
# Scene definition
# ----------------------------------------------------------------------------------------


@configclass
class HandSceneCfg(InteractiveSceneCfg):
    """Scene for fingertip locomotion using a 5-finger hand as a quadruped (+thumb)."""

    # terrain: mild roughness for contact-rich learning
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot articulation (URDF converted to USD recommended)
    robot: ArticulationCfg = MISSING

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Contact forces — track fingertips as "feet"; include thumb tip as fifth contact
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=True,
        track_bodies=["F1T", "F2T", "F3T", "F4T", "FTT"],
    )

    # lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ----------------------------------------------------------------------------------------
# MDP settings
# ----------------------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Velocity tracking commands for fingertip locomotion."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.6,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action space for the hand — position targets on joints."""

    # Keep a neutral global scale=1.0. Per‑joint scales are applied by a small
    # post‑processing helper using ACTION_SCALE_POS.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups for policy and (optional) critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization and resets."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    add_root_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="F2B"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Reward shaping for fingertip gait."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.3,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.12,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F1T|F2T|F3T|F4T"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F1B|F2B|F3B|F4B|FT2T|FTB"),
            "threshold": 1.0,
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="F1B|F2B|F3B|F4B"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum schedule."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ----------------------------------------------------------------------------------------
# Environment configuration
# ----------------------------------------------------------------------------------------


@configclass
class HandVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-tracking locomotion for a 5-finger hand (quadruped-style)."""

    # Scene settings
    scene: HandSceneCfg = HandSceneCfg(num_envs=4096, env_spacing=2.0)

    # Robot asset (fill in your USD/URDF path)
    scene_robot_asset_path = f"{ISAAC_NUCLEUS_DIR}/Projects/hand_walk/FigureAssmURDFR.usd"  # edit me
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file_path=scene_robot_asset_path,
        fix_base=False,
        disable_gravity=False,
        linear_damping=0.01,
        angular_damping=0.01,
        articulation_props={
            "solver_position_iteration_count": 12,
            "solver_velocity_iteration_count": 1,
            "enable_self_collisions": True,
        },
    )

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # enable curriculum for terrain generator if configured
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # --------------------------------------------------------------------------------
        # Hook: expose ACTION_SCALE_POS to downstream env logic.
        # If your runtime converts normalized actions to PD targets, multiply the
        # per-joint amplitude by ACTION_SCALE_POS[name]. For example, in your
        # step() method you might do:
        #   for i, name in enumerate(ACTION_JOINT_ORDER):
        #       amp = 0.5 * (hi - lo) * ACTION_SCALE_POS[name]
        #       target[i] = mid + amp * actions[i]
        # This keeps MCP/PIP/DIP equal per finger and thumb joints equal.
        # --------------------------------------------------------------------------------
        self.action_scale_pos = ACTION_SCALE_POS
        self.action_joint_order = ACTION_JOINT_ORDER
