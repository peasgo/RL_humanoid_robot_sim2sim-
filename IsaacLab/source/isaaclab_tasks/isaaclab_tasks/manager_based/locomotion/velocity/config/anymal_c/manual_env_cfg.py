# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import math
import copy
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg, TerminationsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class ManualCommandsCfg(CommandsCfg):
    """Manual command configuration - no random resampling."""
    
    def __post_init__(self):
        super().__post_init__()
        # Disable command resampling - commands will be set manually
        # Use a very large resampling time instead of math.inf
        self.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
        # Set initial command to zero
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), 
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(0.0, 0.0), 
            heading=(0.0, 0.0)
        )


@configclass
class ManualTerminationsCfg(TerminationsCfg):
    """Manual termination configuration - disable timeout."""
    
    def __post_init__(self):
        super().__post_init__()
        # Disable timeout termination - episodes will run indefinitely
        self.time_out = None


@configclass
class AnymalCRoughManualEnvCfg(AnymalCRoughEnvCfg):
    """Anymal-C rough environment configuration for manual control."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        @configclass
        class ManualSceneCfg(InteractiveSceneCfg):
            # terrain same as rough_env_cfg: global ground generator at /World/ground
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=ROUGH_TERRAINS_CFG,
                max_init_terrain_level=5,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                    project_uvw=True,
                    texture_scale=(0.25, 0.25),
                ),
                debug_vis=False,
            )

            # light
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)),
            )

            ################################################################
            ######################## STUDENT TODO BEGIN ####################
            ################################################################
            # 在此区域添加或修改更多的场景资产（例如更多立方体台阶、斜坡等，或者从USD导入，取决于你。
            # 建议使用与上方相同的 AssetBaseCfg + sim_utils.CuboidCfg 写法，
            # 修改 size / init_state.pos 即可改变尺寸与位置。













            ################################################################
            ######################## STUDENT TODO END ######################
            ################################################################

            # robot & contact sensor
            robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

            # height scanner (required by default ObservationsCfg)
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )

        # func tweaks
        self.scene = ManualSceneCfg(num_envs=1, env_spacing=3.0)
        self.commands = ManualCommandsCfg()
        self.terminations = ManualTerminationsCfg()
        self.curriculum = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
