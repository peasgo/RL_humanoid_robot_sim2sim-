from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, RewardsCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.softfinger import SOFTFINGER_CFG

@configclass
class SoftfingerActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "F1B_F1P",
            "FTB_FTM",
            "F2B_F2P",
            "F3B_F3P",
            "F4B_F4P",
        ],
        scale=0.20,
        use_default_offset=True,
    )

@configclass
class SoftfingerRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # … 你原来的奖励保持 …

@configclass
class SoftfingerRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: SoftfingerRewards = SoftfingerRewards()
    actions: SoftfingerActionsCfg = SoftfingerActionsCfg()  # 👈 绑定新动作表

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOFTFINGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 其它配置保持你原来的（高度扫描/事件/奖励权重/terminations 等）
