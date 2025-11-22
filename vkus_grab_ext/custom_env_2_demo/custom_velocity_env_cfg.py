from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .custom_observations_cfg import ObservationsCfg
from .custom_rewards_cfg import Rewards
from .custom_commands_cfg import CommandsCfg
from .custom_event_cfg import EventCfg
from .custom_scene_cfg import SceneCfg
from .custom_terminations_cfg import TerminationsCfg
from .custom_curriculum_cfg import CurriculumCfg
from .custom_actions_cfg import ActionsCfg
import isaaclab.sim as sim_utils

class CustomLocomotionVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene : SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: Rewards = Rewards()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    terminations : TerminationsCfg  = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    actions: ActionsCfg = ActionsCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
                                                                    friction_combine_mode="multiply",
                                                                    restitution_combine_mode="multiply",
                                                                    static_friction=1.0,
                                                                    dynamic_friction=1.0,
                                                                )        
        
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15



