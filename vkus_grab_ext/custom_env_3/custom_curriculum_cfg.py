from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .param_scheduler import lerp_scalar, lerp_tuple


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # CONDITION BASED CURRICULUM CHANGER
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    '''

    # WEIGHT SCALAR SMOOTH CHANGERS    
    overshoot_penalty_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.velocity_profile_reward.params.k_moving_away", 
            "modify_fn": lerp_scalar,          
            "modify_params": {
                            "start": 0.1, 
                            "end": 2.0, 
                            "num_steps": 200_000, 
                            "start_after": 2_000,
                            "log_name": "overshoot_penalty_curriculum"
                            },
        },
    )
    

    
    # COMMAND RANGE SMOOTH CHANGERS 
    target_joint_pose_resample_time_range_curricilum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.target_joint_pose.resampling_time_range",
            "modify_fn": lerp_tuple,
            "modify_params": {
                            "start": (10.0, 10.0), 
                            "end": (1.0, 1.0), 
                            "num_steps": 200_000, 
                            "start_after": 2_000,
                            "log_name": "target_joint_pose_resample_time_range"
                            },
            
        },
    )
    
    override_velocity_resample_time_range_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.override_velocity.resampling_time_range",
            "modify_fn": lerp_tuple,
            "modify_params": {
                            "start": (10.0, 10.0),
                            "end":   (1.0, 1.0),
                            "num_steps": 200_000,
                            "start_after": 2_000,
                            "log_name": "override_velocity_resample_time_range"
                            },
        },
    ) 
    '''
