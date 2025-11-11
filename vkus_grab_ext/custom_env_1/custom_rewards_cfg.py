import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math

#from .rewards import feet_impact_vel, pelvis_height_target_reward, no_command_motion_penalty, lateral_slip_penalty, heading_alignment_reward, leg_pelvis_torso_coalignment_reward, idle_penalty, angvel_flat_l2_product, alternating_airtime_reward, step_phase_reward, com_projection_reward, step_width_penalty, track_lin_vel_xy_exp_custom, track_ang_vel_z_exp_custom, foot_symmetry_step_reward_cmddir, target_distance_exp_reward

@configclass
class Rewards(RewardsCfg):
    """Reward terms for the MDP."""
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    
    feet_air_time = None
    undesired_contacts = None
    

