import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math

from .rewards import velocity_profile_reward

@configclass
class Rewards(RewardsCfg):
    """Reward terms for the MDP."""
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    
    feet_air_time = None
    undesired_contacts = None
    action_rate_l2 = None
    ang_vel_xy_l2 = None
    dof_acc_l2 = None
    dof_pos_limits = None
    dof_torques_l2 = None
    flat_orientation_l2 = None 
    lin_vel_z_l2 = None
    
    
    
    
    velocity_profile_reward = RewTerm(
        func=velocity_profile_reward,
        weight=1.0,
#        params={
#            "alpha": 0.5,
#            "use_xy": True,
#            "max_dist": 10.0,
#        },
    )  
