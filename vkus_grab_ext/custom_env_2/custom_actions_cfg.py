from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ActionsCfg:
    """Action specifications for the MDP.""" # ActionTermCfg
    
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "finger_joint"], scale=0.5, use_default_offset=True)
    

