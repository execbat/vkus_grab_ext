import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import math
from .commands import UniformVectorCommandCfg, BernoulliMaskCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.markers.config import (
    VisualizationMarkersCfg,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)

from .commands_udp import UdpTargetJointPoseCommandCfg, UdpOverrideVelocityCommandCfg, UdpTargetJointPoseCommand, UdpOverrideVelocityCommand

 
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    '''
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    ) 
    '''
    
    target_joint_pose = UniformVectorCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 9,
        ranges=((-1.0, 1.0),) * 9,        
    )
    
    override_velocity = UniformVectorCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 1,
        ranges=((0.0, 1.0),) * 1,       
    )
    '''
    dof_mask = BernoulliMaskCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 9,
        p_one = 0.5,
        ranges=((-1.0, 1.0),) * 9,
        
    )
    '''

@configclass
class TeleopCommandsCfg:
    """Command specifications for the MDP."""
   
    target_joint_pose = UdpTargetJointPoseCommandCfg(
        class_type=UdpTargetJointPoseCommand,
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        dim=9,
        ranges=((-1.0, 1.0),) * 9,
        default=0.0,
        ip="0.0.0.0",
        port=55001,            
        packet_format="json",
        struct_fmt="<10f",    
        debug_vis=True,
    )

    override_velocity = UdpOverrideVelocityCommandCfg(
        class_type=UdpOverrideVelocityCommand,
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        dim=1,
        ranges=((0.0, 1.0),),
        default=0.0,
        ip="0.0.0.0",
        port=55001,           
        packet_format="json",
        struct_fmt="<10f",
        debug_vis=True,
    )
 

