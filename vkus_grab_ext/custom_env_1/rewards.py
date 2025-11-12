from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
    
import math
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


# Find ALL articulation data at isaaclab.assets.articulation.articulation_data.ArticulationData class
def velocity_profile_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ctrl_vel_command_name: str = "override_velocity",
    target_command_name: str = "target_joint_pose",
    kv: float = 1.0,
    kp: float = 1.0,
    
) -> torch.Tensor:
    """
    Ревард: насколько модуль скорости по осям соответствует эталонному профилю v_des(dist).
    - dist считается в нормированных координатах [-1, 1]
    - beta зависит от команды velocity в [0, 1]
    - ревард для каждого env = exp(-MSE / std_vel^2), где MSE по всем осям.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # WORKING WITH AXIS POSITIONS
    # normalization of joint position [-1,1]
    q_act = asset.data.joint_pos                                       # [N.J]
    device, dtype = q_act.device, q_act.dtype
    qmin   = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax   = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    offset = 0.5 * (qmin + qmax)
    joint_act_norm   = 2.0 * (q_act - offset) / (qmax - qmin + 1e-6)                      # (N,J)
    joint_act_norm = torch.clamp(joint_act_norm, -1.0, 1.0)
        
    # target joint pos by cmd
    joint_target_norm = env.command_manager.get_command(target_command_name)  # [N]    
    
    
    # position difference
    pos_diff_norm = joint_target_norm - joint_act_norm
    
    # WORKING WITH AXIS VELOCITIES
    joint_vel_act = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    joint_vel_act_norm = joint_vel_act / joint_vel_limits              # [-1; 1]
    joint_vel_act_norm = torch.clamp(joint_vel_act_norm, -1.0, 1.0)
    
    vel_regulation = env.command_manager.get_command(ctrl_vel_command_name)     # [N]
    
    
    # Getting desired axis velocities based on pos_diff
    vel_etalon_norm = torch.tanh(pos_diff_norm) * vel_regulation
    
    # CALC VEL DIFF REWARD
    joint_vel_diff_norm = vel_etalon_norm - joint_vel_act_norm
    vel_reward = torch.exp(- joint_vel_diff_norm**2 / kv**2).mean(dim=-1)
    
    # CALC POS DIFF REWARD
    pos_reward = torch.exp(- pos_diff_norm**2 / kp**2).mean(dim=-1)
    
    # OVERALL REWARD
    overall_reward = vel_reward * pos_reward																									
    
    
    
    
    '''
    print('#####')
    print(f'joint_target_norm {joint_target_norm}')
    print(f'joint_act_norm {joint_act_norm}')
    print(f'pos_diff_norm {pos_diff_norm}')
    print()
    print(f'vel_regulation {vel_regulation}')
    print()    
    print(f'joint_vel_act_norm {joint_vel_act_norm}')
    print(f'vel_etalon_norm {vel_etalon_norm}')
    print(f'joint_vel_diff_norm {joint_vel_diff_norm}')
    print(f'vel_regulation {vel_regulation}')
    print() 
    print(f'vel_reward {vel_reward}')
    print(f'pos_reward {pos_reward}')
    print(f'overall_reward {overall_reward}')
    print('#####')
    '''
    
    
   
    return overall_reward



