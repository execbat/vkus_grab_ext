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
    sign_deadband: float = 1e-2,
    k_in_position: float = 2.0, # additional reward weight for being inside of the deadband
    k_moving_away: float = 0.1, # additional penalty weight for going away from the target
    min_vel_threshold: float = 0.1 # if cmd vel is lower thatn this value, robot shouldn't move
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # Index the selected joints
    q_act = asset.data.joint_pos[:, asset_cfg.joint_ids]                       # [N, J]
    device, dtype = q_act.device, q_act.dtype
    eps = 1e-6

    # Joint position limits
    qmin = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1].to(device=device, dtype=dtype)
    rng = (qmax - qmin).clamp_min(eps)                                         # [N, J]
    offset = 0.5 * (qmin + qmax)

    # Normalize positions to [-1, 1]
    joint_act_norm = 2.0 * (q_act - offset) / rng
    joint_act_norm = torch.clamp(joint_act_norm, -1.0, 1.0)

    # Target joint positions (assumed already in [-1, 1]); ensure shape [N, J]
    joint_target_norm = env.command_manager.get_command(target_command_name)   # [N] or [N, J]
    if joint_target_norm.dim() == 1:
        joint_target_norm = joint_target_norm.unsqueeze(-1).expand_as(joint_act_norm)
    else:
        joint_target_norm = joint_target_norm[:, asset_cfg.joint_ids]

    # Position error in normalized coordinates
    pos_diff_norm = joint_target_norm - joint_act_norm                         # [N, J]

    # Joint velocities and limits
    joint_vel_act = asset.data.joint_vel[:, asset_cfg.joint_ids]               # [N, J]
    joint_vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids].to(device=device, dtype=dtype)
    vlim = joint_vel_limits.abs().clamp_min(eps)

    # Normalize velocities to [-1, 1]
    joint_vel_act_norm = torch.clamp(joint_vel_act / vlim, -1.0, 1.0)

    # === L2 weights (squared) ===
    # Position weights ~ (range)^2, per-env normalized so max = 1
    rng_max = rng.max(dim=-1, keepdim=True).values
    axis_pos_weights = (rng / rng_max).pow(2)                                  # [N, J]

    # Velocity weights ~ (v_max)^2, per-env normalized so max = 1
    vlim_max = vlim.max(dim=-1, keepdim=True).values
    axis_vel_weights = (vlim / vlim_max).pow(2)                                # [N, J]

    # Global velocity regulation command per env
    vel_regulation = env.command_manager.get_command(ctrl_vel_command_name)    # [N] or [N, 1]
    if vel_regulation.dim() == 1:
        vel_regulation = vel_regulation.unsqueeze(-1)
        
    must_stand_still = (vel_regulation < min_vel_threshold) * -1.0 # envs, where robots shouldn't move
    must_move = (vel_regulation >= min_vel_threshold) * 1.0       # envs, where robots should move
        
    vel_regulation = vel_regulation.expand_as(pos_diff_norm)

    # === Mask: "moving towards target" — compare position error sign with VELOCITY sign ===
    same_sign = (torch.sign(pos_diff_norm) == torch.sign(joint_vel_act_norm))
    diff_sign = (torch.sign(pos_diff_norm) != torch.sign(joint_vel_act_norm))
    near_zero = pos_diff_norm.abs() <= sign_deadband   # | (joint_vel_act_norm.abs() <= sign_deadband)
    on_path_mask = (same_sign | near_zero).to(dtype=dtype)                     # [N, J]
    
    
    
    in_position_reward = near_zero * k_in_position
    moving_away_penalty = diff_sign * -1 * k_moving_away + same_sign * 1
    

    # Reference velocity (in normalized units)
    vel_etalon_norm = torch.tanh(pos_diff_norm) * vel_regulation               # [N, J]

    # --- Velocity reward (mask by direction, weight-average with velocity weights) ---
    joint_vel_diff_norm = vel_etalon_norm - joint_vel_act_norm
#    vel_term = torch.exp(-(joint_vel_diff_norm ** 2) / (kv ** 2)) # * moving_away_penalty             # [N, J]
    vel_term = (torch.exp(-(joint_vel_diff_norm** 2) / (kv ** 2)) ) * axis_vel_weights
#    vel_num = (vel_term *  axis_vel_weights).sum(dim=-1)                       # [N]      # removed on_path_mask
#    vel_den = axis_vel_weights.sum(dim=-1).clamp_min(eps)                      # [N]      # removed on_path_mask
    
    vel_reward = vel_term.mean(dim=-1)  #vel_num / vel_den                                             # [0..1]
    #vel_penalty = (joint_vel_act_norm.abs() * must_stand_still).mean(dim=-1) * 0.1
    

    # --- Position reward (usually without mask: closeness to target always matters) ---
#    pos_term = torch.exp(-(pos_diff_norm ** 2) / (kp ** 2))     # + in_position_reward                   # [N, J]
    pos_term = (torch.exp(-(pos_diff_norm** 2) / (kp ** 2)) ) * axis_pos_weights   # correction of axis importance based on different axis range limits
#    pos_num = (pos_term * axis_pos_weights).sum(dim=-1)                        # [N]      # removed on_path_mask
#    pos_den = axis_pos_weights.sum(dim=-1).clamp_min(eps)                      # [N]      # removed on_path_mask
    pos_reward = pos_term.mean(dim=-1)  #pos_num / pos_den                                             # [0..1]

    # Final reward
    overall_reward = vel_reward * pos_reward
    return overall_reward

