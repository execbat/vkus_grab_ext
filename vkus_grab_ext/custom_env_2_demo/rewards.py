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
    asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot", joint_names=[".*"]),
    #ctrl_vel_command_name: str = "override_velocity",
    target_command_name: str = "target_joint_pose",
    kv: float = 1.0,
    kp: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    # extract joint ID's 
    joint_ids = asset_cfg.joint_ids

    # Index the selected joints
    q_act = asset.data.joint_pos[:, joint_ids]                       # [N, J]
    
    device, dtype = q_act.device, q_act.dtype
    eps = 1e-6

    # Joint position limits
    qmin = asset.data.soft_joint_pos_limits[:, joint_ids, 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[:, joint_ids, 1].to(device=device, dtype=dtype)
    
    rng = (qmax - qmin).clamp_min(eps)                                         # [N, J]
    offset = 0.5 * (qmin + qmax)

    # Normalize positions to [-1, 1]
    joint_act_norm = 2.0 * (q_act - offset) / rng
    joint_act_norm = torch.clamp(joint_act_norm, -1.0, 1.0)

    # Target joint positions (assumed already in [-1, 1]); ensure shape [N, J]
    joint_target_norm = env.command_manager.get_command(target_command_name)   # [N] or [N, J]

    # Position error in normalized coordinates
    pos_diff_norm = joint_target_norm - joint_act_norm                         # [N, J]

    # Joint velocities and limits
    joint_vel_act = asset.data.joint_vel[:, joint_ids]               # [N, J]
    
    joint_vel_limits = asset.data.joint_vel_limits[:, joint_ids].to(device=device, dtype=dtype)
    
    vlim = joint_vel_limits.abs().clamp_min(eps)

    # Normalize velocities to [-1, 1]
    joint_vel_act_norm = torch.clamp(joint_vel_act / vlim, -1.0, 1.0)

    # === L2 weights (squared) ===
    # Position weights ~ (range)^2, per-env normalized so max = 1
    rng_max = rng.max(dim=-1, keepdim=True).values
    axis_pos_weights = (rng / rng_max).pow(2)                                         # [N, J]
   
    # Reference velocity (in normalized units)
    # vel_etalon_norm = torch.tanh(pos_diff_norm) * vel_regulation                      # [N, J]
    vel_etalon_norm = torch.tanh(pos_diff_norm) * 1.0                      # [N, J]

    # --- Velocity reward (mask by direction, weight-average with velocity weights) ---
    joint_vel_diff_norm = vel_etalon_norm - joint_vel_act_norm
    vel_term = torch.exp(-(joint_vel_diff_norm ** 2) / (kv ** 2)) #                   # [N, J]
    vel_reward = torch.exp(torch.mean(torch.log(vel_term.clamp_min(1e-12)), dim=-1))  # if one of the axis berforms bad, then entire result is bad                                       # [0..1]

    # --- Position reward (usually without mask: closeness to target always matters) ---
    pos_term = (torch.exp(-(pos_diff_norm** 2) / (kp ** 2)) ) * axis_pos_weights      # correction of axis importance based on different axis range limits          
    pos_reward = torch.exp(torch.mean(torch.log(pos_term.clamp_min(1e-12)), dim=-1))  # if one of the axis berforms bad, then entire result is bad

    # Final reward
    overall_reward = vel_reward * pos_reward
    return overall_reward

