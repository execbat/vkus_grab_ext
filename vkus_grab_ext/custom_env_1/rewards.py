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


'''
def velocity_profile_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vmax_norm: float = 1.0,
    beta_min: float = 0.5,
    beta_max: float = 100.0,
    std_vel: float = 0.25,
    vel_fallback: float = 1.0,
) -> torch.Tensor:
    """
    Ревард: насколько модуль скорости по осям соответствует эталонному профилю v_des(dist).
    - dist считается в нормированных координатах [-1, 1]
    - beta зависит от команды velocity в [0, 1]
    - ревард для каждого env = exp(-MSE / std_vel^2), где MSE по всем осям.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_act = asset.data.joint_vel[:, asset_cfg.joint_ids]
    print(f'joint_vel_act {joint_vel_act}')
    
    
    q     = asset.data.joint_pos        # (N, J)
    qdot  = asset.data.joint_vel        # (N, J)
    device, dtype = q.device, q.dtype
    eps = 1e-8

    N, J = q.shape

    # ---------- 1. Нормализованные позиции (ось в [-1, 1]) ----------
    # если хочешь использовать ровно то же, что в ObsTerm(axis_act_norm),
    # можно сюда подставить env.obs_manager.get_term("axis_act_norm")
    qmin = asset.data.soft_joint_pos_limits[..., 0]
    qmax = asset.data.soft_joint_pos_limits[..., 1]

    # нормализация по soft-лимитам, как в joint_pos_limit_normalized
    lo = torch.minimum(qmin, qmax)
    hi = torch.maximum(qmin, qmax)
    center = 0.5 * (hi + lo)
    half_range = 0.5 * (hi - lo)

    bad = (~torch.isfinite(half_range)) | (half_range.abs() < 1e-6)
    half_range_safe = torch.where(bad, torch.ones_like(half_range), half_range)

    qn = (q - center) / half_range_safe
    qn = torch.where(bad, torch.zeros_like(qn), qn)
    qn = torch.clamp(qn, -1.0, 1.0)      # (N, J)

    # ---------- 2. Нормализованный target ([-1, 1], как target_axis_cmd_norm) ----------
    tgt_raw = env.command_manager.get_term("target_joint_pose").command  # (N, J) или (1, J)
    # приводим к (N, J)
    if tgt_raw.shape[0] == 1 and N > 1:
        tgt = tgt_raw.expand(N, -1)
    else:
        tgt = tgt_raw
    tgt = tgt.to(device=device, dtype=dtype)     # (N, J), уже в [-1, 1]

    # ---------- 3. Команда velocity -> beta per-axis ----------
    vel_raw = env.command_manager.get_term("velocity").command  # (N, J) или (1, J)
    if vel_raw.shape[0] == 1 and N > 1:
        vel = vel_raw.expand(N, -1)
    else:
        vel = vel_raw
    vel = vel.to(device=device, dtype=dtype)
    vel = torch.clamp(vel, 0.0, 1.0)            # [0,1]

    # velocity = 1  -> beta = beta_min
    # velocity = 0  -> beta = beta_max
    beta = beta_max - vel * (beta_max - beta_min)   # (N, J)

    # ---------- 4. Эталонный профиль скорости v_des(dist) ----------
    dist = (qn - tgt).abs()                     # (N, J)
    x = dist / (1.0 + eps)
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)   # (N, J)

    # ---------- 5. Нормализуем реальные скорости и берём модуль ----------
    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)

    if vel_limits is None:
        v_lim = torch.full_like(qdot, vel_fallback)
    else:
        # приводим vel_limits к форме (N, J)
        if vel_limits.ndim == 1:
            v_lim = vel_limits.unsqueeze(0).expand(N, -1)
        else:
            v_lim = vel_limits
    v_lim = torch.clamp(v_lim, min=1e-6)

    v_act_norm = torch.clamp(qdot / v_lim, -1.0, 1.0)   # (N, J)
    v_act_abs  = v_act_norm.abs()                       # (N, J)

    # ---------- 6. Ошибка по профилю и ревард ----------
    err = v_act_abs - v_des                            # (N, J)
    mse = (err * err).mean(dim=-1)                     # (N,) среднее по всем осям робота
    
    score = torch.exp(-mse / (std_vel * std_vel + eps))  # (N,) в [0,1]

    return score



