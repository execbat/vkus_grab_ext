from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from dataclasses import MISSING, asdict  

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import TargetChaseVelocityCommandCfg


# COMMAND DESCRIPTIONS


# ───────────────────────────── 1. UNIFORM VECTOR ──────────────────────────────
class UniformVectorCommand(CommandTerm):
    """N-dimensional command vector; new U(-1,1) on reset, unchanged in step."""

    cfg: "UniformVectorCommandCfg"

    # ---------------------------------------------------------------- init
    def __init__(self, cfg: "UniformVectorCommandCfg", env: ManagerBasedEnv):
        # disable automatic resampling between steps
        if math.isinf(cfg.resampling_time_range[0]):
            cfg.resampling_time_range = (0.0, 0.0)

        super().__init__(cfg, env)
        
        if cfg.dim is MISSING:
            raise ValueError("UniformVectorCommandCfg.dim должен быть задан")

        self.dim = cfg.dim

        # -------- ranges a_i,b_i ----------------------------------
        if isinstance(cfg.ranges, tuple):              
            pairs = list(cfg.ranges)
        elif cfg.ranges is MISSING or cfg.ranges is None:
            pairs = [(-1.0, 1.0)] * cfg.dim            
        else:                                          
            pairs = list(asdict(cfg.ranges).values())

        if cfg.dim is MISSING:
            cfg.dim = len(pairs)

        if len(pairs) != cfg.dim:
            raise ValueError("len(ranges) != dim в UniformVectorCommandCfg")

        self.low  = torch.tensor([p[0] for p in pairs], device=self.device)
        self.high = torch.tensor([p[1] for p in pairs], device=self.device)

        # -------- command buffer --------------------------------------
        self._command = torch.empty(self.num_envs, cfg.dim, device=self.device)
        self._resample_command(torch.arange(self.num_envs, device=self.device))


    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass                                           

    def _resample_command(self, env_ids: Sequence[int]):
        """Generates new targets for the given env_ids (reset)."""
        if len(env_ids) == 0:
            return

        self._command[env_ids] = 2.0 * torch.rand((len(env_ids), self.dim), device=self.device) - 1.0


    def _update_command(self):
        """We don't change anything between steps – the command is constant"""
        pass
        
        
# ───────────────────────────── 2. BERNOULLI MASK ─────────────────────────────
class BernoulliMaskCommand(CommandTerm):
    """Binary mask (N_envs, dim):
    • generated on reset: u~U(0,1) → (u < p) ? 1.0 : 0.0
    • unchanged during the episode
    • p is taken from env.MASK_PROB_LEVEL (if present), otherwise cfg.p_one
    """
    cfg: "BernoulliMaskCommandCfg"

    def __init__(self, cfg: "BernoulliMaskCommandCfg", env: "ManagerBasedEnv"):
        if (math.isinf(cfg.resampling_time_range[0])
                or math.isinf(cfg.resampling_time_range[1])):
            cfg.resampling_time_range = (1e9, 1e9)

        super().__init__(cfg, env)

        self._command = torch.zeros(self.num_envs, cfg.dim, device=self.device)
        # init
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass  

    def _current_threshold(self) -> float:
        
        return float(getattr(self.cfg, "p_one", 0.5))

    def _resample_command(self, env_ids: Sequence[int]):
        """Generated only when reset(env_ids)."""
        if len(env_ids) == 0:
            return
        p = self._current_threshold()
        u = torch.rand((len(env_ids), self.cfg.dim), device=self.device)
        self._command[env_ids] = (u < p).float()  # strictly 0.0 or 1.0

    def _update_command(self):
        "The mask does not change during the episode."""
        pass
        
        
        
        
# COMMAND CONFIGS
@configclass
class BernoulliMaskCommandCfg(CommandTermCfg):
    """Configuration for BernoulliMaskCommand."""
    class_type: type = BernoulliMaskCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.
    
    This parameter is only used if :attr:`heading_command` is True.
    """
    dim: int = MISSING 
    
    p_one: float  = 0.5
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        tgt_mask_ax_1: tuple[float, float] = MISSING
        tgt_mask_ax_2: tuple[float, float] = MISSING
        tgt_mask_ax_3: tuple[float, float] = MISSING
        tgt_mask_ax_4: tuple[float, float] = MISSING
        tgt_mask_ax_5: tuple[float, float] = MISSING
        tgt_mask_ax_6: tuple[float, float] = MISSING
        tgt_mask_ax_7: tuple[float, float] = MISSING
        tgt_mask_ax_8: tuple[float, float] = MISSING
        tgt_mask_ax_9: tuple[float, float] = MISSING

    ranges: tuple[tuple[float, float], ...] = MISSING
    """Distribution ranges for the velocity commands."""
    
    
@configclass
class UniformVectorCommandCfg(CommandTermCfg):
    """Configuration for UniformVectorCommand."""
    class_type: type = UniformVectorCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """
    dim: int = MISSING 
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        tgt_norm_ax_1: tuple[float, float] = MISSING
        tgt_norm_ax_2: tuple[float, float] = MISSING
        tgt_norm_ax_3: tuple[float, float] = MISSING
        tgt_norm_ax_4: tuple[float, float] = MISSING
        tgt_norm_ax_5: tuple[float, float] = MISSING
        tgt_norm_ax_6: tuple[float, float] = MISSING
        tgt_norm_ax_7: tuple[float, float] = MISSING
        tgt_norm_ax_8: tuple[float, float] = MISSING
        tgt_norm_ax_9: tuple[float, float] = MISSING

    ranges: tuple[tuple[float, float], ...] = MISSING
    """Distribution ranges for the velocity commands."""        
