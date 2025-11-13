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
from dataclasses import MISSING, asdict, dataclass 

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import TargetChaseVelocityCommandCfg

import socket
import threading
import json
# COMMAND DESCRIPTIONS


_HUBS: Dict[Tuple[str, int], "UdpPacketHub"] = {}

class UdpPacketHub:
    def __init__(self, ip: str, port: int, packet_format: str = "json", struct_fmt: str = "<10f"):
        self.addr = (ip, port)
        self.packet_format = packet_format.lower()
        self.struct_fmt = struct_fmt

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.setblocking(False)
        print(f"[UDP] listening on {self.addr} format={self.packet_format} fmt={self.struct_fmt}", flush=True)

        self._lock = threading.Lock()
        self._latest: Optional[dict] = None

    def poll(self) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                break
            except Exception as ex:
                print(f"[UDP] recv error: {ex}", flush=True)
                break

            print(f"[UDP] recv {len(data)} bytes on {self.addr}", flush=True)

            parsed = self._parse_packet(data)
            if parsed is None:
                print(f"[UDP] parse FAILED (format={self.packet_format})", flush=True)
                continue

            print(f"[UDP] parse OK -> keys {list(parsed.keys())}", flush=True)
            with self._lock:
                self._latest = parsed

    def _parse_packet(self, data: bytes) -> Optional[dict]:
        try:
            if self.packet_format == "json":
                msg = json.loads(data.decode("utf-8"))
                out = {}
                if "target_joint_pose" in msg:
                    out["target_joint_pose"] = torch.as_tensor(msg["target_joint_pose"], dtype=torch.float32)
                if "override_velocity" in msg:
                    v = msg["override_velocity"]
                    out["override_velocity"] = (torch.tensor([float(v)], dtype=torch.float32)
                                                if isinstance(v, (int, float))
                                                else torch.as_tensor(v, dtype=torch.float32))
                return out or None
            else:
                vals = struct.unpack(self.struct_fmt, data)
                if len(vals) >= 10:
                    return {
                        "target_joint_pose": torch.tensor(vals[:9], dtype=torch.float32),
                        "override_velocity": torch.tensor([vals[9]], dtype=torch.float32),
                    }
                return None
        except Exception as ex:
            print(f"[UDP] exception in _parse_packet: {ex}", flush=True)
            return None

    def get(self, field: str) -> Optional[torch.Tensor]:
        with self._lock:
            if self._latest is None or field not in self._latest:
                return None
            return self._latest[field].clone()

def _get_hub(ip: str, port: int, packet_format: str, struct_fmt: str) -> UdpPacketHub:
    key = (ip, port)
    hub = _HUBS.get(key)
    if hub is None:
        print(f"[UDP] binding {key} format={packet_format} struct_fmt={struct_fmt}", flush=True)
        hub = UdpPacketHub(ip, port, packet_format=packet_format, struct_fmt=struct_fmt)
        _HUBS[key] = hub
    return hub


@configclass
@dataclass
class UdpTargetJointPoseCommandCfg(CommandTermCfg):
    class_type: type = None
    asset_name: str = "robot"
    dim: int = 9
    ranges: Tuple[Tuple[float, float], ...] = ((-1.0, 1.0),) * 9
    default: float = 0.0
    ip: str = "0.0.0.0"
    port: int = 6000
    packet_format: str = "json"   # "json" or "struct"
    struct_fmt: str = "<10f"
    resampling_time_range: Tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = True      

class UdpTargetJointPoseCommand(CommandTerm):
    cfg: UdpTargetJointPoseCommandCfg
    def __init__(self, cfg: UdpTargetJointPoseCommandCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._device = self.robot.data.joint_pos.device
        self._dtype = torch.float32
        if len(cfg.ranges) != cfg.dim:
            raise ValueError("ranges length must equal dim")

        self._hub = _get_hub(cfg.ip, cfg.port, cfg.packet_format, cfg.struct_fmt)
        self._cmd = torch.full((self.num_envs, cfg.dim), float(cfg.default), dtype=self._dtype, device=self._device)
        self._low  = torch.tensor([lo for lo, _ in cfg.ranges], dtype=self._dtype, device=self._device).unsqueeze(0)
        self._high = torch.tensor([hi for _, hi in cfg.ranges], dtype=self._dtype, device=self._device).unsqueeze(0)
        self.metrics["packets_applied"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def reset(self, env_ids: Sequence[int] | None = None):
        if self.cfg.debug_vis: print("[CMD] target_pose.reset tick", flush=True)
        self._hub.poll()
        self._apply_from_udp()
        self._update_metrics()
        return {k: v for k, v in self.metrics.items()}

    def sample(self, dt: float):
        if self.cfg.debug_vis: print("[CMD] target_pose.sample tick", flush=True)
        self._hub.poll()
        self._apply_from_udp()
        self._update_metrics()
        return {k: v for k, v in self.metrics.items()}

    def compute(self, dt: float):
        if self.cfg.debug_vis: print("[CMD] target_pose.compute tick", flush=True)
        self._hub.poll()
        self._apply_from_udp()
        self._update_metrics()
        return {k: v for k, v in self.metrics.items()}

    def _resample_command(self, env_ids: Sequence[int]):  # no-op
        return

    def _update_command(self):  # not relied upon — логика в sample/compute
        self._apply_from_udp()

    def _update_metrics(self):
        with torch.no_grad():
            self.metrics["mean_abs_target"] = self._cmd.abs().mean(dim=1)

    def _apply_from_udp(self):
        vec_cpu = self._hub.get("target_joint_pose")
        if vec_cpu is None:
            return
        vec = vec_cpu.to(device=self._device, dtype=self._dtype).view(-1)
        if vec.numel() != self.cfg.dim:
            print(f"[CMD] target_pose wrong length {vec.numel()} != {self.cfg.dim}", flush=True)
            return
        if self.cfg.debug_vis:
            print(f"[CMD] target_pose apply: {vec.tolist()}", flush=True)
        vec = torch.clamp(vec, self._low.squeeze(0), self._high.squeeze(0))
        self._cmd[:] = vec.unsqueeze(0).expand(self.num_envs, -1)
        self.metrics["packets_applied"] += 1.0
        print(f"[CMD] target_pose apply: {vec.tolist()}", flush=True)

UdpTargetJointPoseCommandCfg.class_type = UdpTargetJointPoseCommand


@configclass
@dataclass
class UdpOverrideVelocityCommandCfg(CommandTermCfg):
    class_type: type = None
    asset_name: str = "robot"
    dim: int = 1
    ranges: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)
    default: float = 0.0
    ip: str = "0.0.0.0"
    port: int = 6000
    packet_format: str = "json"
    struct_fmt: str = "<10f"
    resampling_time_range: Tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = True

class UdpOverrideVelocityCommand(CommandTerm):
    cfg: UdpOverrideVelocityCommandCfg
    def __init__(self, cfg: UdpOverrideVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._device = self.robot.data.joint_pos.device
        self._dtype = torch.float32
        if len(cfg.ranges) != cfg.dim:
            raise ValueError("ranges length must equal dim")
        self._hub = _get_hub(cfg.ip, cfg.port, cfg.packet_format, cfg.struct_fmt)
        self._cmd = torch.full((self.num_envs, cfg.dim), float(cfg.default), dtype=self._dtype, device=self._device)
        self._low  = torch.tensor([lo for lo, _ in cfg.ranges], dtype=self._dtype, device=self._device).unsqueeze(0)
        self._high = torch.tensor([hi for _, hi in cfg.ranges], dtype=self._dtype, device=self._device).unsqueeze(0)
        self.metrics["packets_applied"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def reset(self, env_ids: Sequence[int] | None = None):
        if self.cfg.debug_vis: print("[CMD] override.reset tick", flush=True)
        self._hub.poll()
        self._apply_from_udp()
        self._update_metrics()
        return {k: v for k, v in self.metrics.items()}

    def sample(self, dt: float):
        if self.cfg.debug_vis: print("[CMD] override.sample tick", flush=True)
        self._hub.poll()
        self._apply_from_udp()
        self._update_metrics()
        return {k: v for k, v in self.metrics.items()}

    def compute(self, dt: float):
        if self.cfg.debug_vis: print("[CMD] override.compute tick", flush=True)
        self._hub.poll()
        self._apply_from_udp()
        self._update_metrics()
        return {k: v for k, v in self.metrics.items()}

    def _resample_command(self, env_ids: Sequence[int]):  # no-op
        return

    def _update_command(self):
        self._apply_from_udp()

    def _update_metrics(self):
        with torch.no_grad():
            self.metrics["override_velocity_mean"] = self._cmd.mean(dim=1)

    def _apply_from_udp(self):
        vec_cpu = self._hub.get("override_velocity")
        if vec_cpu is None:
            return
        vec = vec_cpu.to(device=self._device, dtype=self._dtype).view(-1)
        if vec.numel() == 1 and self.cfg.dim > 1:
            vec = vec.repeat(self.cfg.dim)
        if vec.numel() != self.cfg.dim:
            print(f"[CMD] override wrong length {vec.numel()} != {self.cfg.dim}", flush=True)
            return
        if self.cfg.debug_vis:
            print(f"[CMD] override apply: {vec.tolist()}", flush=True)
        vec = torch.clamp(vec, self._low.squeeze(0), self._high.squeeze(0))
        self._cmd[:] = vec.unsqueeze(0).expand(self.num_envs, -1)
        self.metrics["packets_applied"] += 1.0
        print(f"[CMD] override apply: {vec.tolist()}", flush=True)

UdpOverrideVelocityCommandCfg.class_type = UdpOverrideVelocityCommand
