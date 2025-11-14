import json
import socket
import struct
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import numpy as np

# -----------------------------------------------------------------------------
# Networking (send UDP packets to your Isaac Lab listener)
# -----------------------------------------------------------------------------
UDP_IP   = "127.0.0.1" 
UDP_PORT = 55001                 # set this to the port your env listens on
PACKET_FORMAT = "json"          # "json" or "struct"
STRUCT_FMT    = "<10f"          # 9 poses + 1 speed (all float32)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ADDR = (UDP_IP, UDP_PORT)

# -----------------------------------------------------------------------------
# GUI constants
# -----------------------------------------------------------------------------
# 9-DOF robot (example joint names for a Franka Panda; adjust as needed)
JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]
NUM_AXES = len(JOINT_NAMES)     # 9

STREAM_INTERVAL_MS = 10        # send every 100 ms

SLIDER_MIN, SLIDER_MAX = -1.0, 1.0    # axis targets in [-1, 1]
SPEED_MIN,  SPEED_MAX  =  0.0, 1.0    # override speed in [0, 1]


class AxisControlApp(tk.Tk):
    """Send 9 axis targets (-1..1) and 1 override speed (0..1) over UDP."""

    def __init__(self):
        super().__init__()
        self.title("Axis Control (9 DOF + Override Speed)")
        self.geometry("900x700")
        self.tk.call("tk", "scaling", 2.0)  # scale up widgets for HiDPI

        # Increase default font globally
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=16)
        self.option_add("*Font", default_font)

        self._make_vars()
        self._build_ui()
        self.after(STREAM_INTERVAL_MS, self._stream_packet)

    # ----------------------------- state vars --------------------------------
    def _make_vars(self):
        # 9 axis sliders in [-1, 1]
        self.slider_vars = [tk.DoubleVar(value=0.0) for _ in range(NUM_AXES)]
        # single override speed in [0, 1]
        self.speed_var = tk.DoubleVar(value=0.0)

    # ------------------------------- UI --------------------------------------
    def _build_ui(self):
        header_font = ("Helvetica", 20, "bold")

        ttk.Label(self, text="Set target per axis ([-1, 1])", font=header_font)\
            .grid(row=0, column=0, columnspan=3, pady=(10, 10), sticky="w")

        # 9 labeled sliders for joint targets
        for i, name in enumerate(JOINT_NAMES):
            row = i + 1
            ttk.Label(self, text=name).grid(row=row, column=0, sticky="w", padx=(10, 8))
            ttk.Scale(
                self, from_=SLIDER_MIN, to=SLIDER_MAX, orient="horizontal",
                length=500, variable=self.slider_vars[i]
            ).grid(row=row, column=1, padx=10, pady=6, sticky="we")
            # live value label
            val_lbl = ttk.Label(self, textvariable=self.slider_vars[i], width=6)
            val_lbl.grid(row=row, column=2, sticky="e", padx=(8, 10))

        # make the slider column stretch with the window
        self.columnconfigure(1, weight=1)

        # override speed slider
        base = NUM_AXES + 1
        ttk.Label(self, text="Override speed ([0, 1])", font=header_font)\
            .grid(row=base, column=0, columnspan=3, pady=(25, 6), sticky="w")

        ttk.Label(self, text="speed").grid(row=base + 1, column=0, sticky="w", padx=(10, 8))
        ttk.Scale(
            self, from_=SPEED_MIN, to=SPEED_MAX, orient="horizontal",
            length=500, variable=self.speed_var
        ).grid(row=base + 1, column=1, padx=10, pady=6, sticky="we")
        ttk.Label(self, textvariable=self.speed_var, width=6)\
            .grid(row=base + 1, column=2, sticky="e", padx=(8, 10))

        # footer
        ttk.Label(
            self,
            text=f"Sending {PACKET_FORMAT.upper()} every {STREAM_INTERVAL_MS} ms to {UDP_IP}:{UDP_PORT}"
        ).grid(row=base + 3, column=0, columnspan=3, pady=(20, 10), sticky="w")

    # -------------------------- payload builders ------------------------------
    def _collect_axes(self):
        """Return a list of 9 floats in [-1, 1] (axis targets)."""
        vals = [float(v.get()) for v in self.slider_vars]
        # clamp defensively (GUI should already constrain)
        vals = [max(SLIDER_MIN, min(SLIDER_MAX, x)) for x in vals]
        return vals

    def _collect_speed(self):
        """Return a single float in [0, 1] (override speed)."""
        v = float(self.speed_var.get())
        return max(SPEED_MIN, min(SPEED_MAX, v))

    def _build_packet_bytes(self) -> bytes:
        """Serialize either JSON or struct: 9 poses + 1 speed."""
        poses = self._collect_axes()                 # list of 9 floats
        speed = self._collect_speed()                # single float

        if PACKET_FORMAT.lower() == "json":
            payload = {
                "target_joint_pose": poses,          # length 9
                "override_velocity": speed,          # scalar
            }
            return json.dumps(payload).encode("utf-8")
        elif PACKET_FORMAT.lower() == "struct":
            vec = np.asarray(poses + [speed], dtype=np.float32)  # 10 floats
            # ensure STRUCT_FMT matches 10 float32: "<10f"
            return struct.pack(STRUCT_FMT, *vec.tolist())
        else:
            raise ValueError(f"Unsupported PACKET_FORMAT: {PACKET_FORMAT}")

    # ------------------------------ UDP loop ----------------------------------
    def _stream_packet(self):
        """Periodically send the latest command packet via UDP."""
        try:
            pkt = self._build_packet_bytes()
            sock.sendto(pkt, ADDR)
        except OSError as ex:
            print(f"[UDP] send failed: {ex}")
        except Exception as ex:
            print(f"[UDP] pack failed: {ex}")
        
        print(pkt)
        self.after(STREAM_INTERVAL_MS, self._stream_packet)


# ----------------------------------------------------------------------
def run_gui() -> None:
    AxisControlApp().mainloop()


if __name__ == "__main__":
    run_gui()

