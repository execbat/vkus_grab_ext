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
PACKET_FORMAT = "json"           # "json" or "struct"
STRUCT_FMT    = "<12f"           # 12 poses (all float32)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ADDR = (UDP_IP, UDP_PORT)

# -----------------------------------------------------------------------------
# GUI constants
# -----------------------------------------------------------------------------
JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "left_inner_knuckle_joint",
    "left_outer_knuckle_joint",
    "left_finger_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
]
NUM_AXES = len(JOINT_NAMES)     # 12

STREAM_INTERVAL_MS = 2          # send every 2 ms

SLIDER_MIN, SLIDER_MAX = -1.0, 1.0    # axis targets in [-1, 1]


class AxisControlApp(tk.Tk):
    """Send 12 axis targets (-1..1) over UDP."""

    def __init__(self):
        super().__init__()
        self.title("Axis Control (12 DOF)")
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
        # 12 axis sliders in [-1, 1]
        self.slider_vars = [tk.DoubleVar(value=0.0) for _ in range(NUM_AXES)]

    # ------------------------------- UI --------------------------------------
    def _build_ui(self):
        header_font = ("Helvetica", 20, "bold")

        ttk.Label(self, text="Set target per axis ([-1, 1])", font=header_font)\
            .grid(row=0, column=0, columnspan=3, pady=(10, 10), sticky="w")

        # 12 labeled sliders for joint targets
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

        # footer
        footer_row = NUM_AXES + 2
        ttk.Label(
            self,
            text=f"Sending {PACKET_FORMAT.upper()} every {STREAM_INTERVAL_MS} ms to {UDP_IP}:{UDP_PORT}"
        ).grid(row=footer_row, column=0, columnspan=3, pady=(20, 10), sticky="w")

    # -------------------------- payload builders ------------------------------
    def _collect_axes(self):
        """Return a list of 12 floats in [-1, 1] (axis targets)."""
        vals = [float(v.get()) for v in self.slider_vars]
        # clamp defensively (GUI should already constrain)
        vals = [max(SLIDER_MIN, min(SLIDER_MAX, x)) for x in vals]
        return vals

    def _build_packet_bytes(self) -> bytes:
        """Serialize either JSON or struct: 12 poses."""
        poses = self._collect_axes()                 # list of 12 floats

        if PACKET_FORMAT.lower() == "json":
            payload = {
                "target_joint_pose": poses,          # length 12
            }
            return json.dumps(payload).encode("utf-8")
        elif PACKET_FORMAT.lower() == "struct":
            vec = np.asarray(poses, dtype=np.float32)  # 12 floats
            # ensure STRUCT_FMT matches 12 float32: "<12f"
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

