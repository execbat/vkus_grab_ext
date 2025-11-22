import json
import socket
import struct
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import numpy as np
import copy

# -----------------------------------------------------------------------------
# Networking (send UDP packets to your Isaac Lab listener)
# -----------------------------------------------------------------------------
UDP_IP   = "127.0.0.1"
UDP_PORT = 55001                 # set this to the port your env listens on
PACKET_FORMAT = "json"           # "json" or "struct"
# формат под кол-во осей:
# будет "<7f" для 7 осей
STRUCT_FMT    = "<7f"

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
    "gripper",
]
NUM_AXES = len(JOINT_NAMES)     # 7

STREAM_INTERVAL_MS = 10         # отправляем каждые 10 ms (100 Гц)
SLIDER_MIN, SLIDER_MAX = -1.0, 1.0    # axis targets in [-1, 1]

# начальная нейтральная поза (все 0)
INIT_BASELINE = np.zeros(NUM_AXES, dtype=np.float32)

# ============================== UTILS ========================================

def smoothstep(t: np.ndarray) -> np.ndarray:
    """3t^2 - 2t^3, монотонная S-кривая, нулевые производные на концах."""
    return t * t * (3.0 - 2.0 * t)

def resample_track(y_old: np.ndarray, new_T: int) -> np.ndarray:
    """Линейный ресэмпл в новый размер."""
    T_old = len(y_old)
    if T_old == new_T:
        return y_old.copy()
    x_old = np.linspace(0.0, 1.0, T_old, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, new_T, dtype=np.float32)
    return np.interp(x_new, x_old, y_old).astype(np.float32)

# =============================== PATTERN EDITOR ==============================

class PatternEditor(tk.Toplevel):
    """
    Редактор паттернов: 7 полос-канвасов (по числу осей),
    редактирование кликом/перетаскиванием.
    Между пинами строится плавный переход (smoothstep).
    Начало/конец — baseline.
    """

    CANVAS_W = 900
    ROW_H    = 80
    PAD_Y    = 6

    def __init__(
        self,
        parent,
        *,
        baseline_vec,
        on_save_callback,
        initial_tracks=None,      # np.ndarray [T, NUM_AXES] или None
        initial_pins=None,        # list[set] или None
        initial_duration_s=None   # float или None
    ):
        super().__init__(parent)
        self.title("Pattern Editor")
        self.transient(parent)
        self.grab_set()

        self.parent = parent
        self.on_save_callback = on_save_callback

        # baseline (NUM_AXES,)
        self.baseline = np.asarray(baseline_vec, dtype=np.float32)

        # state
        if initial_tracks is not None:
            self.tracks = initial_tracks.astype(np.float32).copy()   # [T, NUM_AXES]
            self.T = self.tracks.shape[0]
            self.duration_s = tk.DoubleVar(
                value=float(initial_duration_s if initial_duration_s is not None
                            else self.T * STREAM_INTERVAL_MS / 1000.0)
            )
            if initial_pins is not None:
                self.pins = [set(p) for p in initial_pins]
                for a in range(NUM_AXES):
                    self.pins[a].add(0)
                    self.pins[a].add(self.T - 1)
            else:
                self.pins = [set([0, self.T - 1]) for _ in range(NUM_AXES)]
        else:
            self.duration_s = tk.DoubleVar(value=5.0)
            self.T = max(3, int(round(self.duration_s.get() * 1000.0 / STREAM_INTERVAL_MS)))
            self.tracks = np.tile(self.baseline[None, :], (self.T, 1)).astype(np.float32)
            self.pins = [set([0, self.T - 1]) for _ in range(NUM_AXES)]

        self._build_ui()
        self._redraw_all()

    # -------------------- UI --------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Duration (s):").pack(side="left")
        ttk.Entry(top, textvariable=self.duration_s, width=6).pack(side="left", padx=(4, 12))
        ttk.Button(top, text="Apply", command=self._apply_duration).pack(side="left")

        ttk.Button(top, text="Reset Axis", command=self._reset_selected).pack(side="left", padx=(10, 0))
        ttk.Button(top, text="Reset All", command=self._reset_all).pack(side="left", padx=(6, 0))

        ttk.Button(top, text="Save & Play", command=self._save_and_play).pack(side="right")

        # scroll area with canvases
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.scroll = tk.Canvas(outer, highlightthickness=0,
                                height=NUM_AXES * (self.ROW_H + self.PAD_Y) // 2)
        self.scroll.pack(side="left", fill="both", expand=True)

        vs = ttk.Scrollbar(outer, orient="vertical", command=self.scroll.yview)
        vs.pack(side="right", fill="y")
        self.scroll.configure(yscrollcommand=vs.set)

        self.rows_holder = ttk.Frame(self.scroll)
        self.scroll.create_window((0, 0), window=self.rows_holder, anchor="nw")
        self.rows_holder.bind(
            "<Configure>",
            lambda e: self.scroll.configure(scrollregion=self.scroll.bbox("all")),
        )

        # per-axis canvases
        self.canvases = []
        self.active_axis = 0
        for a in range(NUM_AXES):
            row = ttk.Frame(self.rows_holder)
            row.pack(fill="x", pady=(0, self.PAD_Y))

            ttk.Label(row, text=JOINT_NAMES[a], width=20).pack(side="left", padx=(0, 8))
            c = tk.Canvas(
                row,
                width=self.CANVAS_W,
                height=self.ROW_H,
                bg="#0e0e10",
                highlightthickness=2,
            )
            c.pack(side="left", fill="x", expand=True)
            c.configure(highlightbackground="#2a2a2d", highlightcolor="#4ea1ff")
            c.bind("<Enter>", lambda e, ax=a: self._set_active_axis(ax))
            c.bind("<Button-1>", lambda e, ax=a: self._on_click_drag(ax, e))
            c.bind("<B1-Motion>", lambda e, ax=a: self._on_click_drag(ax, e))
            c.bind("<ButtonRelease-1>", lambda e, ax=a: self._on_release(ax, e))
            self.canvases.append(c)

    # -------------------- helpers --------------------
    def _set_active_axis(self, a):
        self.active_axis = a

    def _apply_duration(self):
        new_d = max(0.1, float(self.duration_s.get()))
        self.duration_s.set(new_d)
        new_T = max(3, int(round(new_d * 1000.0 / STREAM_INTERVAL_MS)))
        if new_T == self.T:
            return
        old_T = self.T
        self.tracks = np.stack(
            [resample_track(self.tracks[:, a], new_T) for a in range(NUM_AXES)],
            axis=1,
        )
        new_pins = []
        for a in range(NUM_AXES):
            scaled = {int(round(p / (old_T - 1) * (new_T - 1))) for p in self.pins[a]}
            scaled.add(0)
            scaled.add(new_T - 1)
            new_pins.append(scaled)
        self.pins = new_pins
        self.T = new_T
        self._redraw_all()

    # координаты
    def _x_to_i(self, x):
        x = np.clip(x, 0, self.CANVAS_W - 1)
        i = int(round(x / (self.CANVAS_W - 1) * (self.T - 1)))
        return int(np.clip(i, 0, self.T - 1))

    def _y_to_val(self, y):
        y = np.clip(y, 0, self.ROW_H - 1)
        v = 1.0 - 2.0 * (y / (self.ROW_H - 1))
        return float(np.clip(v, -1.0, 1.0))

    def _val_to_y(self, v):
        v = float(np.clip(v, -1.0, 1.0))
        y = (1.0 - v) * 0.5 * (self.ROW_H - 1)
        return y

    # монотонный сегмент между двумя пинами
    def _fill_segment_monotone(self, axis, i0, i1):
        if i1 <= i0:
            return
        y0 = self.tracks[i0, axis]
        y1 = self.tracks[i1, axis]
        n = i1 - i0
        xs = np.arange(0, n + 1, dtype=np.float32) / float(n)
        s = smoothstep(xs)
        self.tracks[i0 : i1 + 1, axis] = y0 + (y1 - y0) * s

    def _rebuild_axis_from_pins(self, axis):
        self.pins[axis].add(0)
        self.pins[axis].add(self.T - 1)
        self.tracks[0, axis] = self.baseline[axis]
        self.tracks[-1, axis] = self.baseline[axis]
        pins_sorted = sorted(self.pins[axis])
        for k in range(len(pins_sorted) - 1):
            i0, i1 = pins_sorted[k], pins_sorted[k + 1]
            self._fill_segment_monotone(axis, i0, i1)

    # отрисовка
    def _draw_axis(self, a):
        c = self.canvases[a]
        c.delete("all")
        # линия y=0
        y0 = self._val_to_y(0.0)
        c.create_line(0, y0, self.CANVAS_W, y0, fill="#2a2a2d")

        # baseline
        yb = self._val_to_y(self.baseline[a])
        c.create_line(0, yb, self.CANVAS_W, yb, fill="#2d6cdf", dash=(4, 3))

        stride = max(1, self.T // 1000)
        pts = []
        for i in range(0, self.T, stride):
            x = i / (self.T - 1) * (self.CANVAS_W - 1)
            y = self._val_to_y(self.tracks[i, a])
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#e5e5e5", width=2, smooth=True, splinesteps=12)

        # пины
        for i in self.pins[a]:
            x = i / (self.T - 1) * (self.CANVAS_W - 1)
            y = self._val_to_y(self.tracks[i, a])
            r = 3
            c.create_oval(x - r, y - r, x + r, y + r, fill="#ffffff", outline="")
        c.configure(
            highlightbackground=("#4ea1ff" if a == self.active_axis else "#2a2a2d")
        )

    def _redraw_all(self):
        for a in range(NUM_AXES):
            self._draw_axis(a)

    # события
    def _on_click_drag(self, axis, event):
        self.active_axis = axis
        i = self._x_to_i(event.x)
        v = self._y_to_val(event.y)
        self.tracks[i, axis] = v
        self.pins[axis].add(i)
        self.pins[axis].add(0)
        self.pins[axis].add(self.T - 1)
        self.tracks[0, axis] = self.baseline[axis]
        self.tracks[-1, axis] = self.baseline[axis]
        self._rebuild_axis_from_pins(axis)
        self._draw_axis(axis)

    def _on_release(self, axis, _event):
        pass

    # reset/save
    def _reset_selected(self):
        a = self.active_axis
        self.tracks[:, a] = self.baseline[a]
        self.pins[a] = set([0, self.T - 1])
        self._draw_axis(a)

    def _reset_all(self):
        for a in range(NUM_AXES):
            self.tracks[:, a] = self.baseline[a]
            self.pins[a] = set([0, self.T - 1])
        self._redraw_all()

    def _save_and_play(self):
        traj = self.tracks.copy()
        traj[0, :] = self.baseline
        traj[-1, :] = self.baseline
        self.on_save_callback(
            traj,
            float(self.duration_s.get()),
            copy.deepcopy(self.pins),
            self.baseline.copy(),
        )
        self.destroy()

# ================================ MAIN APP ===================================

class AxisControlApp(tk.Tk):
    """Отправляет 7-осевой вектор каждые STREAM_INTERVAL_MS.
       Паттерн синхронизирует ползунки во время воспроизведения.
    """

    def __init__(self):
        super().__init__()
        self.title("Axis Control (7 DOF + Patterns)")
        self.geometry("900x700")
        self.tk.call("tk", "scaling", 2.0)  # scale up widgets for HiDPI

        # Increase default font globally
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=16)
        self.option_add("*Font", default_font)

        # baseline
        self.global_baseline = INIT_BASELINE.copy()

        # live vars
        self._make_vars()

        # pattern playback state
        self.pattern_active = False
        self.pattern_targets = None   # np.ndarray [T, NUM_AXES]
        self.pattern_index = 0
        self.pattern_total = 0

        # editor persistent state
        self.editor_tracks = None      # np.ndarray [T, NUM_AXES]
        self.editor_pins = None        # list[set]
        self.editor_duration_s = None  # float

        self._build_ui()
        self.after(STREAM_INTERVAL_MS, self._stream_packet)

    # ----------------------------- state vars --------------------------------
    def _make_vars(self):
        # 7 axis sliders in [-1, 1]
        self.slider_vars = [
            tk.DoubleVar(value=float(self.global_baseline[i])) for i in range(NUM_AXES)
        ]

    # ------------------------------- UI --------------------------------------
    def _build_ui(self):
        header_font = ("Helvetica", 20, "bold")

        # верхняя панель: заголовок + кнопка редактора + прогрессбар
        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=3, sticky="we", padx=10, pady=(8, 4))

        ttk.Label(top, text="Set target per axis ([-1, 1])", font=header_font).pack(
            side="left"
        )
        ttk.Button(top, text="Pattern editor", command=self._open_pattern_editor).pack(
            side="left", padx=(12, 0)
        )

        self.pb = ttk.Progressbar(
            self, orient="horizontal", mode="determinate", length=500
        )
        self.pb.grid(row=1, column=0, columnspan=3, pady=(0, 8), padx=10, sticky="we")

        # sliders
        for i, name in enumerate(JOINT_NAMES):
            row = i + 2
            ttk.Label(self, text=name).grid(
                row=row, column=0, sticky="w", padx=(10, 8)
            )
            ttk.Scale(
                self,
                from_=SLIDER_MIN,
                to=SLIDER_MAX,
                orient="horizontal",
                length=500,
                variable=self.slider_vars[i],
            ).grid(row=row, column=1, padx=10, pady=6, sticky="we")
            val_lbl = ttk.Label(self, textvariable=self.slider_vars[i], width=6)
            val_lbl.grid(row=row, column=2, sticky="e", padx=(8, 10))

        # make the slider column stretch with the window
        self.columnconfigure(1, weight=1)

        # footer
        footer_row = NUM_AXES + 3
        ttk.Label(
            self,
            text=f"Sending {PACKET_FORMAT.upper()} every {STREAM_INTERVAL_MS} ms to {UDP_IP}:{UDP_PORT}",
        ).grid(row=footer_row, column=0, columnspan=3, pady=(20, 10), sticky="w")

    # -------------------------- pattern editor --------------------------
    def _open_pattern_editor(self):
        # baseline редактора — текущие значения слайдеров (или global_baseline, если хочешь)
        baseline = np.array(
            [self.slider_vars[i].get() for i in range(NUM_AXES)], dtype=np.float32
        )

        def on_save(traj_TA, duration_s, pins, baseline_out):
            # сохраняем траекторию и запускаем воспроизведение
            self.pattern_targets = traj_TA.astype(np.float32)
            self.pattern_total = int(traj_TA.shape[0])
            self.pattern_index = 0
            self.pattern_active = True
            self.pb.configure(maximum=self.pattern_total, value=0)

            # запоминаем состояние редактора
            self.editor_tracks = traj_TA.copy()
            self.editor_pins = pins
            self.editor_duration_s = float(duration_s)
            self.global_baseline = baseline_out.copy()

        PatternEditor(
            self,
            baseline_vec=baseline,
            on_save_callback=on_save,
            initial_tracks=self.editor_tracks,
            initial_pins=self.editor_pins,
            initial_duration_s=self.editor_duration_s,
        )

    # -------------------------- payload builders ------------------------------
    def _collect_axes(self):
        """Return a list of floats in [-1, 1] (axis targets)."""
        vals = [float(v.get()) for v in self.slider_vars]
        vals = [max(SLIDER_MIN, min(SLIDER_MAX, x)) for x in vals]
        return vals

    def _build_packet_bytes(self) -> bytes:
        """Serialize either JSON or struct: NUM_AXES poses."""
        poses = self._collect_axes()  # list of floats

        if PACKET_FORMAT.lower() == "json":
            payload = {
                "target_joint_pose": poses,  # length NUM_AXES
            }
            return json.dumps(payload).encode("utf-8")
        elif PACKET_FORMAT.lower() == "struct":
            vec = np.asarray(poses, dtype=np.float32)
            return struct.pack(STRUCT_FMT, *vec.tolist())
        else:
            raise ValueError(f"Unsupported PACKET_FORMAT: {PACKET_FORMAT}")

    # ------------------------------ UDP loop ----------------------------------
    def _stream_packet(self):
        """Periodically send the latest command packet via UDP."""
        # если активен паттерн — обновляем слайдеры
        if self.pattern_active and self.pattern_targets is not None:
            if self.pattern_index < self.pattern_total:
                sample = self.pattern_targets[self.pattern_index]  # (NUM_AXES,)
                for i in range(NUM_AXES):
                    self.slider_vars[i].set(float(sample[i]))
                self.pattern_index += 1
                self.pb["value"] = self.pattern_index
            else:
                self.pattern_active = False
                self.pb["value"] = 0

        try:
            pkt = self._build_packet_bytes()
            print(pkt)
            sock.sendto(pkt, ADDR)
        except OSError as ex:
            print(f"[UDP] send failed: {ex}")
        except Exception as ex:
            print(f"[UDP] pack failed: {ex}")

        # для отладки можно оставить:
        # print(pkt)

        self.after(STREAM_INTERVAL_MS, self._stream_packet)


# ----------------------------------------------------------------------
def run_gui() -> None:
    AxisControlApp().mainloop()


if __name__ == "__main__":
    run_gui()

