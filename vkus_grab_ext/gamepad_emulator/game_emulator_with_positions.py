import json
import socket
import struct
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import numpy as np

# -----------------------------------------------------------------------------
# Networking (UDP → Isaac Lab / Isaac Sim)
# -----------------------------------------------------------------------------
UDP_IP   = "127.0.0.1"
UDP_PORT = 55001
PACKET_FORMAT = "json"      # "json" или "struct"
STRUCT_FMT    = "<7f"       # для 7 float32 (joint_1..joint_6 + gripper)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ADDR = (UDP_IP, UDP_PORT)

# -----------------------------------------------------------------------------
# Оси и GUI-константы
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
NUM_AXES = len(JOINT_NAMES)

STREAM_INTERVAL_MS = 10              # период отправки команд (мс)
SLIDER_MIN, SLIDER_MAX = -1.0, 1.0   # нормализованные команды

# нейтральная поза по умолчанию
INIT_BASELINE = np.zeros(NUM_AXES, dtype=np.float32)


# -----------------------------------------------------------------------------
# Вспомогательные функции для паттернов
# -----------------------------------------------------------------------------
def smoothstep(t: np.ndarray) -> np.ndarray:
    """S-кривая 3t^2 - 2t^3 для плавной интерполяции."""
    return t * t * (3.0 - 2.0 * t)


def resample_track(y_old: np.ndarray, new_T: int) -> np.ndarray:
    """Линейный ресэмпл трека в новый размер new_T."""
    T_old = len(y_old)
    if T_old == new_T:
        return y_old.copy()
    x_old = np.linspace(0.0, 1.0, T_old, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, new_T, dtype=np.float32)
    return np.interp(x_new, x_old, y_old).astype(np.float32)


# -----------------------------------------------------------------------------
# PatternEditor: временные шкалы + ключевые точки
# -----------------------------------------------------------------------------
class PatternEditor(tk.Toplevel):
    """
    Редактор паттернов для 7 осей.

    - Есть общее время T (кол-во шагов).
    - Есть набор ключевых кадров:
        key_times  : [K]   — индексы по времени (общие для всех осей)
        key_values : [K,7] — поза по всем 7 осям в эти моменты.
    - Между ключами по каждой оси строится плавная траектория (smoothstep).
    - Кнопка "Save pose" сохраняет текущую позу робота (из главного окна)
      в ключ по текущему времени (current_t).
    - Перетаскивание ключа по оси двигает его по времени для всех осей сразу.
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
        initial_tracks=None,
        initial_key_times=None,
        initial_key_values=None,
        initial_duration_s=None,
    ):
        super().__init__(parent)
        self.title("Pattern Editor")
        self.transient(parent)   # визуально "дочернее" окно
        # не делаем grab_set(), чтобы можно было крутить слайдеры в главном окне
        # self.grab_set()

        self.parent = parent
        self.on_save_callback = on_save_callback

        # baseline — нейтральная поза, нужен как визуальная линия нуля
        self.baseline = np.asarray(baseline_vec, dtype=np.float32)

        # ----------------- инициализация состояний -----------------
        if (
            initial_tracks is not None
            and initial_key_times is not None
            and initial_key_values is not None
        ):
            # восстановление предыдущего паттерна
            self.tracks = initial_tracks.astype(np.float32).copy()
            self.T = self.tracks.shape[0]
            self.key_times = list(int(i) for i in initial_key_times)
            self.key_values = np.asarray(initial_key_values, dtype=np.float32).copy()
            self.duration_s = tk.DoubleVar(
                value=float(
                    initial_duration_s
                    if initial_duration_s is not None
                    else self.T * STREAM_INTERVAL_MS / 1000.0
                )
            )
        else:
            # создаём новый паттерн: по умолчанию вся траектория = baseline
            self.duration_s = tk.DoubleVar(value=5.0)
            self.T = max(3, int(round(self.duration_s.get() * 1000.0 / STREAM_INTERVAL_MS)))
            # два ключа: в начале и в конце — нейтральная поза
            self.key_times = [0, self.T - 1]
            self.key_values = np.stack([self.baseline, self.baseline], axis=0)
            self._rebuild_tracks_from_keys()

        # текущая позиция "курсора" по времени
        self.current_t = self.key_times[0] if self.key_times else 0
        self.dragging_key_idx = None  # какой ключ тащим, если тащим

        self._build_ui()
        self._redraw_all()

    # ------------------------ UI PatternEditor ------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Duration (s):").pack(side="left")
        ttk.Entry(top, textvariable=self.duration_s, width=6).pack(
            side="left", padx=(4, 12)
        )
        ttk.Button(top, text="Apply", command=self._apply_duration).pack(side="left")

        ttk.Button(top, text="Reset All", command=self._reset_all).pack(
            side="left", padx=(10, 0)
        )

        ttk.Button(top, text="Save pose", command=self._save_pose_from_robot).pack(
            side="left", padx=(10, 0)
        )

        ttk.Button(top, text="Save & Play", command=self._save_and_play).pack(
            side="right"
        )

        # область с канвасами
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.scroll = tk.Canvas(
            outer,
            highlightthickness=0,
            height=NUM_AXES * (self.ROW_H + self.PAD_Y) // 2,
        )
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

        self.canvases = []
        for a in range(NUM_AXES):
            row = ttk.Frame(self.rows_holder)
            row.pack(fill="x", pady=(0, self.PAD_Y))

            ttk.Label(row, text=JOINT_NAMES[a], width=20).pack(
                side="left", padx=(0, 8)
            )
            c = tk.Canvas(
                row,
                width=self.CANVAS_W,
                height=self.ROW_H,
                bg="#0e0e10",
                highlightthickness=2,
            )
            c.pack(side="left", fill="x", expand=True)
            c.configure(highlightbackground="#2a2a2d", highlightcolor="#4ea1ff")
            # события мыши
            c.bind("<Button-1>",      lambda e, ax=a: self._on_mouse_down(ax, e))
            c.bind("<B1-Motion>",     lambda e, ax=a: self._on_mouse_drag(ax, e))
            c.bind("<ButtonRelease-1>", lambda e, ax=a: self._on_mouse_up(ax, e))
            self.canvases.append(c)

    # ---------------------- изменение длительности ----------------------
    def _apply_duration(self):
        new_d = max(0.1, float(self.duration_s.get()))
        self.duration_s.set(new_d)
        new_T = max(3, int(round(new_d * 1000.0 / STREAM_INTERVAL_MS)))
        if new_T == self.T:
            return

        old_T = self.T
        # ресэмплим текущие треки
        self.tracks = np.stack(
            [resample_track(self.tracks[:, a], new_T) for a in range(NUM_AXES)],
            axis=1,
        )
        # переносим ключевые моменты на новую шкалу
        new_key_times = []
        for t in self.key_times:
            new_t = int(round(t / (old_T - 1) * (new_T - 1)))
            new_key_times.append(new_t)
        self.key_times = sorted(set(new_key_times))

        # если вдруг ключи "слиплись", переупорядочим значения
        if len(self.key_times) != self.key_values.shape[0]:
            idx_unique = np.argsort(new_key_times)
            self.key_values = self.key_values[idx_unique]

        self.T = new_T
        self.current_t = min(self.current_t, self.T - 1)
        self._rebuild_tracks_from_keys()
        self._redraw_all()

    # ---------------------- координатные преобразования ----------------------
    def _x_to_i(self, x):
        """Экранная X → индекс по времени [0..T-1]."""
        x = np.clip(x, 0, self.CANVAS_W - 1)
        i = int(round(x / (self.CANVAS_W - 1) * (self.T - 1)))
        return int(np.clip(i, 0, self.T - 1))

    def _val_to_y(self, v):
        """Значение [-1..1] → экранная координата Y."""
        v = float(np.clip(v, -1.0, 1.0))
        y = (1.0 - v) * 0.5 * (self.ROW_H - 1)
        return y

    # ---------------------- построение трека по ключам ----------------------
    def _rebuild_tracks_from_keys(self):
        """Пересчёт трека [T,7] по key_times и key_values."""
        if len(self.key_times) == 0:
            self.tracks = np.tile(self.baseline[None, :], (self.T, 1))
            return

        # убираем дубли key_times, синхронизируем key_values
        self.key_times, idxs = np.unique(
            np.array(self.key_times, dtype=np.int32), return_index=True
        )
        self.key_times = self.key_times.tolist()
        self.key_values = self.key_values[idxs]

        # clamp и сортировка по времени
        self.key_times = [int(np.clip(t, 0, self.T - 1)) for t in self.key_times]
        order = np.argsort(self.key_times)
        self.key_times = [self.key_times[i] for i in order]
        self.key_values = self.key_values[order]

        # строим трек
        self.tracks = np.zeros((self.T, NUM_AXES), dtype=np.float32)
        for k in range(len(self.key_times) - 1):
            t0, t1 = self.key_times[k], self.key_times[k + 1]
            v0, v1 = self.key_values[k], self.key_values[k + 1]
            n = t1 - t0
            if n <= 0:
                self.tracks[t0, :] = v0
                continue
            xs = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
            s = smoothstep(xs)
            seg = v0[None, :] + (v1 - v0)[None, :] * s[:, None]
            self.tracks[t0 : t1 + 1, :] = seg

        # после последнего ключа значение просто держим
        t_last = self.key_times[-1]
        self.tracks[t_last:, :] = self.key_values[-1][None, :]

    # -------------------------- отрисовка одной оси --------------------------
    def _draw_axis(self, a):
        c = self.canvases[a]
        c.delete("all")

        # линия нуля
        y0 = self._val_to_y(0.0)
        c.create_line(0, y0, self.CANVAS_W, y0, fill="#2a2a2d")

        # линия baseline
        yb = self._val_to_y(self.baseline[a])
        c.create_line(0, yb, self.CANVAS_W, yb, fill="#2d6cdf", dash=(4, 3))

        # сама траектория
        if self.T > 1:
            pts = []
            stride = max(1, self.T // 1000)
            for i in range(0, self.T, stride):
                x = i / (self.T - 1) * (self.CANVAS_W - 1)
                y = self._val_to_y(self.tracks[i, a])
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(
                    *pts, fill="#e5e5e5", width=2, smooth=True, splinesteps=12
                )

        # ключевые точки
        for t in self.key_times:
            x = t / (self.T - 1) * (self.CANVAS_W - 1)
            y = self._val_to_y(self.tracks[t, a])
            r = 4
            fill = "#ffffff"
            if t == self.current_t:
                fill = "#ffcc00"  # активный ключ
            c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="")

        # вертикальный курсор current_t
        x_cur = self.current_t / (self.T - 1) * (self.CANVAS_W - 1)
        c.create_line(x_cur, 0, x_cur, self.ROW_H, fill="#ff8800", dash=(3, 2))

    def _redraw_all(self):
        for a in range(NUM_AXES):
            self._draw_axis(a)

    # --------------------- поиск ближайшего ключа ---------------------
    def _find_nearest_key_idx(self, i):
        if len(self.key_times) == 0:
            return None, None
        arr = np.array(self.key_times, dtype=np.int32)
        dists = np.abs(arr - i)
        idx = int(dists.argmin())
        return idx, int(dists[idx])

    # --------------------- обработка мыши ---------------------
    def _on_mouse_down(self, axis, event):
        i = self._x_to_i(event.x)
        idx, dist = self._find_nearest_key_idx(i)
        if idx is not None and dist <= 1:
            # попали в существующий ключ — начинаем тащить
            self.dragging_key_idx = idx
            self.current_t = self.key_times[idx]
        else:
            # просто двигаем курсор времени
            self.dragging_key_idx = None
            self.current_t = i
        self._redraw_all()

    def _on_mouse_drag(self, axis, event):
        # если тащим ключ — двигать по времени, с учётом соседей
        if self.dragging_key_idx is None:
            return
        new_i = self._x_to_i(event.x)
        idx = self.dragging_key_idx

        min_i = 0
        max_i = self.T - 1
        if idx > 0:
            min_i = self.key_times[idx - 1] + 1
        if idx < len(self.key_times) - 1:
            max_i = self.key_times[idx + 1] - 1

        new_i = int(np.clip(new_i, min_i, max_i))
        if new_i != self.key_times[idx]:
            self.key_times[idx] = new_i
            self.current_t = new_i
            self._rebuild_tracks_from_keys()
            self._redraw_all()

    def _on_mouse_up(self, axis, event):
        self.dragging_key_idx = None

    # --------------------- сброс и сохранение поз ---------------------
    def _reset_all(self):
        """Сбросить паттерн: один ключ в начале и один в конце, оба = baseline."""
        self.key_times = [0, self.T - 1]
        self.key_values = np.stack([self.baseline, self.baseline], axis=0)
        self.current_t = 0
        self._rebuild_tracks_from_keys()
        self._redraw_all()

    def _save_pose_from_robot(self):
        """
        Сохранить *текущую* позу робота из главного окна как ключ
        в момент времени current_t.
        """
        pose = np.array(
            [self.parent.slider_vars[i].get() for i in range(NUM_AXES)],
            dtype=np.float32,
        )
        t = int(self.current_t)
        if t in self.key_times:
            idx = self.key_times.index(t)
            self.key_values[idx, :] = pose
        else:
            insert_idx = np.searchsorted(self.key_times, t)
            self.key_times.insert(insert_idx, t)
            self.key_values = np.insert(self.key_values, insert_idx, pose, axis=0)
        self._rebuild_tracks_from_keys()
        self._redraw_all()

    def _save_and_play(self):
        """
        Финализировать паттерн:
        - вернуть трек [T,7] и ключи обратно в AxisControlApp
        - сразу запустить воспроизведение.
        """
        traj = self.tracks.copy()
        duration_s = float(self.duration_s.get())
        self.on_save_callback(
            traj,
            duration_s,
            list(self.key_times),
            self.key_values.copy(),
            self.baseline.copy(),
        )
        self.destroy()


# -----------------------------------------------------------------------------
# Главное приложение слайдеров + проигрывание паттернов
# -----------------------------------------------------------------------------
class AxisControlApp(tk.Tk):
    """7 слайдеров по осям + редактор паттернов с ключевыми точками."""

    def __init__(self):
        super().__init__()
        self.title("Axis Control (7 DOF + Patterns)")
        self.geometry("900x700")
        self.tk.call("tk", "scaling", 2.0)

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=16)
        self.option_add("*Font", default_font)

        self.global_baseline = INIT_BASELINE.copy()

        # live-переменные
        self._make_vars()

        # состояние проигрывания паттерна
        self.pattern_active = False
        self.pattern_targets = None  # [T,7]
        self.pattern_index = 0
        self.pattern_total = 0

        # состояние редактора для последующего открытия
        self.editor_tracks = None
        self.editor_key_times = None
        self.editor_key_values = None
        self.editor_duration_s = None

        self._build_ui()
        self.after(STREAM_INTERVAL_MS, self._stream_packet)

    # --------------------- модельные переменные ---------------------
    def _make_vars(self):
        self.slider_vars = [
            tk.DoubleVar(value=float(self.global_baseline[i])) for i in range(NUM_AXES)
        ]

    # --------------------- UI главного окна ---------------------
    def _build_ui(self):
        header_font = ("Helvetica", 20, "bold")

        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=3, sticky="we", padx=10, pady=(8, 4))

        ttk.Label(top, text="Set target per axis ([-1, 1])", font=header_font).pack(
            side="left"
        )
        ttk.Button(top, text="Pattern editor", command=self._open_pattern_editor).pack(
            side="left", padx=(12, 0)
        )

        # прогресс-полоса при проигрывании паттерна
        self.pb = ttk.Progressbar(
            self, orient="horizontal", mode="determinate", length=500
        )
        self.pb.grid(row=1, column=0, columnspan=3, pady=(0, 8), padx=10, sticky="we")

        # слайдеры по осям
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

        self.columnconfigure(1, weight=1)

        footer_row = NUM_AXES + 3
        ttk.Label(
            self,
            text=f"Sending {PACKET_FORMAT.upper()} every {STREAM_INTERVAL_MS} ms to {UDP_IP}:{UDP_PORT}",
        ).grid(row=footer_row, column=0, columnspan=3, pady=(20, 10), sticky="w")

    # --------------------- открытие редактора паттернов ---------------------
    def _open_pattern_editor(self):
        # baseline для редактора = текущая поза слайдеров
        baseline = np.array(
            [self.slider_vars[i].get() for i in range(NUM_AXES)], dtype=np.float32
        )

        def on_save(traj_TA, duration_s, key_times, key_values, baseline_out):
            """Колбэк из редактора после Save & Play."""
            self.pattern_targets = traj_TA.astype(np.float32)
            self.pattern_total = int(traj_TA.shape[0])
            self.pattern_index = 0
            self.pattern_active = True
            self.pb.configure(maximum=self.pattern_total, value=0)

            # сохраняем состояние редактора, чтобы потом открыть и доработать
            self.editor_tracks = traj_TA.copy()
            self.editor_key_times = list(key_times)
            self.editor_key_values = key_values.copy()
            self.editor_duration_s = float(duration_s)
            self.global_baseline = baseline_out.copy()

        PatternEditor(
            self,
            baseline_vec=baseline,
            on_save_callback=on_save,
            initial_tracks=self.editor_tracks,
            initial_key_times=self.editor_key_times,
            initial_key_values=self.editor_key_values,
            initial_duration_s=self.editor_duration_s,
        )

    # --------------------- сбор данных и отправка UDP ---------------------
    def _collect_axes(self):
        vals = [float(v.get()) for v in self.slider_vars]
        vals = [max(SLIDER_MIN, min(SLIDER_MAX, x)) for x in vals]
        return vals

    def _build_packet_bytes(self) -> bytes:
        poses = self._collect_axes()
        if PACKET_FORMAT.lower() == "json":
            payload = {
                "target_joint_pose": poses,
            }
            return json.dumps(payload).encode("utf-8")
        elif PACKET_FORMAT.lower() == "struct":
            vec = np.asarray(poses, dtype=np.float32)
            return struct.pack(STRUCT_FMT, *vec.tolist())
        else:
            raise ValueError(f"Unsupported PACKET_FORMAT: {PACKET_FORMAT}")

    # --------------------- главный цикл отправки ---------------------
    def _stream_packet(self):
        # если активен паттерн — продвигаем проигрывание
        if self.pattern_active and self.pattern_targets is not None:
            if self.pattern_index < self.pattern_total:
                sample = self.pattern_targets[self.pattern_index]
                for i in range(NUM_AXES):
                    self.slider_vars[i].set(float(sample[i]))
                self.pattern_index += 1
                self.pb["value"] = self.pattern_index
            else:
                self.pattern_active = False
                self.pb["value"] = 0

        # отправляем текущие значения осей
        try:
            pkt = self._build_packet_bytes()
            sock.sendto(pkt, ADDR)
        except OSError as ex:
            print(f"[UDP] send failed: {ex}")
        except Exception as ex:
            print(f"[UDP] pack failed: {ex}")

        self.after(STREAM_INTERVAL_MS, self._stream_packet)


# ----------------------------------------------------------------------
def run_gui() -> None:
    AxisControlApp().mainloop()


if __name__ == "__main__":
    run_gui()

