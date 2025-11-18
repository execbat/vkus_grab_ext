# Isaac Sim / Isaac Lab Extension 

![Isaac Lab]


---

## ✅ Prerequisites

- **Isaac Sim 5.x** (installed and licensed on your system).
- **Isaac Lab** cloned under your workspace (this repo expects sibling paths).
- **Conda env** with Python 3.10/3.11, CUDA-ready GPU drivers (NVIDIA).
- Git, CMake toolchain, etc., as required by Isaac Sim/Lab.

> When running with cameras, pass `--enable_cameras`.  
> On lower VRAM GPUs, prefer **headless** training and use reduced camera resolutions.

---

## 📦 Installation

1) Create a folder for your custom packages and clone the extension:
```bash
mkdir -p vkus_grab_ext
cd vkus_grab_ext
git clone https://github.com/execbat/vkus_grab_ext.git
```

2) Install the extension in **editable** mode into your Isaac‑Lab Conda env:
```bash
conda activate isaac5
cd vkus_grab_ext
pip install -e .
```

3) (Optional but recommended) Ensure Isaac Lab knows where to find your package:
- The launchers already do this internally via `ISAACLAB_TASKS_EXTRA_PACKAGES=isaaclab_custom_ext`.
- If you create new packages, add them to this variable (comma-separated).

---

## 🧱 Repository layout (core pieces)

```
vkus_grab_ext/
  ├─ vkus_grab_ext/
  │   ├─ registration/           # gym ids & isaaclab_custom_ext entry points
  │   ├─ custom_env_*/           # env/sensor/obs configs
  │   ├─ unitree_g1_23dof/       # G1 (23-DoF) asset config(s)
  │   ├─ scripts/
  │   │   ├─ run_train_with_ext.py
  │   │   └─ run_play_with_ext.py
  │   └─ ...
  └─ setup.py / pyproject.toml    # pip metadata
```

---

## 🎛️ Environments

- **Version 0 (no sensors):**
  - `Vkus_Ext-Isaac-Velocity-Flat-G1-v0` — standard velocity-tracking task.
  - `Vkus_Ext-Isaac-Velocity-Flat-G1-Play-v0` — play-mode for quick testing.

- **Version 1:**
  - `Vkus_Ext-Isaac-Velocity-Flat-G1-v1` — training environment for Middleware policy. Robot without gripper
  - `Vkus_Ext-Isaac-Velocity-Flat-G1-Play-v1` — testing environment for Middleware policy. Robot without gripper

- **Version 2:**
  - `Vkus_Ext-Isaac-Velocity-Flat-G1-v2` — training environment for Middleware policy. Robot with 2-finger gripper
  - `Vkus_Ext-Isaac-Velocity-Flat-G1-Play-v2` — testing environment for Middleware policy. Robot with 2-finger gripper


## Launch TensorBoard
```
tensorboard --logdir=logs
```


## 🚀 How to run

> Replace paths as needed for your workspace. From `IsaacLab/` root:

### Training — **Version 0 BASIC ENV**
```bash
./isaaclab.sh -p -m\
vkus_grab_ext.scripts.run_train_with_ext \
--task Vkus_Ext-Isaac-Velocity-Flat-G1-v0 \
--num_envs 1 \
--headless
```

### Play/Testing — **Version 0**
```bash
./isaaclab.sh -p -m\
vkus_grab_ext.scripts.run_play_with_ext \
--task Vkus_Ext-Isaac-Velocity-Flat-G1-Play-v0 \
--num_envs 1 \
--checkpoint ./logs/rsl_rl/vkus_experiment/<experiment folder name (contains date-time)>/<model_name>.pt \
--rendering_mode performance
```



### Training — **Version 1 training the Middleware policy. Robot without gripper**
```bash
./isaaclab.sh -p -m\
vkus_grab_ext.scripts.run_train_with_ext \
--task Vkus_Ext-Isaac-Velocity-Flat-G1-v1 \
--num_envs 1 \
--enable_cameras \
--headless
```

### Play/Testing — **Version 1 testing the Middleware policy. Robot without gripper**
```bash
./isaaclab.sh -p -m\
vkus_grab_ext.scripts.run_play_with_ext \
--task Vkus_Ext-Isaac-Velocity-Flat-G1-Play-v1 \
--num_envs 1 \
--enable_cameras \
--checkpoint ./logs/rsl_rl/vkus_experiment/<experiment folder name (contains date-time)>/<model_name>.pt \
--rendering_mode performance
```



### Training — **Version 2 training the Middleware policy. Robot with 2-finger gripper**
```bash
./isaaclab.sh -p -m\
vkus_grab_ext.scripts.run_train_with_ext \
--task Vkus_Ext-Isaac-Velocity-Flat-G1-v1 \
--num_envs 1 \
--enable_cameras \
--headless
```

### Play/Testing — **Version 2 testing the Middleware policy. Robot with 2-finger gripper**
```bash
./isaaclab.sh -p -m\
vkus_grab_ext.scripts.run_play_with_ext \
--task Vkus_Ext-Isaac-Velocity-Flat-G1-Play-v1 \
--num_envs 1 \
--enable_cameras \
--checkpoint ./logs/rsl_rl/vkus_experiment/<experiment folder name (contains date-time)>/<model_name>.pt \
--rendering_mode performance
```



```
https://www.linkedin.com/in/evgenii-dushkin/
```


