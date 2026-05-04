# PerAct Deployment Plan — DreMa → Real Robot

**Goal:** PerAct trained on DreMa-augmented data, deployed on real Franka via robot_io.

---

## STATUS OVERVIEW

| Step | What | Status |
|------|------|--------|
| 1 | Fix RLBench/rlbench/utils.py (remove PyRep) | ✓ DONE |
| 2 | Fix RLBench/tools/prepare_data_for_peract.py (3 bugs) | ✓ DONE |
| 3 | Write deploy_peract.py | ✓ DONE |
| 4 | Fork peract, add dynamic demo loading + drema config | ✓ DONE |
| 5 | Transfer data to lab PC (rsync in progress) | ✓ DONE |
| 6 | Run prepare_data_for_peract.py on lab PC | ✓ DONE |
| 7 | Train on lab PC | ⏳ IN PROGRESS |
| 8 | Inference on real robot | ⬜ TODO |

---

## REPOS AND PATHS

| What | Where |
|------|-------|
| drema_project | https://github.com/berkecyln/drema_project (lab: `/home/ceylanb/DreMa/drema_project`) |
| RLBench fork | https://github.com/berkecyln/RLBench.git (lab: `/home/ceylanb/DreMa/drema_project/RLBench`) |
| PerAct fork | https://github.com/berkecyln/peract.git (lab: `/home/ceylanb/DreMa/drema_project/peract`) |
| robot_io | lab only: `/home/ceylanb/robot/robot_io` |
| Training env | `drema_env` (torch 2.7.1+cu118, clip, yarr, rlbench editable from RLBench fork) |
| Inference env | `robotio` (torch 2.4.1+cu118, franky, pyrealsense2, pyk4a, clip, yarr, rlbench editable) |
| Lab PC | `ceylanb@knoppers` (RTX 3090 24GB) — SSH alias: `robotlab` |
| Calibration files | `assets/calibration/calibration_files/` |

---

## DONE — Technical details

### Step 1 — RLBench/rlbench/utils.py
- Removed `from pyrep.objects import VisionSensor`
- Added pure numpy `_pointcloud_from_depth_and_camera_params(depth_m, extrinsics, intrinsics)`
- **Critical:** inverts T_w2c → T_c2w before transforming to world frame
- Both wrist and overhead extrinsics are T_w2c throughout the pipeline (verified)

### Step 2 — RLBench/tools/prepare_data_for_peract.py
- **Fix 2a:** scope bug — `args.generated_path` → `generated_path` function parameter
- **Fix 2b:** descriptions — load `variation_descriptions.pkl`, inject into `misc['descriptions']` for first obs of each demo
- **Fix 2c:** intrinsics scaling — scale stored 640×360 intrinsics to 128×128 before writing to misc
- Committed `16da563e`, pushed to RLBench fork, pulled on lab PC ✓

### Step 3 — deploy_peract.py (inference script)
Verified decisions (all confirmed from source code):
- Depth from BOTH cameras is **float32 meters already** — do NOT divide by 1000 (RealSense: `raw * depth_scale`, Kinect: `raw / 1000`)
- PerAct action quaternion `[3:7]` is **xyzw** (scipy as_quat) — matches robot_io convention ✓
- Wrist T_w2c computed **per-step** from TCP pose: `T_world_tcp @ T_tcp_cam` → invert
  - `tcp_orn` from `robot.get_state()` is **xyzw** quaternion
  - `T_tcp_cam` = camera-to-TCP transform (T_tcp←cam), loaded from `panda_realsenseD435_T_tcp_cam.npy`
- K_wrist from live camera: `cam_manager.gripper_cam.get_intrinsics()` — no .npy file exists for wrist intrinsics
- `gripper_joint_positions`: binary `[0.04, 0.04]` open / `[0.0, 0.0]` closed (matches training data)
- `gripper_open` threshold: `>= 0.078` (matches robot_env.py line 87)
- Timestep: `(1 - step/(EPISODE_LENGTH-1)) * 2 - 1` (matches peract/helpers/utils.py)
- No `agent.eval()` — PreprocessAgent is not nn.Module; `build(training=False)` handles eval mode
- Robot + cam_manager used directly (not RobotEnv) — RobotEnv._get_obs() returns wrong format for PerAct
- Robot init: Hydra compose from `ROBOT_IO_CONF = '/home/ceylanb/robot/robot_io/robot_io/conf'`

**Remaining uncertainties (check on first inference run):**
- `hydra.compose('robot/panda_franky_interface_policy')` subdirectory syntax — may need adjustment
- `PILImage.fromarray` on float32 depth — should work (PIL mode 'F') but untested

### Step 4 — PerAct fork changes
- `conf/config.yaml`: set tasks, demo_path, cameras, scene_bounds, episode_length, logdir, method=PERACT_BC
- `agents/peract_bc/launch_utils.py`: added `demos=-1` support + fixed `np.bool` → `bool` (numpy removed np.bool)
- Both changes pushed to https://github.com/berkecyln/peract.git

### Steps 5 & 6 — Environment fixes needed on lab PC (drema_env)
- `RLBench/rlbench/__init__.py`: guard pyrep import in try/except; move `CameraConfig`/`ObservationConfig` imports **outside** the guard so they're always available
- `RLBench/rlbench/observation_config.py`: guard `from pyrep.const import RenderMode` with try/except stub
- YARR: install MohitShridhar fork peract branch (NOT stepjam/YARR or PyPI yarr):
  ```bash
  pip uninstall yarr -y
  git clone -b peract https://github.com/MohitShridhar/YARR.git /home/ceylanb/DreMa/YARR
  cd /home/ceylanb/DreMa/YARR && pip install -e . --no-deps
  ```
- `termcolor` missing: `pip install termcolor`
- pytorch3d: build from source (`pip install "git+https://github.com/facebookresearch/pytorch3d.git"`) or install CUDA toolkit headers first if missing

### Step 7 — Training (IN PROGRESS as of 2026-05-02)
- Started in tmux session `peract-train` on lab PC
- Replay buffer filling before training begins
- Monitor: `tensorboard --logdir /home/ceylanb/DreMa/drema_project/logs/peract`
- tmux note: launch with `LD_LIBRARY_PATH="" tmux` due to franky env lib conflict

---

## TODO — Step 6: Run prepare_data_for_peract.py (lab PC, drema_env)

**Prerequisites:** data transfer complete (rsync from local PC running now)

```bash
# Confirm data arrived
ls /home/ceylanb/DreMa/drema_project/data/generated_data/
ls /home/ceylanb/DreMa/drema_project/data/rlbench_input/

# Pull latest code
cd /home/ceylanb/DreMa/drema_project/RLBench && git pull
cd /home/ceylanb/DreMa/drema_project && git pull
cd /home/ceylanb/DreMa/drema_project/peract && git pull

# Run data preparation
conda activate drema_env
cd /home/ceylanb/DreMa/drema_project

python RLBench/tools/prepare_data_for_peract.py \
  --original_path data/rlbench_input \
  --generated_path data/generated_data \
  --output_path data/peract_ready \
  --scenes gello_bottle1_rawtsdf gello_bottle2_rawtsdf gello_bottle3_rawtsdf gello_bottle4_rawtsdf \
  --cameras wrist overhead
```

**Expected output:** `data/peract_ready/three_augmentations/<scene>/all_variations/episodes/episodeXXXX/`

**Verify counts after:**
```bash
for scene in gello_bottle1_rawtsdf gello_bottle2_rawtsdf gello_bottle3_rawtsdf gello_bottle4_rawtsdf; do
  echo -n "$scene: "
  ls data/peract_ready/three_augmentations/$scene/all_variations/episodes/ | wc -l
done
```
Expected: bottle1≈20, bottle2≈13, bottle3≈19, bottle4≈19

---

## TODO — Step 7: Train (lab PC, drema_env)

```bash
conda activate drema_env
cd /home/ceylanb/DreMa/drema_project/peract
python train.py
```

That's it — all config is in `conf/config.yaml`. Key settings already configured:
- `tasks`: all 4 bottle scenes
- `demos: -1` → uses all available per task (auto-detected)
- `cameras: [wrist, overhead]`
- `scene_bounds: [0.15, -0.35, 0.10, 0.55, 0.30, 0.75]`
- `episode_length: 50`
- `camera_resolution: [128, 128]`
- `method: PERACT_BC`, `voxel_sizes: [100]`
- `training_iterations: 40000`
- `logdir: /home/ceylanb/DreMa/drema_project/logs/peract`
- `batch_size: 8` (RTX 3090 24GB — fine)

Monitor:
```bash
tensorboard --logdir /home/ceylanb/DreMa/drema_project/logs/peract
```

Checkpoints saved every 100 iterations to:
`logs/peract/bottle_pickup/PERACT_BC/seed0/weights/`

**Known risk:** hydra-core 1.3.2 in drema_env vs PerAct expecting 1.0.5 — do NOT downgrade.
Watch for config API warnings at startup; they are usually non-fatal.

---

## TODO — Step 8: Inference (lab PC, robotio env)

```bash
conda activate robotio
cd /home/ceylanb/DreMa/drema_project

python deploy_peract.py \
  --weights logs/peract/bottle_pickup/PERACT_BC/seed0/weights \
  --config  logs/peract/bottle_pickup/PERACT_BC/seed0/.hydra/config.yaml
```

---

## Data structure reference

```
data/
  rlbench_input/
    gello_bottle1_rawtsdf/all_variations/episodes/episode0/   # 1 original demo per scene
    gello_bottle2_rawtsdf/...
    gello_bottle3_rawtsdf/...
    gello_bottle4_rawtsdf/...
  generated_data/
    gello_bottle1_rawtsdf_episode0_start/   # 20 generated augmentations
    gello_bottle2_rawtsdf_episode0_start/   # 13 generated augmentations
    gello_bottle3_rawtsdf_episode0_start/   # 19 generated augmentations
    gello_bottle4_rawtsdf_episode0_start/   # 19 generated augmentations
  peract_ready/                             # created by prepare_data_for_peract.py
    three_augmentations/
      gello_bottle1_rawtsdf/all_variations/episodes/episode0000/ ... episode0019/
      ...
```

## Calibration files (assets/calibration/calibration_files/)
- `panda_realsenseD435_T_tcp_cam.npy` — T_tcp←cam (camera-to-TCP), shape (4,4)
- `kinect_overhead_intrinsics.npy` — at 640×360 resolution
- `kinect_overhead_extrinsics.npy` — T_w2c (world-to-camera)
- No wrist intrinsics .npy — get live from `cam_manager.gripper_cam.get_intrinsics()`
