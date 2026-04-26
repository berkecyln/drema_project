# Input Scene Structure

Each scene lives under `input/<scene_name>/` and must be fully prepared before running `generate_new_data.py`. This document explains every file and why it exists.

---

## Directory layout

```
input/gello_purplebottle3_rawtsdf/
├── images/                      # RGB frames from robot scanner (640×360, top-cropped)
├── images_ir_left/              # IR left frames (for FoundationStereo, optional)
├── images_ir_right/             # IR right frames
├── depth_scaled/                # Depth frames in metres as .npy
├── object_mask/                 # Per-frame segmentation masks
├── poses/                       # Per-frame camera extrinsics + intrinsics (COLMAP format)
│   └── 0001.txt … NNNN.txt     # 4×4 extrinsics + 3×3 K matrix per frame
├── output/                      # Created by create_simulation.py
│   ├── objects_ply/             # Per-object Gaussian PLY files (e.g. 1.ply)
│   ├── flat_surface/            # Table mesh + URDF
│   ├── meshes/                  # Object meshes
│   └── urdf/                    # Per-object URDF files
├── aggregated_pointcloud.ply    # Dense reference point cloud for mesh evaluation
├── distortion_coeffs.txt        # Lens distortion coefficients
├── ir_metadata.json             # Raw RealSense intrinsics (pre-crop) + stereo baseline
├── labels.txt                   # Object label → integer id mapping (e.g. dark_blue_bottle;1)
├── visual_prompts.yaml          # SAM3 segmentation prompts and bounding boxes
├── dictionary.pkl               # Trajectory (see below)
├── low_dim_obs.pkl              # RLBench format stub (see below)
├── variation_descriptions.pkl   # Task language description (see below)
└── variation_number.pkl         # Task variation index (see below)
```

The bottom four pkl files are created by `tools/prepare_scene_for_generation.py`.

---

## poses/ format

Each `NNNN.txt` file contains the camera pose and intrinsics for that frame:

```
R[0,0] R[0,1] R[0,2]  t[0]       ← world-to-camera extrinsics
R[1,0] R[1,1] R[1,2]  t[1]
R[2,0] R[2,1] R[2,2]  t[2]
0      0      0        1
                                   ← blank line
fx     0      cx                   ← intrinsics K (crop-corrected)
0      fy     cy
0      0      1
```

Intrinsics here have the **120 px top-crop already applied** (cy ≈ 121 instead of the raw sensor cy ≈ 236). These are what `prepare_scene_for_generation.py` reads for the wrist camera K matrix.

---

## dictionary.pkl

A `list` of N step dicts representing the full robot trajectory. One dict per control step.

| Key | Shape | Content |
|---|---|---|
| `gripper_pose` | `(7,)` | End-effector pose in robot base frame: `[x, y, z, qx, qy, qz, qw]` |
| `joint_positions` | `(7,)` | Franka joint angles in radians |
| `gripper_joint_positions` | `(2,)` | Finger widths in metres: `[0.04, 0.04]` open, `[0.0, 0.0]` closed |
| `gripper_open` | `bool` | True if gripper is open |
| `joint_velocities` | `(7,)` | Finite-difference velocities computed from recording timestamps |
| `joint_forces` | `(7,)` | All zeros — not measured by GELLO, required by PerAct format |
| `ignore_collisions` | `(1,)` bool | Always False |
| `wrist_camera_intrinsics` | `(3,3)` | K matrix from poses/ (crop-corrected) |
| `wrist_camera_extrinsics` | `(4,4)` | World-to-camera transform at initial joint config (from PyBullet FK) |
| `wrist_camera_near` | `float` | 0.1 m — depth clipping for PerAct voxel grid |
| `wrist_camera_far` | `float` | 5.0 m |
| `overhead_camera_*` | — | Same four fields for Azure Kinect — added after Monday calibration |

`generate_new_data.py` reads step 0 to initialise virtual cameras, then updates wrist extrinsics live via PyBullet FK at each step. The static values in the pkl are only the starting point.



## low_dim_obs.pkl

**Why it exists:** `prepare_data_for_peract.py` expects the RLBench episode format, which includes this file in every episode folder. In the original RLBench pipeline it holds a full `Demo` object — the complete sensor log from CoppeliaSim (joint states, camera images, rewards per step). `generate_new_data.py` copies it into each generated episode as `original_low_dim_obs.pkl`.

**What we store:** A `types.SimpleNamespace` with a single field:
- `random_seed` — numpy RNG state tuple, used by `prepare_data_for_peract.py` to seed augmentation randomness for reproducibility.

We don't have a real RLBench `Demo` (we come from GELLO, not CoppeliaSim), so this is the minimum stub that satisfies the downstream code without crashing.

## variation_descriptions.pkl

A `list` of strings — the language instructions for this task. Example:
```python
["pick up the dark blue bottle and place it to the left"]
```

PerAct is a language-conditioned policy. This string is the instruction it trains and evaluates on. Set via `task.description` in `configs/prepare_scene.yaml`.


## variation_number.pkl

The integer `0`.

**Why it exists:** RLBench tasks support multiple "variations" — e.g. "reach the red/blue/green target" = variations 0/1/2, each with its own language description and object configuration. `prepare_data_for_peract.py` uses this to index into the descriptions list and organise the output folder structure.

We always use variation `0` because we have one task per scene. This is the same for all 5 demos of the same task.

