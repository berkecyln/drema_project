# DreMa Freiburg Extension

This is a fork of [DreMa (ICLR 2025)](https://dreamtomanipulate.github.io/) developed as part of a master's thesis at the **University of Freiburg**.

> For the original DreMa codebase readme see [Original README](docs/original_drema_instructions.md).

**Thesis goal:** Extend DreMa and deploy the full pipeline in a real robot lab environment using a Franka Emika Panda with a wrist-mounted RealSense D415 and overhead Azure Kinect DK.

- **Robot control:** [`robot_io`](https://github.com/acl21/robot_io)  see `/robot/` directory. All camera interfacing and robot control goes through robot_io.
- **Trajectory recording:** [GELLO](https://wuphilipp.github.io/gello_software/) teleoperation device used to record manipulation demonstrations. Joint positions saved as `.npy` under `assets/trajectories/npy/`.
- **Cameras:** Wrist RealSense D415 (egocentric) + fixed overhead Azure Kinect DK. Both cameras are rendered per-step in simulation and included in generated episodes for dual-camera training.

## Extensions & Improvements

### Real-world data collection (`robot_scanner/`)
End to end automated data collection for the Franka Panda. Supports multiple movement patterns: Bézier arch, orbit, line-scan, random-pose and GELLO replay trajectories. Handles camera–robot calibration, pose recording in DreMa format, and depth saving.

- **`GelloMover`** (`robot_scanner/movements/gello_replay.py`): replays a GELLO-recorded `.npy` trajectory for environment scanning. Subsamples to `n_frames` waypoints evenly spaced by arc length (uniform spatial coverage independent of recording speed).

### Camera intrinsics fixed
The original codebase computed intrinsics from image width and height and assumed no distortion and a centered lens. Changed to read intrinsics directly from the recorded pose files, which contain the calibrated `K` matrix from the actual sensor. Principal point and focal lengths now reflect the real camera, including crop and resize corrections.

A second instance of the same bug existed in the data generation renderer (`generate_new_data.py` → `drema/environment/observer/camera.py`): image width and height were inferred as `cx*2` and `cy*2`, which only holds if the lens is perfectly centered. For a flipped D415 with `cy≈122` in a 640×360 image, this produced wrong 654×244 renders. Fixed by storing the actual image dimensions in `dictionary.pkl` (read from the recorded `images/` directory in `prepare_scene_for_generation.py`) and using them in the renderer.

### Camera intrinsics in-place mutation bug fixed
`load_cameras_from_trajectory` in `drema/environment/observer/camera.py` scaled intrinsics with `intrinsics[:2,:] *= scale`, which mutated the numpy array in-place. Because the array was a direct reference into `trajectory.demo`, this corrupted the stored intrinsics (4× too large) in the trajectory object. Episode 0 was saved right after this call and therefore stored wrong intrinsics; augmented episodes were unaffected because `env.reset()` reloads the trajectory from disk before saving. Fixed by adding `.copy()` before scaling so the trajectory's stored values are never modified.

### Wrist camera per-step extrinsics
`generate_new_data.py` now writes the FK-derived `T_w2c` for the wrist camera into each step of the saved `generated_trajectory.pkl`. Previously all steps stored the static initial-pose extrinsics. The overhead camera is physically fixed so its stored extrinsics are correct throughout.

### Segmentation: SAM3 backend added
Integrated Meta's [SAM3](https://github.com/facebookresearch/sam2) (video foundation model) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)+[SAM](https://github.com/facebookresearch/segment-anything) as segmentation backends. The original paper used DEVA. Selectable via `configs/segmentation.yaml`.

Three segmentation methods are available (`method` field in config):
- `grounded_sam`: GroundingDINO + SAM, text-prompted, per-frame
- `sam3`: SAM3 image model, text-prompted, per-frame
- `sam3_video`: SAM3 video tracker, **visual-prompted**, propagates across all frames

The `sam3_video` Workflow:
1. Draw bounding boxes interactively on the middle frame (change this frame if its not good): `python tools/pick_visual_prompts.py`
2. Optionally add `sam_text` to any entry in `visual_prompts.yaml` to use a more descriptive tracking text while keeping the DreMa-compatible `label` (e.g. `label: table`, `sam_text: white table surface`)
3. Run segmentation: `python run_segmentation.py`

### Point cloud aggregation
Built a pipeline to aggregate all depth frames from a capture session into a single dense reference point cloud. Used as ground-truth geometry for quantitative mesh evaluation and debugging.

## Mesh Quality Research

A significant part of this work investigates improving mesh extraction quality. The core problem: the wrist-mounted camera never observes object bottoms, resulting in open-bottom meshes with poor quality shapes for simulation.

Six approaches were tried and evaluated quantitatively using bidirectional Chamfer distance against an aggregated reference point cloud:

| Approach | Status | Finding |
|---|---|---|
| `fill_holes()` (Open3D) | Abandoned | No effect , missing bottom is absent geometry, not a bounded hole |
| Poisson reconstruction | Abandoned | One-sided normals extrapolate a skirt, not a bottom cap |
| [NeuS2](https://github.com/19reborn/NeuS2) neural SDF | Abandoned | Top-only cameras collapse the SDF to a flat pancake |
| [SAM3D](https://github.com/facebookresearch/sam3) | On hold | Requires 32GB VRAM minimum |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | On hold | Camera always at 45°+ elevation; TripoSR interprets this as a tent shape. Needs low-elevation frames |
| **Raw sensor depth TSDF** | Active | Bypass GS-rendered depth, feed raw RealSense depth into TSDF directly. 3x precision improvement confirmed via Chamfer evaluation |
| **Geometric model fitting** | **Active** | Fit a watertight primitive (box/cube) to the filtered Gaussian point cloud. Solves the open-bottom problem entirely. Shape type set per-object via `shape` field in `visual_prompts.yaml`. |

Full notes, pipeline diagrams, implementation details, and per-object evaluation tables: [`mesh_improvement_notes.md`](docs/mesh_improvement_notes.md).

### Robot Gaussian alignment (`tools/align_robot_gaussians.py`)
Repositions the original robot surface Gaussians to the current gello joint configuration using pybullet FK-derived per-link SE3 transforms so simulation matches reality.

## Main Scripts

```bash
# data collection
python robot_scanner/run.py                        # collect RGB-D + poses from Franka Panda
python run_segmentation.py                         # generate object masks (SAM3 video/image or GroundingDINO+SAM)

# simulation setup
python create_simulation.py                        # extract Gaussians, meshes, URDFs
python tools/align_robot_gaussians.py              # reposition robot Gaussians to current joint config
python tools/compute_object_center.py <scene> --object-id <id>  # get rotation_center for augmentation config

# data generation
# Edit configs/prepare_scene.yaml (scene path, trajectory, description, overhead camera)
python tools/prepare_scene_for_generation.py       # convert recording + add camera fields + create aux pkl files
python simulate.py                                 # validate reconstruction + replay trajectory interactively
python generate_new_data.py                        # generate approx. 200 augmented episodes (wrist + overhead)

# PerAct training data preparation
conda activate peract_env
python RLBench/tools/prepare_data_for_peract.py --cameras wrist overhead
```

---
## tmux
start tmux terminal:
LD_LIBRARY_PATH="" sudo renice -n 0 $$ && tmux new -s peract-train

export DISPLAY=:0
export PYOPENGL_PLATFORM=egl
rm -rf /tmp/arm/replay
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py 2>&1 | tee train_run.log


to detach from tmux: `Ctrl+b` then `d`

reattach to tmux:
LD_LIBRARY_PATH="" tmux attach -t peract-train

to kill tmux session: `Ctrl+b` then `x` (confirm with `y`)
or 
LD_LIBRARY_PATH="" tmux kill-session -t peract

## Credits

- Original DreMa: [Barcellona et al., ICLR 2025](https://dreamtomanipulate.github.io/) - University of Amsterdam
- Robot control and camera interface: [`robot_io`](https://github.com/acl21/robot_io) - Robot Learning Lab, University of Freiburg
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [2DGS](https://surfsplatting.github.io/), [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [NeuS2](https://github.com/19reborn/NeuS2)
