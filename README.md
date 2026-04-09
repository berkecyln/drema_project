# DreMa Freiburg Extension

This is a fork of [DreMa (ICLR 2025)](https://dreamtomanipulate.github.io/) developed as part of a master's thesis at the **University of Freiburg**.

> For the original DreMa codebase readme see [Original README](docs/original_drema_instructions.md).

**Thesis goal:** Extend DreMa towards articulated object manipulation and deploy the full pipeline in a real robot lab environment using a Franka Emika Panda with a wrist-mounted RealSense D435.

- **Robot control:** [`robot_io`](https://github.com/acl21/robot_io)  see `/robot/` directory. All camera interfacing and robot control goes through robot_io.

## Extensions & Improvements

### Real-world data collection (`data_gather_robot.py`)
Automated data collection script for the Franka Panda, written from scratch. Supports multiple movement patterns: Bézier arch, orbit, line-scan, and random-pose trajectories. Handles camera–robot calibration, pose recording in DreMa format, and depth saving.

### Camera intrinsics fixed
The original codebase computed intrinsics from image width and height and assumed no distortion and a centered lens. Changed to read intrinsics directly from the recorded pose files, which contain the calibrated `K` matrix from the actual sensor. Principal point and focal lengths now reflect the real camera, including crop and resize corrections.

### Lens distortion correction
Added `undistort_data.py` to apply full radial+tangential distortion correction to RGB and depth before processing.

### Segmentation: SAM3 backend added
Integrated Meta's [SAM3](https://github.com/facebookresearch/sam2) (video foundation model) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)+[SAM](https://github.com/facebookresearch/segment-anything) as segmentation backends. The original paper used DEVA. Selectable via `configs/segmentation.yaml`.

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
| **Raw sensor depth TSDF** | **Active** | Bypass GS-rendered depth, feed raw RealSense depth into TSDF directly. 3x precision improvement confirmed via Chamfer evaluation |

Full notes, pipeline diagrams, implementation details, and per-object evaluation tables: [`mesh_improvement_notes.md`](docs/mesh_improvement_notes.md).

## Main Scripts

```bash
python data_gather_robot.py       # collect RGB-D + poses from Franka Panda
python run_segmentation.py        # generate object masks (SAM3 or GroundingDINO+SAM)
python create_simulation.py       # extract Gaussians, meshes, URDFs
python simulate.py                # validate reconstruction interactively
python generate_new_data.py       # generate augmented training data
python eval_mesh_quality.py       # quantitative mesh evaluation (Chamfer distance)
```

---

## Credits

- Original DreMa: [Barcellona et al., ICLR 2025](https://dreamtomanipulate.github.io/) - University of Amsterdam
- Robot control and camera interface: [`robot_io`](https://github.com/acl21/robot_io) - Robot Learning Lab, University of Freiburg
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [2DGS](https://surfsplatting.github.io/), [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [NeuS2](https://github.com/19reborn/NeuS2)
