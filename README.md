# DreMa Freiburg Extension

This is a fork of [DreMa (ICLR 2025)](https://dreamtomanipulate.github.io/) developed as part of a master's project at the **University of Freiburg** by Berke Ceylan.

> For the original DreMa codebase readme see [Original README](docs/original_drema_instructions.md).

The goal of this project was to take DreMa to a complete real-robot pipeline: collecting data with a Franka Panda, building the Gaussian splatting world model, generating augmented demonstrations, and training and deploying a PerAct policy back on the real robot.

![DreMa real-robot pipeline showcase](assets/media/showcase.gif)

## Extensions & Improvements
These are the main extensions and improvements implemented to deploy and evaluate DreMa on a real Franka Panda robot in Freiburg:

### Real-world data collection (`robot_scanner/`)
End-to-end automated data collection for the Franka Panda. Supports multiple movement patterns: Bézier arch, orbit, line-scan, random-pose and GELLO replay trajectories. Handles camera–robot calibration, pose recording in DreMa format, and depth saving.

### Point cloud aggregation
To make sure the gathered data is accurate and complete, run `robot_scanner/tools/pointcloud_aggregation.py` to aggregate the RGB-D frames into a single world-frame point cloud. The same cloud is also used as the reference for mesh evaluation below.

### Segmentation: SAM3 backend added
Integrated Meta's [SAM3](https://github.com/facebookresearch/sam3) (video foundation model) as segmentation backend. The original paper used DEVA. Configurable via `configs/segmentation.yaml`.

Three segmentation methods are available (`method` field in config):
- `grounded_sam`: GroundingDINO + SAM, text-prompted, per-frame
- `sam3`: SAM3 image model, text-prompted, per-frame
- `sam3_video`: SAM3 video tracker, **visual-prompted**, propagates across all frames

To use `sam3_video`, first run `tools/pick_visual_prompts.py` to draw bounding boxes on the middle frame (or a selected frame via `prompt_frame`) of the video and save them as visual prompts. Then run `run_segmentation.py` to propagate these prompts across the video and generate masks.

![Visual prompt picking and propagation](assets/media/visual_prompt.gif)

### Mesh quality improvement
The original paper extracts meshes from GS-rendered depth, which inherits the noise and incompleteness of the rendering — not ideal for PyBullet interaction. A second problem is that the wrist-mounted camera never observes object bottoms, so meshes come out open-bottomed with poor shapes for simulation.

We replaced the GS-rendered depth with raw RealSense depth fed directly into TSDF fusion. Mesh quality is measured with bidirectional Chamfer distance against the aggregated point cloud (`tools/eval_mesh_quality.py`); on our scenes this brought mean recall error from ~10mm down to ~6mm.

![GS-rendered depth mesh vs raw depth mesh](assets/media/showcase_mesh.gif)

### Dual-camera setup
The simulation renders both the wrist camera (RealSense D415) and an overhead camera (Azure Kinect) with matching real-world calibration, including per-step wrist extrinsics. This fixes the wrist-camera rendering mismatch of the original release and gives PerAct two consistent views in training and deployment.

### PerAct training & deployment
The generated episodes are converted to PerAct's input format through our [RLBench fork](https://github.com/berkecyln/RLBench) (no CoppeliaSim needed), and a [PerAct fork](https://github.com/berkecyln/peract) trains directly on them. The trained policy runs both inside the DreMa simulation (`simulate_peract.py`) and on the real robot (`deploy_peract.py`).

## Main Scripts

```bash
# data collection
python robot_scanner/run.py                        # automatic collection of RGB-D + poses
python run_segmentation.py                         # generate object masks (SAM3 video)

# simulation setup
python create_simulation.py                        # extract Gaussians, meshes, URDFs
python simulate.py

# data generation
python tools/prepare_scene_for_generation.py       # convert recording + add camera fields + create aux pkl files
python generate_new_data.py                        # generate augmented episodes (wrist + overhead)

# train peract via peract's train.py on the generated data

# deploy the trained PerAct policy in the generated simulation
python simulate_peract.py
# deploy the trained PerAct policy on the real robot
python deploy_peract.py
```

## Credits

- Original DreMa: [Barcellona et al., ICLR 2025](https://dreamtomanipulate.github.io/) - University of Amsterdam
- Robot control and camera interface: [`robot_io`](https://github.com/acl21/robot_io) - Robot Learning Lab, University of Freiburg
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Trajectory recording:** [GELLO](https://wuphilipp.github.io/gello_software/) teleoperation device used to record manipulation demos
