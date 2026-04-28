
## PerAct

[PerAct](https://peract.github.io/) (Perceiver-Actor) is a language-conditioned manipulation policy that voxelizes multi-camera RGB-D observations into a 3D occupancy grid and runs a Perceiver transformer over it to predict a discrete action.

**Action representation:** absolute 6-DoF end-effector pose + gripper open/close, all in world frame. No relative actions.

**Hardware requirements:** the default configuration (100³ voxel grid) was trained on 8×V100 GPUs and requires approximately 20-40 GB VRAM at batch size 1. Running on a single consumer GPU (12 GB) requires reducing the voxel resolution to 64³ or below, which risks accuracy loss.

**Deployment pipeline:**
1. Capture RGB-D frames from wrist and overhead cameras
2. Back-project to point cloud using calibrated intrinsics and extrinsics
3. Voxelize into the workspace bounds (x=[0.15,0.65], y=[-0.45,0.45], z=[0.0,0.55])
4. Run PerAct inference → outputs (x, y, z, rotation bin, gripper state)
5. Send absolute TCP target to robot via robot_io

The voxelization at inference must exactly replicate the training-time voxelization. This requires a small inference wrapper that is not included in the original codebase and must be implemented separately.

## RVT (Robotic View Transformer)

[RVT](https://robotic-view-transformer.github.io/) is an alternative policy that avoids the 3D voxel grid entirely. Instead, it re-renders the scene from a set of fixed virtual viewpoints (e.g. top, front, side) as 2D images and processes them with a 2D transformer.

**Action representation:** identical to PerAct - absolute 6-DoF pose + gripper state in world frame.

**Advantages over PerAct:**
- Fits on a 12 GB consumer GPU (no 3D voxel grid)
- Trains in hours
- Matches or outperforms PerAct on most RLBench benchmarks (RVT-2 paper)
- Reads the same RLBench Demo format - no changes to the data pipeline

**Deployment pipeline:**
1. Capture RGB-D frames from wrist and overhead cameras
2. Back-project to point cloud
3. Re-render point cloud from virtual viewpoints (Open3D, straightforward)
4. Run RVT inference → absolute action
5. Send TCP target via robot_io

The virtual re-rendering step is simpler to implement than PerAct voxelization and has fewer hyperparameters to match between training and inference.

