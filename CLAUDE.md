# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DreMa** (Dream to Manipulate) is a compositional world model framework for robot imitation learning, presented at ICLR 2025. It combines 3D/2D Gaussian Splatting for scene reconstruction with PyBullet physics simulation to generate augmented training data for robot manipulation policies (e.g., PerAct).

## Setup & Installation

**Environment creation (CUDA 12.8+ / RTX 50-series):**
```bash
conda env create -f environment.yml
conda activate drema_env
# See newer_setup.md for RTX 50-series specifics
```

**Compile CUDA submodules (must do after env setup):**
```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"  # Adjust to your GPU arch

pip install submodules/simple-knn --no-cache-dir
pip install submodules/diff-gaussian-rasterization --no-cache-dir
pip install submodules/diff-gaussian-rasterization-depth --no-cache-dir
pip install submodules/diff-surfel-rasterization --no-cache-dir
pip install -r requirements.txt
```

**SAM3 (segmentation):**
```bash
huggingface-cli login
cd sam3 && pip install -e ".[notebooks]"
```

## Main Entry Points

```bash
# 1. Extract Gaussians, meshes, and URDFs from raw RGB-D data
python create_simulation.py

# 2. Test/visualize reconstructed environment interactively
python simulate.py        # r=reset, q/esc=quit, arrow keys=camera

# 3. Generate augmented training data
python generate_new_data.py

# 4. Prepare segmentation masks (SAM3 or GroundingDINO+SAM)
python run_segmentation.py

# 5. Collect real robot data (Franka Panda)
python data_gather_robot.py
```

## Architecture

### Pipeline Flow
1. **Raw data** (RGB-D images + segmentation masks + camera poses) →
2. `create_simulation.py` → **assets** (per-object Gaussians PLY + collision URDFs) →
3. `simulate.py` → **PyBullet environment** with Gaussian-rendered views →
4. `generate_new_data.py` → **augmented RGB-D dataset** for policy training

### Key Modules

**`drema/r2s_builder/`** — Real-to-Sim pipeline
- `assets_extractor.py`: Orchestrates full extraction (trains Gaussians per object, extracts meshes, builds URDFs)
- `gaussians_optimizers/`: Four training backends — `base`, `depth`, `surf`, `surf_depth`
- `extractors/`: Mesh extraction from trained Gaussians; URDF generation

**`drema/environment/`** — Simulation environment
- `builder.py`: Constructs the full scene (loads Gaussians, spawns PyBullet bodies, sets up cameras)
- `base_environment.py` / `robot_environment.py`: Physics loop, rendering, trajectory execution
- `robot/trajectory.py`: Loads and executes waypoint trajectories

**`drema/drema_scene/`** — Interactive Gaussian manipulation
- Extends core `GaussianModel` with filter-by-mask, translate, rotate, scale operations
- Enables scene composition: separate environment from objects, recombine after transformations

**`drema/gaussian_renderer/`** — Three rendering backends
- `original_gaussian_renderer/`: Standard 3DGS
- `depth_gaussian_renderer/`: Depth-supervised variant
- `surf_gaussian_renderer/`: 2DGS surface splatting

**`drema/scene/gaussian_model.py`** — Core 3DGS model (based on original GS repo): manages Gaussian parameters, optimization, PLY I/O

### Configuration System (Hydra)

All scripts use Hydra configs from `configs/`:
- `config.yaml` / `config_real.yaml` — top-level; sets `data.source_path` and `data.assets_path`
- `configs/training/` — Gaussian training hyperparameters, what to extract
- `configs/simulation/` — environment setup, robot, camera, generation parameters
- `configs/segmentation.yaml` — segmentation model selection

Override any config on the command line: `python create_simulation.py training.optimization.gaussians_iterations=10000`

### Input Data Format
```
task_name/
├── images/          # RGB PNGs
├── depth_scaled/    # Depth as NPY arrays
├── object_masks/    # Segmentation masks (PNG)
├── poses/           # Camera extrinsics/intrinsics
└── labels.txt       # "object_name;id" per line
```

### Output Assets Format
```
assets_path/
├── objects_ply/     # Per-object Gaussian PLY files
├── urdf/            # Per-object URDF with collision shapes
└── flat_surface/    # Table/surface Gaussian representation
```

## Important Notes

- **Table must be extracted first** — it's used to filter object meshes. If pipeline errors, delete `assets_path` and rerun.
- **Gaussian backends**: depth and surf variants give better geometry; choose via `training.preparation` flags in config.
- **VRAM**: Gaussian training is memory-intensive. Reduce `mesh_res` or `gaussians_iterations` if OOM.
- **Wrist camera**: Has depth artifacts; current workaround uses 3 fixed cameras instead.
- **PerAct integration**: See `COPPELIA.md` for preparing augmented data for PerAct/RLBench training.
- **Physics**: PyBullet supports both `GUI` and `DIRECT` modes; controlled via simulation config.

## Project Context & Collaboration Notes

This repo is a fork/extension of the original Amsterdam University DreMa project. The current developer is a master's student at University of Freiburg who has read the entire codebase and written all extensions personally. Treat this accordingly: **no bulk changes, no aggressive refactors, explain every change before implementing, get explicit approval first.**

**Hardware setup:** Franka Emika Panda + RealSense wrist-mounted gripper camera. Robot controlled via `robot_io` (see `/home/gunnleif/Projects/drema/robot/`). Data gathered with `data_gather_robot.py` using Bézier arch / orbit / line-scan / random-pose movement patterns.

**Extensions already implemented by this developer (not in original repo):**
- Removed hardcoded intrinsics — reads camera intrinsics directly from recorded pose files
- Added SAM3 (Meta's video segmentation) as a segmentation backend alongside GroundingDINO+SAM
- `data_gather_robot.py` — automated data collection script for Franka Panda
- Lens distortion correction for RealSense data
- Point cloud aggregation pipeline
- Various bug fixes and real-world robustness improvements

**Active research direction:** Improving mesh quality for static objects. The current TSDF-from-Gaussians pipeline produces incomplete meshes because the wrist-mounted camera never sees the bottom of objects resting on the table — that region is completely absent (never captured by TSDF, not a bounded hole). Articulated objects are a future goal, not the current focus.

**What was tried and why it failed:**
- `open3d fill_holes()` — no effect. Missing bottom is absent geometry (TSDF never created voxels there), not a bounded hole. Vertex count identical before and after.
- **Poisson v1** (opacity>0.5, density_threshold=0.1) — better skeleton shape but many holes. Only 23% of Gaussians used. Density trim of 10% removed the extrapolated bottom — contradiction.
- **Poisson v2** (opacity>0.1, density_threshold=0.0) — unseen regions stretched/ballooned outward. Classic Poisson failure on absent data: side-wall normals extrapolate into a skirt rather than a flat cap. Structurally the wrong tool for a one-sided (always above) capture setup.
- **SAM3D on hold:** Requires 32GB VRAM minimum — does not run on any available hardware (RTX 3090 = 24GB max).

**NeuS2 — ABANDONED (2026-04-06): structurally wrong for top-only cameras**

NeuS2 was fully built and integrated (9 bugs fixed). Final result: flat pancake — no vertical sides. Root cause is structural: NeuS2 needs rays from multiple directions to constrain the SDF on all faces. With all cameras looking downward, the SDF collapses to a horizontal plane. TSDF (despite its open bottom) is dramatically better. Integration code remains in `drema/r2s_builder/extractors/neus2_extractor.py`, not in active use.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable base |
| `mesh_improve` | Fill-holes + Poisson mesh option |
| `triposr` | TripoSR single-image mesh extraction (based on `mesh_improve`) — **active** |
| `neus2` | NeuS2 integration (based on `mesh_improve`, abandoned) |

## TripoSR — INTEGRATED (2026-04-06)

Single-image transformer (LRM architecture, Objaverse-trained). Infers closed meshes including bottom face from shape priors. 6GB VRAM, ~2 sec per object. Replaces only the mesh extraction step — Gaussian splatting unchanged.

**How it works:**
- `select_best_image`: picks frame with most foreground pixels for the object_id (override with `triposr_image: "0099"` in config)
- `_prepare_image`: crops to foreground bbox, pads to square, adds border so object fills `triposr_foreground_ratio` of the frame; gray background (0.5 — TripoSR convention)
- World-space fitting: scale by max(x/y **Gaussian** bbox span), place bottom at Gaussian `box_min[2]`. Uses filtered Gaussian positions (DBSCAN-cleaned) not raw depth bbox — avoids depth-noise outliers inflating scale on reflective objects.
- `filter_mesh` is skipped for triposr (mesh is already in world space; filtering would cut the hallucinated bottom)
- VRAM freed explicitly after each object (`del model, scene_codes, meshes` + `empty_cache()`) to prevent OOM on multi-object scenes
- Set `mesh_method: "triposr"` in `configs/training/real_world_params.yaml`

**Critical camera elevation constraint (discovered 2026-04-06):**
The wrist-camera arch trajectory stays at **41–53° elevation** — there are no low-elevation frames. TripoSR trains on 0° horizontal cameras. This creates a systematic interpretation problem:
- At high elevation, the visible faces (top + two sides of a cube) look exactly like a tent/roof to TripoSR's horizontal-camera assumption → reconstructs a **triangular prism** instead of a cube.
- `triposr_foreground_ratio` 0.3–0.85 was fully tested — no setting gives a usable mesh. Low → rhombus (off-center perspective), high → tent, mid → correct shape but too rounded.
- **Fix: capture 1-2 extra frames per object at ~15–25° elevation** (arm extended sideways, near-horizontal). Keep normal arch for Gaussians. Override TripoSR with `triposr_image: "0xxx"` pointing to the low-elevation frame. Only one frame needed for TripoSR.

**Known limitations:**
- TripoSR uses a NeRF density field internally — smooth implicit function that cannot represent sharp geometric edges. Cubes will always appear rounded/blob-like regardless of resolution.
- Shape prior is Objaverse-trained: works best on complex/textured objects; simple geometric shapes are ambiguous and may hallucinate furniture.
- Wrist-camera datasets (all frames 41–53° elevation) always give top-down views. Use `triposr_foreground_ratio: 0.3–0.4` to rely on shape prior.

**Tunable config params:**
- `triposr_resolution: 384` — marching cubes grid (256 default, 512 OOMs on 11GB VRAM)
- `triposr_threshold: 25.0` — isosurface density threshold (lower=more geometry, try 15-25)
- `triposr_foreground_ratio: 0.4` — object fill fraction in crop (0.3–0.4 for wrist-cam top-down datasets; 0.85 for side-on datasets)
- `triposr_image: ""` — override image filename for testing (empty = auto select)

**Key files:**
- `drema/r2s_builder/extractors/triposr_extractor.py`
- `submodules/torchmcubes/` — CUDA marching cubes (custom setup.py, see below)
- `TripoSR/` — cloned repo, added to sys.path at runtime (not a git submodule)

**TripoSR setup (one-time):**
```bash
cd ~/Projects/drema/drema_project
git clone https://github.com/VAST-AI-Research/TripoSR.git

pip install einops omegaconf transformers diffusers accelerate huggingface_hub
pip install rembg onnxruntime
pip install "numpy<2.0"   # rembg upgrades numpy to 2.x — pin it back
# opencv: must be full build (not headless) — rembg installs headless which breaks cv2.imshow
pip install "opencv-python==4.8.0.76"

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST='12.0'
cd submodules/torchmcubes && pip install -e . && cd ../..
```

**torchmcubes build note:** upstream uses cmake + scikit-build-core which fails here (pip-packaged CUDA headers, PyTorch nightly CUDA version string mismatch). Replaced with `torch.utils.cpp_extension` in `setup.py` + patched `pyproject.toml`. Both `mcubes_cpu` and `mcubes_cuda` compile successfully; dispatch is automatic based on tensor device.

**opencv conflict:** `rembg` pulls in `opencv-python-headless` which disables `cv2.imshow` (breaks `simulate.py`). Fix: `pip install "opencv-python==4.8.0.76"` after installing rembg. Do NOT `pip install opencv-python>=4.13` — it forces numpy>=2 which breaks all compiled submodules.

**mesh_method options:**
- `"tsdf"` — default, fast, open bottom
- `"poisson"` — Gaussian positions → Poisson reconstruction
- `"triposr"` — single-image transformer, closes the bottom face

**Workflow order:**
1. `data_gather_robot.py` → collect RGB-D + poses
2. `run_segmentation.py` → generate object masks
3. `create_simulation.py` → extract Gaussians + meshes + URDFs
4. `simulate.py` → validate the reconstruction
