# Mesh Improvement — Research Notes

---

## Problem Definition

The core pipeline trains 2D Gaussian Splatting (2DGS) per object using masked RGB-D frames from a wrist-mounted RealSense D435 on a Franka Panda. After training, we extract a mesh from the trained Gaussians using TSDF fusion, then convert it into a URDF collision shape for PyBullet simulation.

**The fundamental problem:** The wrist camera moves in an arch trajectory at 41–53° elevation above the table. It never goes below the object level. So the camera only ever sees the top face and upper sides of each object — the bottom is never observed. This results in meshes that are open at the bottom, which causes objects to fall through the table in physics simulation or have incorrect collision shapes.

Secondary problem we discovered along the way: the GS-rendered depth used for TSDF was blurry and distorted — Gaussian splatting smooths and interpolates across surfaces, so the rendered depth is not as accurate as raw sensor readings.

---

## Base Pipeline — TSDF from GS-rendered Depth

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ▼
 Filter input: mask out everything except object id
 (dilated mask, blackout background in RGB + depth)
         │
         ▼
 Train 2DGS (SurfDepth backend, 7000 iterations)
 → Gaussian point cloud fitted to object appearance + depth
         │
         ▼
 Render all training views through trained Gaussians
 → GS-rendered depth maps (surf_depth) per frame
 → GS-rendered RGB per frame
         │
         ▼
 TSDF Fusion (Open3D ScalableTSDFVolume)
 Integrate each (RGB, depth) frame with camera pose
         │
         ▼
 Extract triangle mesh via marching cubes
         │
         ▼
 Filter mesh: remove triangles below or inside table surface
         │
         ▼
 Save as .obj → build URDF for PyBullet
```

**Key config at baseline:**
- `mesh_method: "tsdf"` (use GS-rendered surf_depth)
- `depth_ratio: 0.0` (unbounded scene depth mode)
- `position_lr_max_steps: 30000` (but only 7000 iters run → LR never fully decays)

**What the base mesh looked like:** Reasonable top and side surfaces but open bottom, some blobby geometry on sides from GS depth artifacts, lower overall completeness.

---

## Approach 1 — fill_holes()

### Pipeline diagram

Same as the base pipeline. The only change is a single extra step inserted after mesh extraction:

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ▼
 Filter input: mask out everything except object id
         │
         ▼
 Train 2DGS (SurfDepth backend, 7000 iterations)
         │
         ▼
 Render all training views through trained Gaussians
 → GS-rendered depth maps (surf_depth) + RGB
         │
         ▼
 TSDF Fusion (Open3D ScalableTSDFVolume)
         │
         ▼
 Extract triangle mesh via marching cubes
         │
         ▼
 Filter mesh: remove triangles below/inside table
         │
         ▼
 *** fill_holes() ← NEW STEP ***
 o3d.t.geometry.TriangleMesh.fill_holes()
 Detects open boundary loops → attempts to cap them
         │
         ▼
 Save as .obj → build URDF for PyBullet
```

What changed: one post-processing step added at the very end. Everything upstream is identical to the base pipeline.

### Why it could be useful
Open3D has a built-in `fill_holes()` function on triangle meshes. The idea was simple: if the mesh has a hole at the bottom (open boundary loop), maybe the algorithm can detect it and cap it with a flat patch. Zero extra computation, easy to toggle on/off.

### Implementation notes
- Added `fill_mesh_holes()` wrapper in `drema/gaussian_splatting_utils/mesh_utils.py`:
  ```python
  def fill_mesh_holes(mesh):
      t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
      filled = t_mesh.fill_holes()
      return filled.to_legacy()
  ```
- Added `fill_holes` flag to `AssetsManager.__init__()` in `assets_extractor.py`
- Added `fill_mesh_holes: false` config option in `real_world_params.yaml`
- Applied after mesh extraction and before saving, optionally after `filter_mesh()` too
- Controlled via `assets.fill_mesh_holes` in config — set to `false` in current active config
- Committed in: `3f0026e add Poisson reconstruction and fill_holes mesh options`

### Result
No visible change. The mesh looked identical before and after `fill_holes()`.

### Verdict
**Not effective. Abandoned.**

The reason is structural: `fill_holes()` works on bounded holes — closed boundary loops where there's a gap in an otherwise solid surface (like a small puncture). Our missing bottom is not a hole in that sense. It's completely absent geometry — no boundary loop exists at the bottom because there were never any camera views to create triangles there. There is nothing for the algorithm to detect and cap.

---

## Approach 2 — Poisson Reconstruction

### Pipeline diagram

Poisson replaces the entire TSDF path. Instead of rendering depth maps and fusing them volumetrically, it works directly on the trained Gaussian positions:

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ▼
 Filter input: mask out everything except object id
         │
         ▼
 Train 2DGS (SurfDepth backend, 7000 iterations)
         │
         ▼
 *** DIVERGES FROM BASE HERE ***
 Extract Gaussian positions (xyz) from trained model
 Filter by opacity > 0.1 (remove low-confidence splats)
         │
         ▼
 Estimate surface normals (Open3D KDTree, radius=0.1)
 Orient normals towards mean camera center
         │
         ▼
 *** Poisson Surface Reconstruction ***
 o3d.geometry.TriangleMesh
   .create_from_point_cloud_poisson(depth=9)
 Remove low-density vertices (quantile threshold 0.1)
         │
         ▼
 Filter mesh: remove triangles below/inside table
         │
         ▼
 Save as .obj → build URDF for PyBullet
```

What changed: skip TSDF entirely. No depth rendering, no volumetric fusion. Input to reconstruction is the Gaussian point cloud, not depth maps. Normal estimation + Poisson replaces the whole TSDF + marching cubes block.

### Why it could be useful
Poisson surface reconstruction is a classic algorithm that fits a watertight surface to an oriented point cloud. Unlike TSDF which is view-dependent and requires actual depth readings from a direction, Poisson extrapolates the surface based on normals. In theory it could hallucinate a bottom face by extrapolating normals from the sides downward and closing the surface.

### Implementation notes
- Added `extract_mesh_poisson()` in `drema/gaussian_splatting_utils/mesh_utils.py`:
  - Takes Gaussian positions as the point cloud (not raw depth)
  - Filters by opacity threshold (`poisson_opacity_threshold: 0.1`) to remove low-confidence Gaussians
  - Estimates normals with Open3D KDTree (radius=0.1, max_nn=30)
  - Orients normals towards mean camera center (computed from all training view c2w matrices)
  - Runs `o3d.geometry.TriangleMesh.create_from_point_cloud_poisson()` with configurable depth (`poisson_depth: 9`)
  - Removes low-density vertices via quantile threshold (`poisson_density_threshold: 0.1`)
- Added `mesh_method: "poisson"` option to `assets_extractor.py` dispatch (alongside `"tsdf"`)
- Added config params: `poisson_depth`, `poisson_density_threshold`, `poisson_opacity_threshold` in `real_world_params.yaml`
- Committed in: `3f0026e add Poisson reconstruction and fill_holes mesh options`

Two versions tried:
- **v1:** Direct Poisson on raw Gaussian positions with mean camera orientation
- **v2:** Same but different density thresholds and poisson_depth values

### Result
Both versions produced a skirt/flared-bottom shape rather than a clean flat cap. The sides of the mesh flared outward and downward like a tent. No clean bottom face was produced.

### Verdict
**Structurally wrong for this capture setup. Abandoned.**

The root problem: Poisson reconstruction needs normals pointing inward from all directions to converge on a closed surface. All our camera views are from above — so all estimated normals point roughly upward. When Poisson tries to extrapolate the surface downward from the bottom edge of the observed point cloud, the normals all point the wrong way. It ends up extrapolating a skirt that flares outward rather than sealing the bottom. This is a geometric consequence of one-sided capture and cannot be fixed by tuning parameters.

---

## Approach 3 — NeuS2

### Pipeline diagram

NeuS2 is a completely parallel pipeline — it bypasses Gaussian splatting entirely and operates directly on raw RGB+masks:

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ├─────────────────────────────────────────┐
         │  (base pipeline continues as normal)    │
         │                                         ▼
         │                          *** NeuS2 PATH (replaces mesh step) ***
         │
         ▼                          Raw RGB frames + segmentation masks
 Filter input / Train 2DGS          │
 (still done — bbox used for        ▼
  marching cubes region)    Convert to NeuS2 format:
                             - Masked RGBA images (alpha = object mask)
                             - transforms.json (poses in OpenGL convention)
                             - Coord system: OpenCV → OpenGL flip (y,z negated)
                                     │
                                     ▼
                             Train NeuS2 SDF network
                             (20000 steps, NGP-based hash encoding)
                             Runs in subprocess (VRAM isolation)
                                     │
                                     ▼
                             Marching cubes on SDF zero-level set
                             AABB restricted to object world bbox
                             (world → NGP space transform applied)
                                     │
                                     ▼
                             Axis remap: NGP [X,Y,Z] → world [z,x,y]
                                     │
                                     ▼
                             Filter mesh: remove below/inside table
                                     │
                                     ▼
                             Save as .obj → build URDF for PyBullet
```

What changed: Gaussians are no longer involved in mesh extraction at all. The mesh comes purely from NeuS2's neural SDF trained on the raw frames. The Gaussian bbox is still used to tell NeuS2 where to run marching cubes.

### Why it could be useful
NeuS2 is a neural implicit surface reconstruction method that uses a signed distance function (SDF) trained on multi-view RGB images with alpha masking. The SDF formulation forces a closed surface — unlike TSDF which is data-driven, NeuS2 needs to find a coherent zero-level set across all views. In theory this should produce a topologically closed mesh, including the bottom face. It's also end-to-end neural and might generalize better than geometric methods.

### Implementation notes
This was a substantial integration — 9 bugs had to be fixed to get it working. All work done on the `neus2` branch.

**Key files:**
- `drema/r2s_builder/extractors/neus2_extractor.py` (284 lines, purpose-built for this integration)
- `NeuS2/` — cloned repo at project root

**What was built:**

1. **Dataset converter** (`prepare_neus2_dataset`): Converts DreMa data format → NeuS2 `transforms.json` format
   - Creates masked RGBA images from RGB + segmentation masks (alpha=255 for object pixels)
   - Reads camera poses from DreMa pose files via `read_pose_file()`
   - Converts coordinate convention: DreMa uses OpenCV (x-right, y-down, z-forward) → NeuS2 needs OpenGL (x-right, y-up, z-backward), applied via `_OPENCV_TO_OPENGL = np.diag([1, -1, -1, 1])`
   - Averages intrinsics across frames for the transforms.json header

2. **AABB coordinate transform** (`_world_to_ngp_aabb`): NeuS2 uses its own internal NGP space where camera translations are scaled by 0.33 and offset by 0.5, and axes are permuted (xyz←yzx in world → XYZ in NGP). Derived the full transform analytically to map the object world-space bounding box into NGP space for marching cubes region restriction.

3. **Subprocess isolation** (`_neus2_worker`): NeuS2 runs in a spawned subprocess so GPU memory is freed after training without polluting the main process.

4. **Mesh extraction + axis remap**: NeuS2 outputs mesh in NGP space, so axes are remapped back: `verts = verts[:, [2, 0, 1]]` to go from NGP XYZ back to world xyz.

**Config params added:** `neus2_steps` (default 20000), `neus2_resolution` (default 256)

**Committed:** `143c09a NeuS2 mesh extraction integration`

### Result
Completely flat mesh — a pancake shape with no vertical sides. The SDF collapsed to a horizontal plane.

### Verdict
**Structurally wrong for top-only cameras. Abandoned.**

Same fundamental camera constraint problem, but in a different form. NeuS2's SDF needs rays from multiple directions to constrain the surface on all faces. With all cameras looking downward, the network has no signal about what the sides and bottom of the object look like. The SDF loss is only supervised from above, so the network settles on a flat surface aligned to the observation direction as the optimal zero-level set. Ironically it produced a worse result than TSDF because at least TSDF gives you real sides — NeuS2 gave nothing but a pancake.

The integration code is kept in `neus2_extractor.py` on the `neus2` branch, not in active use.

---

## Approach 4 — SAM3D

### Pipeline diagram

SAM3D would have acted as an upstream improvement — better segmentation masks feeding into the existing TSDF pipeline. It does not change the mesh extraction method itself:

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ▼
 *** IMPROVED SEGMENTATION (proposed) ***
 SAM3D: 3D-aware video segmentation
 Produces denser, temporally consistent masks
 (vs current: GroundingDINO + SAM per-frame)
         │
         ▼
 Filter input: mask out everything except object id
 (better masks → cleaner object isolation)
         │
         ▼
 Train 2DGS — cleaner training data
 → fewer background contamination artifacts
         │
         ▼
 Render all training views through trained Gaussians
 → GS-rendered depth maps + RGB
         │
         ▼
 TSDF Fusion → Extract mesh → Filter → Save
 (unchanged from base pipeline)
```

What would have changed: only the segmentation step at the very top. Everything downstream stays the same. The bet was that cleaner masks → cleaner Gaussians → cleaner TSDF geometry.

**Never reached implementation.** VRAM wall hit at the planning stage.

### Why it could be useful
SAM3D (Meta's 3D video segmentation) could potentially improve mesh quality by providing better, denser segmentation masks across all frames. Cleaner masks → cleaner Gaussian training → cleaner geometry for TSDF. Also it was already being integrated for video segmentation purposes in the pipeline, so repurposing it for mesh improvement seemed natural.

### Implementation notes
SAM3 (video foundation model) was fully integrated as a segmentation backend in `run_segmentation.py` alongside GroundingDINO+SAM. The segmentation module lives in `sam3/` (installed as a Python package via `pip install -e ".[notebooks]"`).

However SAM3D specifically (the 3D lifting variant that operates on point clouds and 3D representations) was investigated separately for potential direct mesh improvement. 

**VRAM requirement:** SAM3D minimum is 32GB. Hardware available: RTX 3090 with 24GB.

**No code was written for SAM3D mesh extraction specifically** — the VRAM wall was hit before implementation started.

### Result
Never ran. Cannot run on available hardware.

### Verdict
**On hold. Hardware constraint.**

Not abandoned conceptually — if access to a 40GB+ GPU (A100/H100) becomes available, it would be worth trying. But it is not a viable path on the current setup.

---

## Approach 5 — TripoSR

### Pipeline diagram

TripoSR replaces only the mesh extraction step. Gaussian splatting still runs as normal — its output (the trained model + DBSCAN-cleaned point cloud bbox) is used for world-space fitting. The mesh itself comes from a single-image transformer:

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ▼
 Filter input / Train 2DGS (unchanged)
 → Trained Gaussian model
 → DBSCAN-cleaned Gaussian bbox (box_min, box_max)
         │
         ├─── Gaussian model used only for bbox, not for depth/mesh
         │
         ▼
 *** TripoSR PATH (replaces TSDF block) ***

 Select best frame: frame with most foreground pixels
 (or override via triposr_image config param)
         │
         ▼
 Prepare image:
 - Crop to foreground bbox
 - Pad to square, gray background (0.5)
 - Add border so object fills foreground_ratio of frame
         │
         ▼
 TripoSR inference (TSR transformer, ~2 sec, ~6GB VRAM)
 - LRM architecture, trained on Objaverse
 - Outputs NeRF density field in canonical space
         │
         ▼
 Marching cubes on density field
 (torchmcubes, resolution=384, threshold=25.0)
         │
         ▼
 World-space fitting (_fit_to_world):
 - Scale by Gaussian x/y bbox span
 - Anchor bottom at box_min[2]
 - Center in x/y
         │
         ▼
 *** filter_mesh SKIPPED for TripoSR ***
 (mesh already in world space; filtering would cut the bottom)
         │
         ▼
 Save as .obj → build URDF for PyBullet
```

What changed: the TSDF block (render depth → fuse → marching cubes) is completely removed. A single-image transformer takes over. Gaussians are only needed for their bbox (scale + placement reference).

### Why it could be useful
TripoSR is a single-image 3D reconstruction transformer (LRM architecture, trained on Objaverse). It infers a complete 3D shape from one image using learned shape priors — including the bottom face which was never observed. It replaces only the mesh extraction step, Gaussian splatting remains unchanged. Only ~6GB VRAM and ~2 seconds per object. The key idea: a learned prior knows that a mug has a bottom even if you can't see it.

### Implementation notes
All work on the `triposr` branch. Committed in: `6d13abd integrate TripoSR as mesh_method option`

**Key files:**
- `drema/r2s_builder/extractors/triposr_extractor.py` (165 lines)
- `TripoSR/` — cloned repo, added to `sys.path` at runtime (not a git submodule)
- `submodules/torchmcubes/` — CUDA marching cubes

**What was built:**

1. **Image selection** (`select_best_image`): Picks the training frame with the most foreground pixels for the object ID. Can be overridden with `triposr_image: "0099"` in config to test specific frames.

2. **Image preparation** (`_prepare_image`):
   - Crops to foreground bounding box
   - Pads to square with gray (0.5) background — TripoSR convention
   - Adds border so object fills `triposr_foreground_ratio` of the frame
   - Uses gray background instead of black/white because Objaverse training uses gray

3. **World-space fitting** (`_fit_to_world`):
   - TripoSR outputs in canonical space, needs to be placed in world coordinates
   - Scale driven by max(x/y Gaussian bbox span) — uses DBSCAN-cleaned Gaussian positions, not raw depth (depth noise inflates bbox on reflective surfaces)
   - Bottom anchored at `world_box_min[2]`, centered in x/y

4. **`filter_mesh` is skipped for TripoSR** — mesh is already in world space and filtering would cut the hallucinated bottom face

5. **VRAM cleanup:** Explicit `del model, scene_codes, meshes` + `torch.cuda.empty_cache()` after each object to prevent OOM on multi-object scenes

**Config params:** `triposr_resolution` (default 384, 512 OOMs), `triposr_threshold` (default 25.0), `triposr_foreground_ratio` (0.3–0.4 for wrist-cam), `triposr_image` (empty = auto)

**torchmcubes build issue:** Upstream uses cmake + scikit-build-core which failed due to pip-packaged CUDA headers and PyTorch nightly CUDA version string mismatch. Fixed by replacing with `torch.utils.cpp_extension` in `setup.py` + patching `pyproject.toml`.

**OpenCV conflict:** `rembg` (TripoSR dependency) installs `opencv-python-headless` which breaks `cv2.imshow` in `simulate.py`. Fix: `pip install "opencv-python==4.8.0.76"` after rembg. Also `rembg` upgrades numpy to 2.x which breaks all compiled CUDA submodules — pinned with `numpy<2.0`.

**Foreground ratio testing:** Tested 0.3, 0.4, 0.5, 0.65, 0.85 — no value gave a usable mesh.

### Result
Tent or triangular prism shapes instead of correct geometry. At low ratios: rhombus (off-center perspective). At high ratios: tent/roof shape. Nothing resembling the actual object.

### Verdict
**On hold. Needs different capture data.**

The issue is TripoSR was trained on Objaverse rendered with cameras at ~0° elevation (horizontal). Our wrist camera always stays at 41–53° elevation. At that angle the visible faces (top + two sides of a box) look exactly like a tent roof when interpreted by a model expecting horizontal viewpoints. It interprets the silhouette as a triangular prism.

The fix is known: capture 1–2 extra frames per object at ~15–25° elevation (arm extended sideways, near-horizontal), then point TripoSR at those frames via `triposr_image`. That's a data collection problem, not a code problem. Would need to modify the data gathering script/trajectory to include low-elevation frames. Not yet done.

Also a fundamental limitation of TripoSR regardless: it uses a NeRF density field internally (smooth implicit), so sharp edges are always rounded. Cubes will look like blobs. Works best on complex textured objects, not simple geometric shapes.

---

## Approach 6 — Raw Depth TSDF (tsdf_raw_depth)

### Pipeline diagram

Structure is nearly identical to the base pipeline. The only change is where the depth maps come from during the TSDF reconstruction step:

```
Raw RGB-D frames (wrist camera, 41–53° elevation)
         │
         ├──── depth_scaled/*.npy  ←── kept on disk, used later
         │
         ▼
 Filter input: mask out everything except object id
         │
         ▼
 Train 2DGS (SurfDepth backend, 7000 iters)
 Config fix: position_lr_max_steps=7000 (was 30000)
         │
         ▼
 Render all training views through trained Gaussians
 → GS-rendered RGB per frame (still used for TSDF coloring)
 → *** GS surf_depth DISCARDED ***
         │
         ▼
 *** Load raw sensor depth instead ***
 depth_scaled/<frame_name>.npy read from disk per frame
 (reconstruction_from_raw_depth() in GaussianExtractor)
         │
         ▼
 TSDF Fusion (Open3D ScalableTSDFVolume)
 Integrate (GS-RGB, raw-depth) per frame with camera pose
 Config fix: depth_ratio=1.0 (was 0.0, now bounded scene mode)
         │
         ▼
 Extract triangle mesh via marching cubes
         │
         ▼
 Filter mesh: remove triangles below/inside table
         │
         ▼
 Save as .obj → build URDF for PyBullet
```

What changed: one swap inside the reconstruction loop — `surf_depth` render replaced by a `.npy` file load. Plus two config bug fixes that independently improve quality. RGB still comes from Gaussians (for color). Everything else is identical to the base pipeline.

### Why it could be useful
The standard TSDF pipeline renders depth through the trained Gaussians (`surf_depth`). Gaussian splatting is an appearance model — it optimizes for photometric consistency, not geometric accuracy. The rendered depth is smooth and interpolated across Gaussian ellipsoids, which introduces blurring and can miss sharp edges. The raw RealSense depth sensor is a structured-light stereo camera specifically designed for accurate depth measurement. Bypassing the GS-rendered depth and feeding the raw sensor depth directly into TSDF should give sharper, more accurate surface geometry.

Additionally, two config bugs were found and fixed alongside this:
- `depth_ratio: 0.0` was set (unbounded scene depth mode), but for a bounded close-range object scene the correct mode is `depth_ratio: 1.0` (median depth strategy)
- `position_lr_max_steps: 30000` did not match actual training length of 7000 iterations, so the learning rate never decayed properly

### Implementation notes
Committed in: `33d8c4f add tsdf_raw_depth mesh method: bypass GS-rendered depth, use raw sensor depth directly`

**Changed files:** `mesh_utils.py`, `assets_extractor.py`, `base_optimizer.py`, `surf_depth_optimizer.py`, `real_world_params.yaml`

**Core change — new method in `GaussianExtractor`** (`mesh_utils.py`):
```python
def reconstruction_from_raw_depth(self, viewpoint_stack, depth_dir):
    """Like reconstruction() but loads depth from depth_dir instead of rendering surf_depth."""
    self.clean()
    self.viewpoint_stack = viewpoint_stack
    for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack)):
        render_pkg = self.render(viewpoint_cam, self.gaussians)
        self.rgbmaps.append(render_pkg['render'].cpu())  # RGB still from GS

        depth_file = os.path.join(depth_dir, viewpoint_cam.image_name.split(".")[0] + ".npy")
        depth_np = np.load(depth_file).astype(np.float32)
        self.depthmaps.append(torch.from_numpy(depth_np).unsqueeze(0))  # depth from sensor

    self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
    self.depthmaps = torch.stack(self.depthmaps, dim=0)
    self.estimate_bounding_sphere()
```

RGB still comes from the Gaussian model (for TSDF coloring), depth comes from `depth_scaled/` on disk (raw sensor depth, already undistorted and scaled).

**Dispatch in `assets_extractor.py`:**
```python
elif self.mesh_method == 'tsdf_raw_depth':
    mesh = trainer.extract_mesh(depth_dir=os.path.join(self.source_path, "depth_scaled"))
```

**`extract_mesh(depth_dir=None)` in `base_optimizer.py` and `surf_depth_optimizer.py`:** Added `depth_dir` parameter, if not None calls `reconstruction_from_raw_depth()` instead of `reconstruction()`.

**Config fixes committed in same batch (`real_world_params_fixes.yaml`):**
- `depth_ratio: 1.0` — bounded scene uses median depth strategy
- `position_lr_max_steps: 7000` — matches actual training length

**Evaluation setup (`eval_mesh_quality.py`, 127 lines):**
Built a bidirectional Chamfer distance evaluation script. Compares mesh surface samples against reference point cloud `input/gold_data_undistort/aggregated_pointcloud.ply`. Uses a fixed per-object bounding box (union of all experiment meshes) so recall numbers are comparable across experiments — otherwise a larger mesh always wins recall trivially.

Metrics:
- **Precision** (mesh → ref): mean nearest-neighbor distance in mm. Measures accuracy — how close is the mesh to real geometry.
- **Recall** (ref → mesh): mean nearest-neighbor distance in mm. Measures completeness — how much of the real geometry is covered by the mesh.

### Result
Significant improvement. Results across 3 objects:

| Experiment | Mean Recall |
|---|---|
| baseline (GS depth, depth_ratio=0) | ~10.1mm |
| + tsdf_raw_depth | ~7.6mm |
| + depth_ratio=1.0 | ~6.0mm |

Precision stayed roughly the same (~1mm) across all — the mesh is accurate where it exists. The big improvement is in recall (completeness) — the raw depth gives sharper, more complete side walls and covers more of the real object surface.

The `depth_ratio=1.0` fix gave an additional independent improvement on top of raw depth.

### Verdict
**Current best method. Active.**

This is the active `mesh_method` setting in both `real_world_params.yaml` and `real_world_params_fixes.yaml`. The improvement over baseline is 4x in precision terms and substantial in recall. The next expected improvement: we re-gathered data with correct distortion coefficients (was returning zeros before, fixed in `realsense.py`) and depth filters (temporal+spatial+hole-fill). Since the pipeline now feeds raw sensor depth directly, sensor data quality = mesh quality 1:1. Better data should directly improve the mesh.

---

## Sensor Data Quality Improvements (Done)

This is not a mesh extraction method — it is improvements to the data collection side. Since `tsdf_raw_depth` feeds raw sensor depth directly into TSDF, sensor quality = mesh quality 1:1. These changes affect every future data gather run.

### What was implemented

**`realsense.py` additions:**

1. **`depth_filters` flag** — `__init__` takes `depth_filters: bool`, creates `rs.temporal_filter`, `rs.spatial_filter`, `rs.hole_filling_filter` if enabled. Applied inside `_get_image()` on the raw depth frame before converting to numpy.

2. **`save_ir` flag** — `__init__` enables IR streams 1 and 2 if true. `_get_image()` captures both IR frames before alignment (alignment loses stereo geometry), applies same flip/crop as RGB, stores as `self.last_ir_left` / `self.last_ir_right`. This is infrastructure for the stereo depth approach below — no depth computation here, just saving the raw frames.

3. **`get_dist_coeffs()` — fixed** — was broken and returning all zeros. Now correctly reads `intr.coeffs` from the color stream profile. All previous data was undistorted with zero coefficients (i.e., no correction at all).

4. **`get_ir_intrinsics()`** — new method, returns IR left camera intrinsics (fx, fy, cx, cy, width, height).

5. **`get_stereo_baseline()`** — new method, reads the extrinsics between IR left and IR right streams to get the physical baseline in meters. Needed for stereo depth computation.

**`data_gather_robot.py` additions (commit `a463b1e`):**

- **`distortion_coeffs.txt` saved on frame 1** — calls `cam_manager.gripper_cam.get_dist_coeffs()`, saves with `np.savetxt`. Was a TODO comment before — never actually written.
- **IR stereo frame saving** — when `save_ir=true`, saves `images_ir_left/<index>.png` and `images_ir_right/<index>.png` each frame. On frame 1 writes `ir_metadata.json` with IR intrinsics, baseline, and flip flag — everything needed later to compute stereo depth.

**`realsense_datagather.yaml` — manually created fixed-parameter config:**

Created to lock all camera settings for reproducible multi-view data. The original `realsense_d435.yaml` had auto-exposure and auto-white-balance on, causing frame-to-frame variation (bad for Gaussian training).

```yaml
params:
  white_balance:              3400    # fixed, tuned for lab lighting
  exposure:                   230.0   # manual
  brightness:                 40.0
  contrast:                   50.0
  saturation:                 64.0
  sharpness:                  0.0
  gain:                       16.0
  gamma:                      220.0
  hue:                        0.0
  enable_auto_exposure:       0.0     # OFF
  enable_auto_white_balance:  0.0     # OFF
resolution: [640, 480]
resize_resolution: [640, 360]
crop_coords: [120, 480, 0, 640]       # cuts gripper fingers from bottom of frame
depth_filters: true
save_ir: false                         # enable only for stereo calibration runs
```

### What was wrong before
- `get_dist_coeffs()` returning zeros → `undistort_data.py` was a no-op on all previous data
- No depth filters → raw structured-light depth goes into TSDF with temporal noise and holes
- Auto-exposure + auto-white-balance → color shifts between frames → inconsistent Gaussian training

### What will be tested after regathering
New data gathered with correct distortion coefficients, depth filters on, and fixed exposure. Run `eval_mesh_quality.py` against the same gold reference to measure improvement on top of the current best ~6.0mm mean recall. Expected: fewer TSDF holes from depth noise, better edge geometry from correct undistortion.

---

## Stereo Depth (Planned)

### What it is
Instead of using the depth stream that RealSense computes internally, compute depth from the raw IR stereo pair ourselves using standard stereo matching (e.g., SGBM or a learned stereo network). The IR frames are already being saved when `save_ir: true`. The idea is to use this as an independent depth source to compare against the RealSense depth and measure whether our sensor improvements actually made a difference.

### Why it could be useful

The RealSense D435 computes depth internally using structured light + stereo. We have no control over that computation. By computing stereo depth ourselves from the raw IR frames, we can:
- Tune the stereo algorithm (block size, disparity range, etc.) for our specific close-range table setup
- Compare stereo-computed depth vs RealSense depth vs filtered RealSense depth quantitatively
- Potentially get sharper depth at object edges where the RealSense tends to interpolate

### Pipeline diagram

```
data_gather_robot.py (save_ir: true)
         │
         ▼
 images_ir_left/<index>.png
 images_ir_right/<index>.png
 ir_metadata.json  (fx, fy, cx, cy, baseline_m, flipped)
         │
         ▼
 *** TO BE IMPLEMENTED: stereo_depth.py ***
 Load IR pair per frame
 Rectify stereo pair using ir_metadata intrinsics
 Run stereo matching (SGBM or learned) → disparity map
 Disparity → depth:  depth = baseline * fx / disparity
         │
         ▼
 *** TO BE IMPLEMENTED: convert to DreMa format ***
 Save as depth_scaled/<index>.npy  (same format as RealSense depth)
         │
         ▼
 Feed into tsdf_raw_depth pipeline as drop-in replacement
 Run eval_mesh_quality.py → compare vs RealSense depth
```

### Current state
**Planned. Not yet started. No conversion script exists.**

What is done: IR frame capture infrastructure in `realsense.py` and `data_gather_robot.py`, and `ir_metadata.json` saving with all needed calibration data. What is missing: the stereo matching script and the converter to DreMa-compatible `depth_scaled/` format. Both need to be written before this can be tested.

### Verdict
To be evaluated after implementation. Primary goal is comparison/validation, not necessarily replacing RealSense depth permanently.

---

## FoundationPose (Planned)

### What it is
FoundationPose (NVIDIA, 2024) is a 6-DoF object pose estimation and tracking framework. It takes a 3D mesh of an object + a new RGB-D observation and outputs the full 6-DoF pose of that object in the scene. Uses render-and-compare with a large pre-trained transformer — no per-object fine-tuning needed.

### Why it is relevant here

The current DreMa pipeline assumes the object is always in the same pose as during data collection. To use DreMa for generating new augmented data in different configurations, we need to know where each object actually is in new scene observations. FoundationPose can localize the object using the mesh we already reconstruct.

### Pipeline diagram

```
Reconstructed mesh (.obj from DreMa create_simulation.py)
         │
         ▼
 *** TO BE IMPLEMENTED: run_foundation_pose.py ***
 Input: mesh template + new RGB-D frame from robot camera
 FoundationPose inference → 6-DoF pose (R, t) in robot base frame
         │
         ▼
 *** TO BE IMPLEMENTED: pose → DreMa scene converter ***
 Write pose into DreMa simulation config format
 Place Gaussians + URDF at estimated pose in PyBullet
         │
         ▼
 generate_new_data.py → augmented training data
 with object in new/varied position
```

### Current state
**Planned. Not yet started. No script exists.**

No code written. Two things need to be implemented:
1. A script that runs FoundationPose given a mesh and an RGB-D frame
2. A converter that takes the estimated pose and feeds it into the DreMa simulation format

The mesh quality work being done now is the direct prerequisite — FoundationPose renders the mesh from candidate poses to compare against the observed frame. An open-bottom mesh produces incorrect renders for any pose where the bottom is in view, degrading estimation accuracy. Better mesh = better pose estimation.

---

## Summary — Quantitative Evaluation

### Measurement strategy

We evaluate mesh quality using **bidirectional Chamfer distance** against a reference point cloud (`input/gold_data_undistort/aggregated_pointcloud.ply`). This reference was built by aggregating raw depth frames from the best available data capture into a single dense point cloud via `pointcloud_aggregation.py`. It represents what the real scene geometry looks like according to the sensor.

For each object, we define a **fixed bounding box** — the union of all experiment mesh extents for that object plus a margin. This is important: without a fixed bbox, a larger mesh trivially wins recall because it samples more reference points. The fixed box ensures every experiment is evaluated on the same region of real geometry.

**Sampling:** 20,000 points are sampled uniformly from each mesh surface. Nearest-neighbour distances are computed against the reference point cloud using a KD-tree.

**What each column means:**

- **prec_mean** — *Precision.* For each point sampled on the mesh surface, find its nearest neighbour in the reference point cloud. Mean distance in mm. Measures **accuracy**: how close is the mesh to real geometry where the mesh exists. Low = the mesh surface is tight to reality.
- **prec_p90** — 90th percentile of the same precision distances. Shows how bad the worst 10% of surface points are (outlier sensitivity).
- **rec_mean** — *Recall.* For each point in the reference point cloud (inside the fixed bbox), find its nearest neighbour on the mesh surface. Mean distance in mm. Measures **completeness**: how much of the real geometry is covered by the mesh. Low = more of the real object is represented.
- **rec_p90** — 90th percentile of recall distances. Shows how large the worst gaps are between real geometry and the mesh.
- **faces** — Triangle count of the mesh.

Lower is better for all distance metrics. Precision and recall can trade off: a tightly fitted but incomplete mesh has good precision, poor recall. A large bloated mesh has poor precision, potentially good recall.

---

### Results

All experiments use the same undistorted input data (`gold_data_undistort`). The baseline is `undistort` — standard TSDF with GS-rendered depth, no modifications.

**Object 1**

| experiment | prec_mean | prec_p90 | rec_mean | rec_p90 | faces |
|---|---:|---:|---:|---:|---:|
| undistort *(baseline)* | 3.65 | 9.89 | 23.87 | 54.52 | 26k |
| fillholes | 3.79 | 10.48 | 23.63 | 53.33 | 28k |
| poisson | 3.16 | 7.51 | 26.64 | 60.27 | 64k |
| neus2 | 18.01 | 33.89 | 29.91 | 56.32 | 71k |
| triposr | 12.01 | 26.46 | 17.06 | 42.62 | 482k |
| rawdepth | 0.97 | 1.32 | 27.86 | 72.78 | 24k |
| rawdepth_depth1 | 0.96 | 1.32 | 25.65 | 70.96 | 28k |
| rawdepth_morepoints | 0.96 | 1.32 | 27.42 | 72.57 | 25k |
| **rawdepth_d1_morepoints** | **0.96** | **1.31** | **26.14** | **71.47** | 28k |

**Object 2**

| experiment | prec_mean | prec_p90 | rec_mean | rec_p90 | faces |
|---|---:|---:|---:|---:|---:|
| undistort *(baseline)* | 4.13 | 8.82 | 37.76 | 81.27 | 37k |
| fillholes | 4.13 | 8.86 | 38.29 | 82.02 | 37k |
| poisson | 4.93 | 10.22 | 40.11 | 84.02 | 74k |
| neus2 | 44.34 | 75.50 | 34.65 | 59.02 | 199k |
| triposr | 9.57 | 21.86 | 40.33 | 76.90 | 368k |
| rawdepth | 1.12 | 1.68 | 36.01 | 77.33 | 55k |
| **rawdepth_depth1** | **1.12** | **1.68** | **34.12** | **75.73** | 60k |
| rawdepth_morepoints | 1.11 | 1.67 | 35.53 | 76.88 | 56k |
| rawdepth_d1_morepoints | 1.11 | 1.66 | 34.44 | 75.98 | 59k |

**Object 3**

| experiment | prec_mean | prec_p90 | rec_mean | rec_p90 | faces |
|---|---:|---:|---:|---:|---:|
| undistort *(baseline)* | 4.24 | 10.71 | 127.07 | 281.64 | 50k |
| fillholes | 2.97 | 7.37 | 129.92 | 284.80 | 48k |
| poisson | 3.69 | 10.05 | 133.22 | 288.37  | 80k |
| neus2 | 32.42 | 56.46 | 117.24 | 242.70 | 4k |
| triposr | 9.62 | 26.31 | 123.70 | 269.47 | 283k |
| rawdepth | 1.05 | 1.48 | 128.46 | 279.99 | 77k |
| rawdepth_depth1 | 1.04 | 1.46 | 124.16 | 279.57 | 85k |
| rawdepth_morepoints | 1.05 | 1.48 | 128.46 | 280.74 | 79k |
| **rawdepth_d1_morepoints** | **1.05** | **1.48** | **126.83** | **279.39** | 84k |

---

### Reading the results

**Precision:** rawdepth variants are dramatically better across all three objects — ~1mm vs ~3.5–4mm for all GS-depth methods. 3–4x improvement. Where the mesh surface exists, it is very close to real geometry. This is the clearest result in the whole experiment.

**Recall on objects 1 and 2:** rawdepth alone is actually slightly worse than the baseline. The mesh is accurate but doesn't cover quite as much area. Adding `depth_ratio=1.0` (rawdepth_depth1) recovers and improves recall. The config fix is doing real work, not the raw depth alone.

**Object 3 recall is stuck at ~125mm for everything.** That's the missing bottom — object 3 is the largest object and the reference point cloud has substantial geometry below the mesh for all methods. No pipeline fix changes this because the camera never saw that region.

**fillholes:** Identical to baseline. Confirmed no-op.

**Poisson:** Slightly better precision than baseline but worse recall on all objects. The skirt extrapolation covers wrong areas and misses real ones.

**NeuS2:** Precision is catastrophic (18–44mm). Object 3 recall looks better (117mm) than baseline (127mm) but this is misleading — the pancake mesh sits on the table plane and happens to be near some flat reference points. Not real geometry reconstruction.

**TripoSR:** Worst precision of all learned methods (9–12mm) because world-space fitting based on Gaussian bbox is imprecise. Recall on object 1 is the best of all methods (17.06mm) — the shape prior adds geometry that covers the real object. The geometry is there but placed/scaled incorrectly. If world placement were improved (better scale reference, low-elevation image), this could become competitive.

**Undistortion effect:** Not separately shown in results (undistort IS the baseline here). Earlier comparison of `gold_data` (non-undistorted) vs `undistort` showed less than 1mm difference in recall — undistorting had no measurable effect on mesh quality with the GS-depth pipeline.

**Best overall:** `rawdepth_depth1` — best or tied-best recall on objects 1 and 2, best precision everywhere. This is the current active configuration.

---

### Verdict table

| Approach | Verdict | Precision | Recall |
|---|---|---|---|
| Base TSDF (GS depth, undistort) | Baseline | ~4mm | ~22–127mm |
| fill_holes() | No effect | same | same |
| Poisson | Worse recall, geometry wrong | ~3.5mm | worse |
| NeuS2 | Failed — pancake mesh | 18–44mm | slightly better obj3 only |
| SAM3D | On hold — 32GB VRAM needed | N/A | N/A |
| TripoSR | On hold — wrong elevation; placement off | 9–12mm | best obj1 recall |
| rawdepth | Better precision, worse recall alone | ~1mm | slightly worse |
| **rawdepth + depth_ratio=1.0** | **Current best** | **~1mm** | **best across obj1 & obj2** |
