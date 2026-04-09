# TripoSR Setup Guide

## Overview

TripoSR is a single-image transformer (LRM architecture, Objaverse-trained) that reconstructs a full closed mesh from one RGB image. Used here to fix the open-bottom problem that wrist-mounted cameras create — TSDF gives 5 correct faces, TripoSR fills the 6th from shape priors.

~6 GB VRAM, ~2 sec per object.

## 1. Clone TripoSR

```bash
cd ~/Projects/drema/drema_project
git clone https://github.com/VAST-AI-Research/TripoSR.git
```

No submodules needed. The pipeline adds `TripoSR/` to `sys.path` at runtime.

## 2. Python dependencies

```bash
conda activate drema_env

pip install einops omegaconf transformers diffusers accelerate huggingface_hub
pip install rembg onnxruntime
pip install "numpy<2.0"
```

`rembg` + `onnxruntime` are imported unconditionally by `tsr/utils.py`. The numpy pin is required because rembg can upgrade numpy to 2.x, which breaks drema's compiled submodules.

## 3. Build torchmcubes

TripoSR uses torchmcubes for marching cubes. The upstream repo uses cmake + scikit-build-core which fails in this environment (pip-packaged CUDA headers, PyTorch nightly version mismatch). The submodule here has a replacement `setup.py` using `torch.utils.cpp_extension`.

```bash
conda activate drema_env
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST='12.0'   # RTX 50-series; adjust for your GPU

cd ~/Projects/drema/drema_project/submodules/torchmcubes
pip install -e .
```

torchmcubes dispatches based on tensor device: `vol.is_cuda` → `mcubes_cuda`, else `mcubes_cpu`. Both are compiled. As long as the model runs on GPU (default), marching cubes runs on GPU too.

## 4. Configure

In `configs/training/real_world_params.yaml`:

```yaml
assets:
  mesh_method: "triposr"

optimization:
  triposr_model: "stabilityai/TripoSR"
  triposr_resolution: 256
  triposr_chunk_size: 8192
```

The model weights (~1 GB) are downloaded automatically from HuggingFace on first run.

## 5. Run

```bash
python create_simulation.py --config configs/config_real.yaml
```

## How image selection works

For each object, `select_best_image` scans all frames in `object_masks/`, counts pixels equal to `object_id`, and picks the frame with the most foreground pixels. This gives the top-view frame where the object is most visible.

The selected frame is composited against a gray background (0.5 — TripoSR convention) before inference.

## World-space fitting

TripoSR outputs a mesh in its own canonical space. After extraction, `_fit_to_world` scales it to match the object's x/y bounding box (from Gaussian splatting), then places the bottom at `world_box_min[2]`.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: rembg` | `pip install rembg onnxruntime` |
| `RuntimeError: CUDA version mismatch` in torchmcubes | `setup.py` already patches this — rebuild with `pip install -e .` |
| `numpy` ABI errors | `pip install "numpy<2.0"` |
| OOM during TripoSR inference | Reduce `triposr_resolution` (e.g. 128) or `triposr_chunk_size` |
