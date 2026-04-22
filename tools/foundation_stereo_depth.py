"""
Run FoundationStereo on RealSense IR stereo pairs and save metric depth as .npy files.

Reads ir_metadata.json from the data folder for intrinsics and baseline.
Outputs depth_fs/<frame>.npy (float32, meters) alongside the existing depth_scaled/.

Images are downscaled by --scale for inference then depth is upsampled back to original
resolution. Default scale=0.5 works on 11.5GB GPU; use 1.0 if you have more VRAM.

Usage (must run inside the foundation_stereo conda env):
    cd /path/to/FoundationStereo
    python /path/to/drema_project/tools/foundation_stereo_depth.py \
        --data_dir /path/to/drema_project/input/gello_woodencube3 \
        --ckpt_dir pretrained_models/23-51-11/model_best_bp2.pth
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def setup_fs_path(fs_repo: str):
    sys.path.insert(0, fs_repo)


def load_model(ckpt_dir: str):
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_dir), "cfg.yaml"))
    # cfg.yaml omits inference-only keys — fill defaults matching run_demo.py
    model_args = OmegaConf.merge(cfg, OmegaConf.create({
        "vit_size": "vitl",
        "valid_iters": 32,
        "mixed_precision": True,
        "corr_implementation": "reg",
    }))

    model = FoundationStereo(model_args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"Checkpoint global_step={ckpt['global_step']}, epoch={ckpt['epoch']}")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    return model, model_args


def sorted_images(folder: Path) -> list:
    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    try:
        files.sort(key=lambda p: int(p.stem))
    except ValueError:
        files.sort()
    return files


def run_stereo(model, model_args, left_np: np.ndarray, right_np: np.ndarray):
    from core.utils.utils import InputPadder

    if left_np.ndim == 2:
        left_np = np.stack([left_np] * 3, axis=-1)
        right_np = np.stack([right_np] * 3, axis=-1)

    H, W = left_np.shape[:2]
    img0 = torch.as_tensor(left_np).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_np).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.amp.autocast("cuda", enabled=model_args.mixed_precision):
        disp = model.forward(img0, img1, iters=model_args.valid_iters, test_mode=True)

    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)
    return disp


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="FoundationStereo batch depth from IR pairs")
    parser.add_argument("--data_dir", required=True, help="Folder with images_ir_left/, images_ir_right/, ir_metadata.json")
    parser.add_argument("--ckpt_dir", default="pretrained_models/23-51-11/model_best_bp2.pth")
    parser.add_argument("--out_subdir", default="depth_fs", help="Output subfolder name inside data_dir")
    parser.add_argument("--scale", default=0.5, type=float, help="Downscale images for inference (<=1). Depth is upsampled back. Default 0.5 fits 11.5GB GPU.")
    parser.add_argument("--fs_repo", default=os.getcwd(), help="Path to FoundationStereo repo root (default: cwd)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)

    meta = json.loads((data_dir / "ir_metadata.json").read_text())
    fx = meta["intrinsics"]["fx"]
    baseline = meta["baseline_m"]
    logging.info(f"Camera: fx={fx:.3f}, baseline={baseline*100:.2f}cm, inference scale={args.scale}")

    setup_fs_path(args.fs_repo)
    logging.info("Loading FoundationStereo model...")
    model, model_args = load_model(args.ckpt_dir)
    logging.info("Model loaded.")

    left_files = sorted_images(data_dir / "images_ir_left")
    right_files = sorted_images(data_dir / "images_ir_right")
    assert len(left_files) == len(right_files), "Left/right frame count mismatch"
    logging.info(f"Processing {len(left_files)} frames → {out_dir}")

    for left_path, right_path in tqdm(zip(left_files, right_files), total=len(left_files)):
        assert left_path.stem == right_path.stem, f"Frame mismatch: {left_path.stem} vs {right_path.stem}"

        left_np = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right_np = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        H_orig, W_orig = left_np.shape[:2]

        if args.scale < 1.0:
            left_s = cv2.resize(left_np, dsize=None, fx=args.scale, fy=args.scale)
            right_s = cv2.resize(right_np, dsize=None, fx=args.scale, fy=args.scale)
        else:
            left_s, right_s = left_np, right_np

        disp = run_stereo(model, model_args, left_s, right_s)

        # Mask pixels where right image has no coverage
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
        disp[xx - disp < 0] = np.inf

        # Disparity → metric depth using scaled fx
        fx_scaled = fx * args.scale
        with np.errstate(divide="ignore"):
            depth = fx_scaled * baseline / disp
        depth[~np.isfinite(depth)] = 0.0

        # Upsample back to original resolution
        if args.scale < 1.0:
            depth = cv2.resize(depth, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        np.save(out_dir / f"{left_path.stem}.npy", depth.astype(np.float32))
        torch.cuda.empty_cache()

    logging.info(f"Done. Saved {len(left_files)} depth maps to {out_dir}")


if __name__ == "__main__":
    main()
