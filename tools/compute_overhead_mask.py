"""
Compute a fixed overhead-camera mask from a single training image.

Since the overhead camera is static, black regions in the rendered training
data are always in the same place. Run this once to produce the mask, then
deploy_peract.py applies it at every inference step.

Usage:
    conda activate robotio
    python tools/compute_overhead_mask.py

Output:
    assets/calibration/overhead_mask_128x128.npy  -- bool array (H,W), True = valid pixel
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

DATA_ROOT    = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_data')
OUT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'assets', 'calibration',
                            'overhead_mask_128x128.npy')
IMAGE_SIZE   = (128, 128)
BLACK_THRESH = 5    # per-channel value below which a pixel is considered black
DILATION_PX  = 10  # expand valid region outward to keep mask loose


def main():
    imgs = glob.glob(os.path.join(DATA_ROOT, '**', 'overhead_rgb', '0.png'), recursive=True)
    assert imgs, f"No overhead images found under {DATA_ROOT}"

    # accumulate across a few episodes so sporadic dark scene pixels don't become mask
    accumulator = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.float32)
    n = min(20, len(imgs))
    for path in imgs[:n]:
        img = np.array(Image.open(path).resize(IMAGE_SIZE))  # (H,W,3)
        accumulator += np.any(img > BLACK_THRESH, axis=-1).astype(np.float32)

    # pixel is valid if it was non-black in ANY of the sampled images
    mask = accumulator > 0
    # expand valid region so the mask is not too tight
    mask = binary_dilation(mask, iterations=DILATION_PX)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, mask)
    print(f"Saved mask to {OUT_PATH}")
    print(f"Valid pixels: {mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")

    # save a visual preview
    preview = (mask * 255).astype(np.uint8)
    preview_path = OUT_PATH.replace('.npy', '_preview.png')
    Image.fromarray(preview).save(preview_path)
    print(f"Preview saved to {preview_path}")


if __name__ == '__main__':
    main()
