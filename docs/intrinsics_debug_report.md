# Intrinsics Debug Report

**Date:** 2026-03-05  
**Branch:** `fix-intrinsics`  
**Problem:** DreMa gaussian splatting produces bad reconstruction despite correct RGB images and correct robot poses.

---

## 1. Setup

- 300 RGB-D images from RealSense D435 mounted on Franka Panda
- Poses from robot FK + hand-eye calibration (`T_base_tcp @ T_tcp_cam`)
- Same images run through COLMAP → nerfstudio GS → **perfect reconstruction**
- Same images with robot poses → nerfstudio GS → **bad reconstruction**
- Conclusion: images are fine, something is wrong in how poses/intrinsics are used

---

## 2. Pose Verification

### Test: relative rotation consistency between robot and COLMAP poses

```python
import json, numpy as np

with open('input/test1/transforms.json') as f:
    robot = json.load(f)
with open('input/test1_colmap/transforms.json') as f:
    colmap = json.load(f)

r0 = np.array(robot['frames'][0]['transform_matrix'])
r150 = np.array(robot['frames'][149]['transform_matrix'])
rel_robot = np.linalg.inv(r0) @ r150

c0 = np.array(colmap['frames'][0]['transform_matrix'])
c150 = np.array(colmap['frames'][149]['transform_matrix'])
rel_colmap = np.linalg.inv(c0) @ c150

# rotation angle between frame 0 and 150
angle_robot = np.degrees(np.arccos(np.clip((np.trace(rel_robot[:3,:3])-1)/2, -1, 1)))
angle_colmap = np.degrees(np.arccos(np.clip((np.trace(rel_colmap[:3,:3])-1)/2, -1, 1)))
```

**Output:**
```
Robot:  42.65°
COLMAP: 42.51°
```

**Result:** 0.14° difference — poses are geometrically correct. Problem is elsewhere.

---

## 3. Calibration Matrix Verification

```python
import numpy as np

T = np.load('input/panda_realsenseD435_T_tcp_cam.npy')
R = T[:3,:3]
print(f'det(R) = {np.linalg.det(R):.6f}')
print(f'R^T @ R = I ? max error: {np.max(np.abs(R.T @ R - np.eye(3))):.6e}')
print(f'Camera Z in TCP frame: {R[:, 2]}')
print(f'Translation: {T[:3, 3]}')
```

**Output:**
```
det(R) = 1.000000
R^T @ R = I ? max error: 6.661338e-16
Camera Z in TCP frame: [-0.46  0.004  0.888]
Translation: [ 0.121  0.021 -0.107]
```

**Result:** Valid rotation matrix, physically reasonable mounting position. Calibration is correct.

---

## 4. The Bug: `read_txt_intrinsics` computes wrong image dimensions

### Code in `drema/utils/drema_camera_utils.py` (before fix):

```python
width  = given_intrinsics[0,2] * 2  # cx * 2
height = given_intrinsics[1,2] * 2  # cy * 2
```

### Test:

```python
# From pose file
fx, fy, cx, cy = 606.65625, 605.260742, 326.685974, 121.651581

# What DreMa computes
drema_w = cx * 2  # = 653.4
drema_h = cy * 2  # = 243.3

# Actual image size
import cv2
img = cv2.imread('input/test1/images/0001.png')
actual_h, actual_w = img.shape[:2]  # 360, 640
```

**Output:**
```
Actual image size:              640 × 360
DreMa computed size (cx*2, cy*2): 653.4 × 243.3
Width error:  13.4 px
Height error: 116.7 px
```

### Why `cy * 2 ≠ height`

The formula assumes `cy = height / 2` (principal point at image center). The RealSense D435 has `cy = 121.7` for a 360px tall image — the optical center is in the top third, not the middle. This is normal for real cameras.

COLMAP independently estimated `cy = 124.5`, confirming the camera's reported value is correct.

### Impact on Field of View:

```python
import math
fov_y_correct = 2 * math.atan(360 / (2 * 605.26))    # 33.12°
fov_y_drema   = 2 * math.atan(243.3 / (2 * 605.26))  # 22.73°
```

**Output:**
```
Correct FoV Y: 33.12°
DreMa FoV Y:   22.73°  (31.4% error)
```

---

## 5. Second Bug: projection matrix assumes centered principal point

`getProjectionMatrix` in `graphics_utils.py` builds a symmetric frustum:

```python
top    =  tanHalfFovY * znear
bottom = -top     # assumes cy = h/2
right  =  tanHalfFovX * znear
left   = -right   # assumes cx = w/2
```

This projects the optical axis to pixel `(w/2, h/2) = (320, 180)` instead of the real `(326.7, 121.7)` — a **58px vertical error**.

---

## 6. Fix Applied

### Fix 1: `read_txt_intrinsics` — read actual image dimensions

```python
# BEFORE:
width  = given_intrinsics[0,2] * 2
height = given_intrinsics[1,2] * 2

# AFTER:
images_dir = os.path.join(path, "images")
image_files = sorted([f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
first_image = Image.open(os.path.join(images_dir, image_files[0]))
width, height = first_image.size
```

### Fix 2: `getProjectionMatrix` — support off-center principal point

```python
# Added optional parameters:
def getProjectionMatrix(znear, zfar, fovX, fovY, cx=None, cy=None, img_w=None, img_h=None):
    # ... existing symmetric frustum code ...

    # Apply principal point offset if provided
    if cx is not None and img_w is not None:
        P[0, 2] = (2 * cx - img_w + 1) / img_w
    if cy is not None and img_h is not None:
        P[1, 2] = (2 * cy - img_h + 1) / img_h
```

### Fix 3: Thread `cx, cy` through the pipeline

`CameraInfo` → `CameraInfoDepth` → `loadCam` → `DepthCamera` → `Camera` → `getProjectionMatrix`

All with optional `cx=None, cy=None` defaults so existing COLMAP paths are unaffected.

### Fix 4: `fetchTxtPly` — use real principal point for depth unprojection

```python
# BEFORE:
K = np.array([[fx, 0, camera.width/2], [0, fy, camera.height/2], [0, 0, 1]])

# AFTER:
ppx = camera.cx if camera.cx is not None else camera.width / 2
ppy = camera.cy if camera.cy is not None else camera.height / 2
K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
```

---

## 7. Verification

```python
from drema.gaussian_splatting_utils.graphics_utils import getProjectionMatrix

fx, fy, cx, cy = 606.65625, 605.260742, 326.685974, 121.651581
W, H = 640, 360
fovX = 2 * math.atan(W / (2 * fx))
fovY = 2 * math.atan(H / (2 * fy))

# BEFORE
P_old = getProjectionMatrix(0.0001, 50.0, fovX, fovY)
px_x = P_old[0,2].item() * W/2 + (W-1)/2  # 319.5
px_y = P_old[1,2].item() * H/2 + (H-1)/2  # 179.5

# AFTER
P_new = getProjectionMatrix(0.0001, 50.0, fovX, fovY, cx=cx, cy=cy, img_w=W, img_h=H)
px_x = P_new[0,2].item() * W/2 + (W-1)/2  # 326.7
px_y = P_new[1,2].item() * H/2 + (H-1)/2  # 121.7
```

**Output:**
```
BEFORE FIX:
  Optical axis projects to pixel: (319.5, 179.5)
  Should be:                      (326.7, 121.7)
  Error:                          (7.2px, 57.8px)

AFTER FIX:
  Optical axis projects to pixel: (326.7, 121.7)
  Should be:                      (326.7, 121.7)
  Error:                          (0.0px, 0.0px)
```

---

## 8. Files Changed

| File | Change |
|------|--------|
| `drema/utils/drema_camera_utils.py` | Read real image dims from `images/` folder |
| `drema/gaussian_splatting_utils/graphics_utils.py` | `getProjectionMatrix` with `cx,cy` support |
| `drema/scene/dataset_readers.py` | `CameraInfo` + `readColmapCameras`: extract and pass `cx,cy` |
| `drema/drema_scene/drema_dataset_readers.py` | `CameraInfoDepth` + `fetchTxtPly`: pass and use `cx,cy` |
| `drema/scene/cameras.py` | `Camera.__init__`: accept `cx,cy`, pass to projection |
| `drema/drema_scene/drema_cameras.py` | `DepthCamera.__init__`: forward `cx,cy` |
| `drema/gaussian_splatting_utils/camera_utils.py` | `loadCam`: scale and forward `cx,cy` |

---

## 9. Summary

| Item | Status |
|------|--------|
| Robot poses | **Correct** — 0.14° match with COLMAP |
| Calibration matrix | **Correct** — valid SO(3), physically reasonable |
| Image dimensions in DreMa | **Bug** — `cy*2 = 243` instead of actual `360` |
| Principal point in projection | **Bug** — assumed centered, 58px off vertically |
| Fix backward-compatible | **Yes** — `cx=None` defaults preserve existing behavior |
