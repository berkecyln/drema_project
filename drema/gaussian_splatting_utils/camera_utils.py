#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from drema.drema_scene.drema_cameras import DepthCamera
from drema.scene.cameras import Camera
import numpy as np
from drema.gaussian_splatting_utils.graphics_utils import fov2focal
from drema.gaussian_splatting_utils.general_utils import PILtoTorch

import torch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    if cam_info.depth_image is not None:
        depth_image = torch.from_numpy(np.array(cam_info.depth_image))
        resized_image_depth = torch.nn.functional.interpolate(depth_image.unsqueeze(0).unsqueeze(0).float(),
                                                                size=resized_image_rgb.shape[-2:],
                                                                mode='nearest').squeeze(0).squeeze(0)

        loaded_mask = torch.ones(resized_image_rgb.shape[1:]).unsqueeze(dim=0)
        loaded_mask[0, resized_image_depth <= 0] = 0
        #loaded_mask = None
    else:
        resized_image_depth = None
        loaded_mask = None

    gt_image = resized_image_rgb[:3, ...]

    # Pass principal point (cx, cy) if available; scale them with image resolution
    cx = getattr(cam_info, 'cx', None)
    cy = getattr(cam_info, 'cy', None)
    if cx is not None and orig_w != resolution[0]:
        cx = cx * resolution[0] / orig_w
    if cy is not None and orig_h != resolution[1]:
        cy = cy * resolution[1] / orig_h

    return DepthCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, depth=resized_image_depth,
                  cx=cx, cy=cy)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry