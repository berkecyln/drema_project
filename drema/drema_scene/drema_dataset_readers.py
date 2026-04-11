import os
from typing import NamedTuple
import open3d as o3d

import numpy as np
import cv2
import matplotlib.pyplot as plt

from drema.gaussian_splatting_utils.graphics_utils import fov2focal, BasicPointCloud
from drema.scene.dataset_readers import CameraInfo, sceneLoadTypeCallbacks, SceneInfo, getNerfppNorm, readColmapCameras
from drema.utils.drema_camera_utils import read_txt_intrinsics, read_txt_extrinsic
from drema.utils.point_cloud_utils import project_depth


class CameraInfoDepth(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_path: str
    depth_image: np.array
    cx: float = None
    cy: float = None


def readTxtSceneInfo(path, images, eval, depth_folder):

    print(path)
    cam_extrinsics = read_txt_extrinsic(path)
    cam_intrinsics = read_txt_intrinsics(path)


    reading_dir = "images" if images == None else images
    depth_dir = 'depth' if depth_folder == None else depth_folder
    #  depth_folder=os.path.join(path, depth_dir)
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))

    # add depth images
    cam_infos_depth_unsorted = []
    for cam_info in cam_infos_unsorted:
        depth_path = os.path.join(path, depth_dir, cam_info.image_name.split(".")[0] + ".npy")
        depth_image = np.load(depth_path)

        cam_infos_depth_unsorted.append(CameraInfoDepth(uid=cam_info.uid, image_name=cam_info.image_name,
                                                        image=cam_info.image, image_path=cam_info.image_path,
                                                        depth_path=depth_path, depth_image=depth_image, R=cam_info.R,
                                                        T=cam_info.T, FovX=cam_info.FovX, FovY=cam_info.FovY,
                                                        width=cam_info.width, height=cam_info.height,
                                                        cx=cam_info.cx, cy=cam_info.cy))

    cam_infos = sorted(cam_infos_depth_unsorted.copy(), key=lambda x : x.image_name)
    train_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")

    pcd = fetchTxtPly(cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def fetchTxtPly(cameras):
    scene_points = []
    scene_colors = []
    for camera in cameras:
        fy = fov2focal(camera.FovY, camera.height)
        fx = fov2focal(camera.FovX, camera.width)

        image = camera.image
        depth = camera.depth_image

        # Use actual principal point (cx, cy) from intrinsics if available,
        # otherwise fall back to image center
        ppx = camera.cx if camera.cx is not None else camera.width / 2
        ppy = camera.cy if camera.cy is not None else camera.height / 2
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])

        points = project_depth(depth, K)
        points = points[(depth > 0).reshape(-1), :]
        colors = np.array(image)
        colors = colors.reshape(-1, 3)
        colors = colors[(depth > 0).reshape(-1), :]
        #colors = colors.reshape(-1, 3)
        colors = colors / 255.0

        # apply extrinsics
        R = camera.R
        #T = camera.T

        # inverted_translation = -np.dot(inverted_rotation, give_translation)
        T = -np.dot(R, camera.T)

        points = np.dot(R, points.T).T + T

        #sample random points
        idx = np.random.choice(points.shape[0], min(500, len(points)), replace=False)

        scene_points.append(points[idx])
        scene_colors.append(colors[idx])

    return BasicPointCloud(points=np.vstack(scene_points), colors=np.vstack(scene_colors), normals=np.zeros((np.vstack(scene_points).shape[0], 3)))

sceneLoadTypeCallbacks["txt"] = readTxtSceneInfo