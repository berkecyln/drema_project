import os

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R

from drema.scene import colmap_loader
from drema.utils.point_cloud_utils import project_depth

###### READ AND WRITE CAMERA PARAMETERS ######

def read_pose_file(path, separator=" "):
    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    intrinsics = np.eye(3)
    with open(path, "r") as file:

        lines = file.read().split("\n")

        rotation[0] = np.array(lines[0].split(separator)[:3]).astype(float)
        rotation[1] = np.array(lines[1].split(separator)[:3]).astype(float)
        rotation[2] = np.array(lines[2].split(separator)[:3]).astype(float)
        translation[0] = float(lines[0].split(separator)[3])
        translation[1] = float(lines[1].split(separator)[3])
        translation[2] = float(lines[2].split(separator)[3])

        intrinsics[0,0] = float(lines[5].split(separator)[0])
        intrinsics[0,2] = float(lines[5].split(separator)[2])
        intrinsics[1,1] = float(lines[6].split(separator)[1])
        intrinsics[1,2] = float(lines[6].split(separator)[2])

    return rotation, translation, intrinsics


def write_pose_file(path,extrinsic, intrinsics, separator=" "):
    with open(path, "w") as f:
        # write rotation and translation to file
        for row in extrinsic:
            for value in row:
                f.write(str(value) + separator)
            f.write("\n")
        f.write("\n")
        for row in intrinsics:
            for value in row:
                f.write(str(value) + separator)
            f.write("\n")

def read_txt_extrinsic(path):

    cam_extrinsics = {}
    files = os.path.join(path, "poses")
    extrinsics_files = [f for f in os.listdir(files) if os.path.isfile(os.path.join(files, f))]
    extrinsics_files = [f for f in extrinsics_files if len(f) == 8]
    extrinsics_files.sort()
    for k, file in enumerate(extrinsics_files):
        give_rotation, give_translation, given_intrinsics = read_pose_file(os.path.join(files, file), separator=" ")
        inverted_rotation = R.from_matrix(give_rotation).inv().as_matrix()
        qvec = R.from_matrix(inverted_rotation).as_quat()
        inverted_translation = -np.dot(inverted_rotation, give_translation)
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])

        image_name = file.split(".")[0] + ".png"
        cam_extrinsics[k] = colmap_loader.Image(
            id=k, qvec=qvec, tvec=inverted_translation,
            camera_id=1, name=image_name,
            xys=[], point3D_ids=[])

    return cam_extrinsics


def read_txt_intrinsics(path):

        cameras = {}
        files = os.path.join(path, "poses")
        extrinsics_files = [f for f in os.listdir(files) if os.path.isfile(os.path.join(files, f))]
        extrinsics_files = [f for f in extrinsics_files if len(f) == 8]
        extrinsics_files.sort()
        give_rotation, give_translation, given_intrinsics = read_pose_file(os.path.join(files, extrinsics_files[0]), separator=" ")

        camera_id = 1
        model_id = 1
        model_name = colmap_loader.CAMERA_MODEL_IDS[model_id].model_name
        params = np.array([given_intrinsics[0,0], given_intrinsics[1,1], given_intrinsics[0,2], given_intrinsics[1,2]])

        # Get actual image dimensions from the images folder instead of cx*2/cy*2
        # (the principal point may not be at the image center for real cameras)
        images_dir = os.path.join(path, "images")
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        first_image = Image.open(os.path.join(images_dir, image_files[0]))
        width, height = first_image.size

        cameras[camera_id] = colmap_loader.Camera(id=camera_id,
                                    model=model_name,
                                    width=width,
                                    height=height,
                                    params=np.array(params))

        return cameras


def filter_depth(depth, intrinsics):
    # filter the depth
    point_camera_frame = project_depth(depth, intrinsics)
    pc_gs = o3d.geometry.PointCloud()
    pc_gs.points = o3d.utility.Vector3dVector(point_camera_frame)
    cl, ind = pc_gs.remove_radius_outlier(nb_points=16, radius=0.05)

    # get removed indices from ind
    ind_removed = np.where(np.isin(np.arange(len(point_camera_frame)), ind) == False)[0]

    # create point clouds for the removed points
    #removed_points = point_camera_frame[ind_removed]
    #pc_removed = o3d.geometry.PointCloud()
    #pc_removed.points = o3d.utility.Vector3dVector(removed_points)

    # color the original points as green
    #pc_gs.paint_uniform_color([0, 1, 0])  # green
    # color the removed points as red
    #pc_removed.paint_uniform_color([1, 0, 0])  # red

    # visualize the point clouds
    #o3d.visualization.draw_geometries([pc_gs, pc_removed])

    point_camera_frame = point_camera_frame[ind]


    return point_camera_frame, ind_removed

def depth_to_rlbench(depth, intrinsics_gs, extrinsics_gs, intrinsics_sim, extrinsics_sim, scale, rotation, translation, filter=True):
    # Project depth to 3D and change to simulation frame
    if filter:
        point_camera_frame, _ = filter_depth(depth, intrinsics_gs)
    else:
        point_camera_frame = project_depth(depth, intrinsics_gs)

    point_world_gs = np.dot(extrinsics_gs[:3, :3], point_camera_frame.transpose()).transpose() + extrinsics_gs[:3, 3]
    point_world_sim = scale * np.dot(rotation, point_world_gs.transpose()).transpose() + translation

    # Project the 3D points to the camera frame
    inverted_extrinsics_sim = np.eye(4)
    inverted_extrinsics_sim[:3, :3] = extrinsics_sim[:3, :3].transpose()
    inverted_extrinsics_sim[:3, 3] = - inverted_extrinsics_sim[:3, :3] @ extrinsics_sim[:3, 3]

    point_sim_camera_frame = np.dot(inverted_extrinsics_sim[:3, :3],
                                    point_world_sim.transpose()).transpose() + inverted_extrinsics_sim[:3, 3]
    image = np.zeros(depth.shape)
    x = (point_sim_camera_frame[:, 0] / point_sim_camera_frame[:, 2] * intrinsics_sim[0, 0] + intrinsics_sim[0, 2])
    y = (point_sim_camera_frame[:, 1] / point_sim_camera_frame[:, 2] * intrinsics_sim[1, 1] + intrinsics_sim[1, 2])
    x = x.astype(int)
    y = y.astype(int)
    mask = (0 <= x) & (x < image.shape[1]) & (0 <= y) & (y < image.shape[0])
    image[y[mask], x[mask]] = point_sim_camera_frame[mask, 2]

    return image

