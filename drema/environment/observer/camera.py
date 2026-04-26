import os

import numpy as np
import torch
import cv2

from drema.gaussian_splatting_utils.graphics_utils import focal2fov
from drema.scene.cameras import Camera as GsCamera
from drema.scene.colmap_loader import qvec2rotmat
from drema.utils.drema_camera_utils import read_txt_extrinsic, read_txt_intrinsics


class CameraManager:
    def __init__(self, visualization_camera_index=0):
        self.simulation_cameras = {}
        self.visualization_cameras = {}
        self.visualization_cameras_keys = []
        self.visualization_camera_index = 0

    def translate_visualization_camera(self, translation):
        """ Translates the visualization camera """
        self.get_visualization_camera().translate_camera(translation)

    def rotate_visualization_camera(self, yaw=0, pitch=0):
        """ Rotates the visualization camera using yaw and pitch in degrees """
        self.get_visualization_camera().rotate_camera(yaw, pitch)

    def get_forward(self):
        """ Returns the forward vector of the camera """

        return self.get_visualization_camera().rotation[:, 2]

    def get_right(self):
        """ Returns the right vector of the camera """

        return self.get_visualization_camera().rotation[:, 0]

    def get_up(self):
        """ Returns the up vector of the camera """

        return self.get_visualization_camera().rotation[:, 1]

    def update_visualization_camera_extrinsics(self, rotation, translation):
        """
        Update the camera extrinsics given a name and the new extrinsics parameters
        :param rotation: new rotation encoded in a 3x3 matrix
        :param translation: new translation encoded in a 3x1 vector
        :return: None
        """

        self.get_visualization_camera().update_extrinsics(rotation, translation)


    def get_next_visualization_camera(self):
        """
        Get the next visualization camera
        :return: None
        """
        self.visualization_camera_index = (self.visualization_camera_index + 1) % len(self.visualization_cameras_keys)

    def get_previous_visualization_camera(self):
        """
        Get the previous visualization camera
        :return: None
        """
        self.visualization_camera_index = (self.visualization_camera_index - 1) % len(self.visualization_cameras_keys)

    def get_visualization_camera(self):
        """
        Get the visualization camera given the index
        :return: visualization camera
        """
        return self.visualization_cameras[self.visualization_cameras_keys[self.visualization_camera_index]]

    def get_simulation_cameras(self):
        """
        Get the simulation cameras
        :return: simulation cameras
        """
        return self.simulation_cameras.keys(), self.simulation_cameras.values()

    def update_camera_extrinsics(self, name, rotation, translation):
        """
        Update the camera extrinsics given a name and the new extrinsics parameters
        :param name: camera name
        :param rotation: new rotation encoded in a 3x3 matrix
        :param translation: new translation encoded in a 3x1 vector
        :return: None
        """

        if name in self.simulation_cameras.keys():
            self.simulation_cameras[name].update_extrinsics(rotation, translation)

        if name in self.visualization_cameras.keys():
            self.visualization_cameras[name].update_extrinsics(rotation, translation)

    def add_simulation_camera(self, id, scale, name, rotation, translation, intrinsics, width, height, near=0, far=10, crop_top=0):
        """
        Add a new camera to the simulation
        :param id: camera id
        :param scale: camera scale
        :param name: camera name
        :param rotation: camera rotation encoded in a 3x3 matrix
        :param translation: camera translation encoded in a 3x1 vector
        :param intrinsics: camera intrinsics encoded in a 3x3 matrix
        :param width: camera width
        :param height: camera height
        :param crop_top: rows to remove from top of rendered image after subsampling
        :return: None
        """
        camera = CameraWrapper(id, scale, name, rotation, translation, intrinsics, width, height, near, far, crop_top=crop_top)
        self.simulation_cameras[name] = camera

    def add_visualization_camera(self, id, scale, name, rotation, translation, intrinsics, width, height):
        """
        Add a new camera to the visualization
        :param id: camera id
        :param scale: camera scale
        :param name: camera name
        :param rotation: camera rotation encoded in a 3x3 matrix
        :param translation: camera translation encoded in a 3x1 vector
        :param intrinsics: camera intrinsics encoded in a 3x3 matrix
        :param width: camera width
        :param height: camera height
        :return:
        """
        camera = CameraWrapper(id, scale, name, rotation, translation, intrinsics, width, height)
        self.visualization_cameras[name] = camera
        self.visualization_cameras_keys.append(name)

    def load_cameras_from_trajectory(self, trajectory, scale=2, simulation=True, visualization=False):
        """
        Load the cameras given a dictionary that wraps demonstration data
        :param trajectory:
        :param scale:
        :param simulation:
        :param visualization:
        :return: None
        """

        assert simulation or visualization, "At least one of the flags simulation or visualization must be True"

        camera_params = trajectory.get_cameras()

        # iterate the dictionary
        for k, camera_key in enumerate(camera_params):
            params = camera_params[camera_key]

            name = params[0]
            extrinsics = params[1]
            intrinsics = params[2]
            far = params[3]
            near = params[4]
            stored_w = params[5] if len(params) > 5 else None
            stored_h = params[6] if len(params) > 6 else None
            crop_top = params[7] if len(params) > 7 else 0

            rotation = extrinsics[:3, :3]
            translation = extrinsics[:3, 3]

            translation = - np.dot(np.transpose(rotation), translation)

            # increases the dimensions of the intrinsics
            intrinsics[:2, :] *= scale

            # use stored image dimensions; fall back to cx*2/cy*2 for legacy data
            if stored_w is not None and stored_h is not None:
                width = int(stored_w * scale)
                # render at pre-crop height so crop_top rows can be removed after rendering
                height = int((stored_h + crop_top) * scale)
            else:
                height = int(intrinsics[1, 2] * 2)
                width = int(intrinsics[0, 2] * 2)
                crop_top = 0

            if simulation:
                self.add_simulation_camera(k, scale, name, rotation, translation, intrinsics, width, height, near, far, crop_top=crop_top)

            if visualization:
                self.add_visualization_camera(k, scale, name, rotation, translation, intrinsics, width, height)

    def load_cameras_from_directory(self, path, scale=1, image_dir="images", simulation=False, visualization=True):
        camera_extrinsics = read_txt_extrinsic(path)
        camera_intrinsics = read_txt_intrinsics(path)

        for idx, value in enumerate(camera_extrinsics):
            extrinsics = camera_extrinsics[value]
            intrinsics_params = camera_intrinsics[extrinsics.camera_id].params

            # create intrinsics matrix
            intrinsics = np.zeros((3, 3))
            intrinsics[0, 0] = intrinsics_params[0]
            intrinsics[1, 1] = intrinsics_params[1]
            intrinsics[0, 2] = intrinsics_params[2]
            intrinsics[1, 2] = intrinsics_params[3]

            # get the rotation and translation
            rotation = np.transpose(qvec2rotmat(extrinsics.qvec))
            translation = np.array(extrinsics.tvec)

            # get name from path
            image_path = os.path.join(path, image_dir, os.path.basename(extrinsics.name))
            name = os.path.basename(image_path).split(".")[0]

            # get the image
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # increases the dimensions of the image
            intrinsics[:2, :] *= scale
            height *= scale
            width *= scale

            if simulation:
                self.add_simulation_camera(idx, scale, name, rotation, translation, intrinsics, width, height, near=-1, far=-1)

            if visualization:
                self.add_visualization_camera(idx, scale, name, rotation, translation, intrinsics, width, height)


class CameraWrapper:
    def __init__(self, id, scale, name, rotation, translation, intrinsics, width, height, near=0, far=10, crop_top=0):
        self.id = id
        self.scale = scale
        self.name = name
        self.rotation = rotation
        self.translation = translation
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.crop_top = crop_top  # rows removed from top after subsampling (matches real camera finger crop)

        # initialize the parameters
        self.FovX = 0
        self.FovY = 0
        self.image = torch.zeros((3, int(height), int(width)), dtype=torch.float32)

        # compute the FoV
        self.compute_FoV()

        # initialize the view object from GS
        self.view = GsCamera(0, self.rotation, self.translation, self.FovX, self.FovY, self.image,
                             None, self.name, self.id, trans=np.array([0.0, 0.0, 0.0]), scale=1,
                              data_device="cpu")

    def update(self, rotation, translation, intrinsics, width, height):
        """
        Update the camera parameters and the view
        :param rotation: new rotation encoded in a 3x3 matrix
        :param translation: new translation encoded in a 3x1 vector
        :param intrinsics: new intrinsics encoded in a 3x3 matrix
        :param width: new width
        :param height: new height
        :return:
        """

        self.rotation = rotation
        self.translation = translation
        self.intrinsics = intrinsics
        self.width = width
        self.height = height

        self.view = GsCamera(0, self.rotation, self.translation, self.FovX, self.FovY, self.image,
                             None, self.name, self.id, trans=np.array([0.0, 0.0, 0.0]), scale=1,
                              data_device="cpu")

    def update_extrinsics(self, rotation, translation):
        """
        Update the camera extrinsics
        :param rotation: new rotation encoded in a 3x3 matrix
        :param translation: new translation encoded in a 3x1 vector
        :return: None
        """
        self.rotation = rotation
        self.translation = -rotation.transpose() @ translation

        self.view = GsCamera(0, self.rotation, self.translation, self.FovX, self.FovY, self.image,
                             None, self.name, self.id, trans=np.array([0.0, 0.0, 0.0]), scale=1,
                              data_device="cpu")

    def compute_FoV(self):
        """
        Compute the field of view of the camera
        :return: None
        """
        self.FovX = focal2fov(self.intrinsics[0, 0], self.width)
        self.FovY = focal2fov(self.intrinsics[1, 1], self.height)

    def get_view(self):
        """
        Get the camera view
        :return: camera view
        """
        return self.view

    def rotate_camera(self, yaw=0, pitch=0):
        """
        Rotates the camera using yaw and pitch in degrees
        :param yaw: yaw angle in degrees
        :param pitch: pitch angle in degrees
        """
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        # Rotation matrices
        R_yaw = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])

        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])

        # Apply rotations
        self.rotation = R_yaw @ self.rotation  # Apply yaw
        self.rotation = self.rotation @ R_pitch  # Apply pitch

        # Normalize rotation matrix to prevent drift
        U, _, Vt = np.linalg.svd(self.rotation)  # Ensure it's still a valid rotation matrix
        self.rotation = U @ Vt
        
        # Update the view
        self.view = GsCamera(0, self.rotation, self.translation, self.FovX, self.FovY, self.image,
                             None, self.name, self.id, trans=np.array([0.0, 0.0, 0.0]), scale=1,
                             data_device="cpu")

    def translate_camera(self, translation):
        """
        Translates the camera
        :param translation: translation vector
        """
        self.translation += -self.rotation.transpose() @ translation

        # Update the view
        self.view = GsCamera(0, self.rotation, self.translation, self.FovX, self.FovY, self.image,
                             None, self.name, self.id, trans=np.array([0.0, 0.0, 0.0]), scale=1,
                             data_device="cpu")

if __name__ == "__main__":

    # test
    camera_manager = CameraManager()

    # load cameras from a directory
    path = "/home/leonardo/workspace/git_repo/SimulationGaussianSplatting/data/place_shape_in_shape_sorter_episode0_start"
    camera_manager.load_cameras_from_directory(path, scale=1, image_dir="images", simulation=False, visualization=True)
