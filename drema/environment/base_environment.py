import numpy as np
import pybullet
import pybullet_data
import torch

from drema.environment.assets.gaussians import GaussianWrapper
from drema.environment.assets.object import Object, FlatSurface
from drema.utils.point_cloud_utils import filter_scharr, filter_radious_outlier


class Environment:
    def __init__(self, objects, flat_surface, pipe_args, sh_degrees=3, environment_path=None, visualize=False):
        # create the gaussian scene
        self.gs = GaussianWrapper(pipe_args, sh_degrees)

        # path to the environment gaussians
        self.environment_path = environment_path

        # create the flat surface object
        self.flat_surface = FlatSurface(flat_surface["urdf_path"], flat_surface["gaussians_path"],
                                        flat_surface["initial_position"], flat_surface["initial_orientation"])

        # for each object in the dictionary "objects"
        self.objects = {}
        for name, obj in objects.items():
            # create the simple object
            self.objects[name] = Object(name, obj["urdf_path"], obj["gaussians_path"], obj["initial_position"],
                                        obj["initial_orientation"], obj["collidable"], obj["fixed"], obj["mass"],
                                        obj["lateral_friction"], obj["spinning_friction"], obj["rolling_friction"])

        # pybullet client
        self.client = pybullet # pybullet client
        self.bullet_ids = []

        # set the visualization flag
        self.visualize = visualize # visualize the simulation

        # flag to know if the gaussians were updated
        self.gaussians_updated = True

    def build_environment(self):
        """
        Build the environment
        :return:
        """

        # create the simulation client
        self.set_physics_client()

        # load the flat surface
        self.flat_surface.load(self.client)

        # for each object in the dictionary "objects"
        for name, obj in self.objects.items():
            # load the object in the simulation environment
            bullet_id = obj.load(self.client)
            self.bullet_ids.append(bullet_id)

            # load the gaussians
            gaussians_obj_id = self.gs.add_object_guassians(obj.gaussians_path)
            obj.set_gaussians_labels(gaussians_obj_id)

        # load the environment gaussians
        if self.environment_path is not None:
            self.gs.add_environment_gaussians(self.environment_path)

            # filter the gaussians
            #BN: this function does not any effect and duplicate idea from create_simulation's assets_manager.filter_environment()
            self.gs.filter_object_gaussians() 

        # set the gaussians mask for each object
        labels = self.gs.get_labels()
        for obj in self.objects.values():
            obj.set_gaussians_mask(labels == obj.get_gaussians_labels())

        # save the initial state
        self.gs.save_state_as_initial()

        # set fixed objects
        for obj in self.objects.values():
            if obj.fixed:
                obj.set_fixed(self.client)

        # set not collidable objects
        for obj in self.objects.values():
            if not obj.collidable:
                obj.set_not_collidable(self.client, self.objects.values())

        # set gravity
        self.client.setGravity(0, 0, -9.81)

    def set_physics_client(self):
        """
        Set the physics client
        :return:
        """
        if self.visualize:
            self.client.connect(pybullet.GUI)
            self.client.resetDebugVisualizerCamera(3, 90, -30, [0.0, 2.0, -0.0])
        else:
            self.client.connect(pybullet.DIRECT)

        self.client.setTimeStep(1 / 240.)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

    def step(self):
        """
        Step the simulation
        :return:
        """
        self.client.stepSimulation()

        return 0


    def update_state(self):
        """
        Update the state of the environment
        :return:
        """
        for obj in self.objects.values():
            #prev_position, prev_orientation, current_position, current_orientation = obj.update(self.client)

            obj.update(self.client)
            self.gaussians_updated = False

            '''
            mask = obj.get_gaussians_mask()

            # update the positions and orientations
            new_positions = torch.tensor(current_position, dtype=torch.float32, device="cuda")
            current_positions = torch.tensor(prev_position, dtype=torch.float32, device="cuda")

            # update the orientations
            new_orientations = torch.tensor(current_orientation, dtype=torch.float32, device="cuda").t()
            current_orientations = torch.tensor(prev_orientation, dtype=torch.float32, device="cuda")

            # update the gaussians
            self.gs.rototranslate_object(mask, current_positions, current_orientations, new_positions, new_orientations)
            '''

    def reset(self):
        # flag to know if the gaussians were updated
        self.gaussians_updated = True

        # reset the objects
        for obj in self.objects.values():
            obj.reset(self.client)

        # reset the gaussians
        self.gs.reset_gaussians()

        # reset the flat surface
        self.flat_surface.reset(self.client)

    def observe_state(self):
        self.update_state()

        state = []
        for obj in self.objects.values():
            state.append(obj.get_state(self.client))
        return state

    def render_cameras(self, cameras, filter_depth=True, radius_filter=False, compress_depth=True, visualize=False, threshold=2):
        self.update_state()
        self.update_gaussians()

        # when the cameras are renderd, the guassians are updated
        rgbs = []
        depths = []

        for cam in cameras:
            rgb, depth = self.gs.render_gaussians(cam.get_view())

            if filter_depth:
                if radius_filter:
                    depth = filter_radious_outlier(depth, cam.intrinsics, visualize=visualize)
                else:
                    depth = filter_scharr(depth, cam.intrinsics, th=threshold, visualize=visualize)

            scale = cam.scale
            rgb = (rgb[::scale, ::scale] * 255).astype(np.uint8)
            depth = depth[::scale, ::scale]

            if compress_depth:
                near = cam.near
                far = cam.far

                if near >= 0 and far >= 0:

                    depth[depth < near] = near
                    depth[depth > far] = far
                    depth = (depth - near) / (far - near)


            rgbs.append(rgb)
            depths.append(depth)

        return rgbs, depths

    def update_gaussians(self):
        """
        Update the gaussians
        :return:
        """
        for obj in self.objects.values():
            current_position, current_orientation = obj.get_state(self.client)
            prev_position, prev_orientation = obj.get_previous_state(self.client)

            mask = obj.get_gaussians_mask()

            # update the positions and orientations
            new_positions = torch.tensor(current_position, dtype=torch.float32, device="cuda")
            current_positions = torch.tensor(prev_position, dtype=torch.float32, device="cuda")

            # update the orientations
            new_orientations = torch.tensor(current_orientation, dtype=torch.float32, device="cuda")
            current_orientations = torch.tensor(prev_orientation, dtype=torch.float32, device="cuda").t()

            # update the gaussians
            self.gs.rototranslate_object(mask, current_positions, current_orientations, new_positions, new_orientations)

            # update the previous state
            obj.update_previous_state()

        self.gaussians_updated = True

    def get_pybullet_camera(self, transform_matrix):
        """
        Get the pybullet camera. Note PyBullet uses OpenGL camera model, which is different from OpenCV.
        :return: rotation, translation
        """
        _, _, view_matrix, _, _, _, _, _, _, _, _, _ = self.client.getDebugVisualizerCamera()

        # Convert view matrix to 4x4 NumPy array
        view_matrix = np.array(view_matrix).reshape(4, 4).T  # Transpose to match row-major order


        view_matrix[:3, :3] = transform_matrix @ view_matrix[:3,:3]
        view_matrix[:3, 3] = transform_matrix @ view_matrix[:3, 3]

        R_opencv = view_matrix[:3,:3].T
        t_opencv = -R_opencv @ view_matrix[:3, 3]

        return R_opencv, t_opencv