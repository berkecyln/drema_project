import torch
import numpy as np

from drema.drema_scene.interactive_gaussian_model import InteractiveGaussianModel
from drema.gaussian_renderer.depth_gaussian_renderer import render_depth
from drema.gaussian_renderer.surf_gaussian_renderer import render_surf
from drema.gaussian_splatting_utils.loss_utils import gaussian


class GaussianWrapper:
    def __init__(self, pipe_args, sh_degree):
        self.pipe_args = pipe_args
        self.sh_degree = sh_degree
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.gs = InteractiveGaussianModel(sh_degree)
        self.labels = torch.zeros(0, dtype=torch.int32, device="cuda")
        self.max_label = 0

        self.initial_gs = None


    def add_environment_gaussians(self, environment_path):
        """
        Add environment gaussians to the scene
        :param env: path to the environment asset
        :return: label of the environment
        """

        # load the environment gaussians
        gs = InteractiveGaussianModel(self.sh_degree)
        gs.load_ply(environment_path)

        # add the environment gaussians to the scene
        self.gs.add(gs)

        # set the labels
        self.labels = torch.cat((self.labels, torch.zeros(gs.get_xyz.shape[0], dtype=torch.int32, device="cuda")))

        return 0


    def add_object_guassians(self, obj_path, translation=None):
        """
        Add object guassians to the scene
        :param obj: path to the object asset
        :return: label of the object
        """

        # increase the label counter
        self.max_label += 1

        # load the object gaussians
        gs = InteractiveGaussianModel(self.sh_degree)
        gs.load_ply(obj_path)
        if translation is not None:
            gs.translate(translation)

        # add the object gaussians to the scene
        self.gs.add(gs)

        # set the labels
        self.labels = torch.cat((self.labels, torch.ones(gs.get_xyz.shape[0], dtype=torch.int32, device="cuda") * self.max_label))

        return self.max_label

    def save_state_as_initial(self):
        """
        Save the current state as the initial state
        :return:
        """
        self.initial_gs = self.gs.clone()


    def reset_gaussians(self):
        """
        Reset the gaussian scene
        :return:
        """
        self.gs = self.initial_gs.clone()


    def filter_closest_gaussians(self, positions):
        """
        Filter the closest gaussians to the given positions
        :return:
        """
        pass


    def filter_object_gaussians(self):
        """
        Filter the environment gaussians
        :return:
        """

        #TODO: CHECK THIS FUNCTION
        objects_mask = self.labels > 0

        # If there are no object gaussians, return
        if objects_mask.sum() == 0:
            return  

        # compute the knn between objects and environment
        dists, indices = self.gs.get_close_gaussians(self.gs.get_xyz[objects_mask], 20)
        dists = dists.flatten()
        indices = indices.flatten()

        # retrieve the labels of the closest gaussians
        labels = self.labels[indices]

        close = indices[dists < 0.01]
        labels = labels[dists < 0.01]

        # change the lables of the close gaussians
        self.labels[close] = labels

    def filter_robot_gaussians(self, robot_labels):
        """
        Filter the robot gaussians
        :return:
        """
        robot_mask = torch.isin(self.labels, robot_labels)
        environment_mask = self.labels == 0
        dists, indices = self.gs.get_close_gaussians(self.gs.get_xyz[robot_mask], 100)

        dists = torch.min(dists, dim=1).values

        gaussians_to_keep = torch.ones(self.gs.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        gaussians_to_keep[dists < 0.02] = False
        gaussians_to_keep[~environment_mask] = True

        # filter the gaussians
        self.gs.filter_by_mask(gaussians_to_keep)
        # filter the labels
        self.labels = self.labels[gaussians_to_keep]




    def render_gaussians(self, camera):
        """
        Render the gaussian scene from a given camera
        :return: rendered image and depth
        """
        with torch.no_grad():
            # if the gaussians are 2D, call the render_surf function for the depth
            if self.gs.get_scaling.shape[1] == 2:
                rendered_pkg = render_surf(camera, self.gs, self.pipe_args, self.background)
                depth = rendered_pkg["surf_depth"][0]

            # if the gaussians are 3D, call the render_depth function
            else:
                rendered_pkg = render_depth(camera, self.gs, self.pipe_args, self.background)
                depth = rendered_pkg["depth_map"]

            rendered_image = rendered_pkg["render"]
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            display_image = rendered_image.permute(1, 2, 0).cpu().numpy()
            depth = depth.cpu().numpy()

        return display_image, depth

    def rototranslate_object(self, obj_mask, current_position, current_rotation, new_position, new_rotation):
        """
        Rototranslate the object in the scene
        :param obj_mask: mask of the object to rotate and translate
        :param new_rotation: new rotation matrix
        :param current_rotation: current rotation matrix
        :param new_position: new position
        :param current_position: current position
        :return:
        """
        self.gs.translate(-current_position.cuda(), obj_mask)
        self.gs.rotate_abs(current_rotation.cuda(), obj_mask)
        self.gs.rotate_abs(new_rotation.cuda(), obj_mask)
        self.gs.translate(new_position.cuda(), obj_mask)

    def rototranslate_link(self, obj_mask, difference_rotation, difference_translation, previous_panda_link_gs_positions):
        """
        Rototranslate the articulated object in the scene
        :param obj_mask: mask of the object to rotate and translate
        :param difference_rotation: difference in rotation
        :param difference_translation: difference in translation
        :param previous_panda_link_gs_positions: previous positions of the panda links
        :return:
        """
        self.rotate(obj_mask, difference_rotation, previous_panda_link_gs_positions)
        self.translate(obj_mask, difference_translation)

    def rotate(self, mask, rotation, center):
        """
        Rotate the scene
        :param mask: mask of the gaussians to rotate
        :param rotation: rotation matrix
        :param center: center of the rotation
        :return:
        """
        self.gs._xyz[mask] = self.gs.get_xyz[mask] - center
        self.gs._xyz[mask] = torch.matmul(rotation, self.gs._xyz[mask].transpose(0, 1)).transpose(0, 1)
        self.gs._xyz[mask] = self.gs.get_xyz[mask] + center

        self.gs.rotate_covariance(rotation, mask=mask)

    def translate(self, mask, translation):
        """
        Translate the scene
        :param mask: mask of the gaussians to translate
        :param translation: translation vector
        :return:
        """
        self.gs._xyz[mask] += translation.float().cuda()

    def filter_outside_workspace(self, workspace, apply_to_robot=False, robot_mask=None):
        """
        Filter out the gaussians outside the workspace
        :param workspace: workspace encoded as a list of 6 values [x_min, x_max, y_min, y_max, z_min, z_max]
        :return:
        """
        centers = self.gs.get_xyz
        # filter the gaussians outside the workspace
        filtered_centers = np.where((centers[:, 0] > workspace[0]) & (centers[:, 0] < workspace[1]) &
                                   (centers[:, 1] > workspace[2]) & (centers[:, 1] < workspace[3]) &
                                   (centers[:, 2] > workspace[4]) & (centers[:, 2] < workspace[5]))[0]

        # if the robot is not considered, change the filtered mask using the robot mask
        if not apply_to_robot:
            assert robot_mask is not None, "If apply_to_robot is False, robot_mask must be set"
            filtered_centers = filtered_centers | robot_mask

        self.gs.filter_by_mask(filtered_centers)

    def get_labels(self):
        """
        Get the labels of the gaussians
        :return: torch tensor
        """
        return self.labels

    def __len__(self):
        return self.gs.get_xyz.shape[0]