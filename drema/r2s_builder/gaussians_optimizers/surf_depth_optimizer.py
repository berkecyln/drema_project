from random import randint
import matplotlib.pyplot as plt


from drema.drema_scene import DremaScene
from drema.drema_scene.surface_interactive_gaussian_model import SurfInteractiveGaussianModel
from drema.gaussian_renderer.surf_gaussian_renderer import render_surf
from drema.gaussian_splatting_utils.loss_utils import l1_loss, ssim
from drema.gaussian_splatting_utils.mesh_utils import GaussianExtractor
from drema.r2s_builder.gaussians_optimizers.base_optimizer import BaseTrainer


class SurfDepthTrainer(BaseTrainer):

    def __init__(self, dataset, opt, pipe, saving_iterations, experiment_name="Exp_default"):
        super().__init__(dataset, opt, pipe, saving_iterations, experiment_name)

    def create_scene(self, dataset):
        return DremaScene(dataset, SurfInteractiveGaussianModel(dataset.sh_degree))

    def step(self, iteration):
        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        self.viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        render_pkg = render_surf(self.viewpoint_cam, self.gaussians, self.pipe, self.background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = self.viewpoint_cam.original_image.cuda()
        gt_depth = self.viewpoint_cam.depth.clone().cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        lambda_normal = self.opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = self.opt.lambda_dist if iteration > 3000 else 0.0 #BN: reason of huge spike at 3000 since we exclude dist loss before

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        depth = render_pkg['surf_depth']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        depth_gt_cloned = gt_depth.clone()
        depth_gt_cloned[depth_gt_cloned == 0] = depth[0, depth_gt_cloned == 0]
        depth_loss = self.opt.lambda_depth * l1_loss(depth, depth_gt_cloned.unsqueeze(0))

        # loss
        total_loss = loss + dist_loss + normal_loss + depth_loss

        total_loss.backward()

        return total_loss, Ll1, viewspace_point_tensor, visibility_filter, radii, render_pkg


    def extract_mesh(self):
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        gaussExtractor = GaussianExtractor(self.gaussians, render_surf, self.pipe, bg_color=bg_color)
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(self.scene.getTrainCameras())
        depth_trunc = (gaussExtractor.radius * 2.0) if self.opt.depth_trunc < 0 else self.opt.depth_trunc
        voxel_size = (depth_trunc / self.opt.mesh_res) if self.opt.voxel_size < 0 else self.opt.voxel_size
        sdf_trunc = 5.0 * voxel_size if self.opt.sdf_trunc < 0 else self.opt.sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        return mesh