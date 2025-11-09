import torch
from random import randint

from drema.gaussian_renderer.depth_gaussian_renderer import render_depth
from drema.gaussian_splatting_utils.loss_utils import l1_loss, ssim
from drema.r2s_builder.gaussians_optimizers.base_optimizer import BaseTrainer


class DepthTrainer(BaseTrainer):

    def __init__(self, dataset, opt, pipe, saving_iterations, experiment_name="Exp_default"):
        super().__init__(dataset, opt, pipe, saving_iterations, experiment_name)

    def step(self, iteration):
        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        #BN: diffrence from base optimizer depth renderer used
        render_pkg = render_depth(viewpoint_cam, self.gaussians, self.pipe, self.background)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth_map"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.depth.clone().cuda()

        Ll1 = l1_loss(image, gt_image)

        # Create a mask where gt_image is not equal to 0
        mask = gt_depth != 0

        # Apply the mask to the images
        masked_depth = depth[mask]
        masked_gt_depth = gt_depth[mask]

        # Compute the L1 loss
        if masked_depth.shape[0] == 0:
            depth_loss = torch.tensor(0.0, device="cuda")
        else:
            depth_loss = self.opt.lambda_depth * l1_loss(masked_depth, masked_gt_depth.unsqueeze(0))

        #BN: difference from base optimizer add depth loss to total loss
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_loss

        loss.backward()

        return loss, Ll1, viewspace_point_tensor, visibility_filter, radii, render_pkg


