import torch
from tqdm import tqdm
from random import randint

from drema.drema_scene.interactive_gaussian_model import InteractiveGaussianModel
from drema.gaussian_renderer.depth_gaussian_renderer import render_depth
from drema.gaussian_renderer.original_gaussian_renderer import render
from drema.gaussian_splatting_utils.loss_utils import l1_loss, ssim
from drema.gaussian_splatting_utils.mesh_utils import GaussianExtractorDepth
from drema.scene import Scene
from drema.drema_scene import DremaScene



class BaseTrainer:

    def __init__(self, dataset, opt, pipe, saving_iterations):
        self.dataset = dataset
        self.opt = opt #BN: mesh optimization options. We have this since mesh extraction also done here
        self.pipe = pipe
        self.saving_iterations = saving_iterations
        #self.checkpoint_iterations = checkpoint_iterations

        self.scene = self.create_scene(dataset)
        self.gaussians = self.scene.gaussians
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.gaussians_to_save = None

    def create_scene(self, dataset):
        return DremaScene(dataset, InteractiveGaussianModel(dataset.sh_degree))

    def step(self, iteration):

        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background


        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        return loss, Ll1, viewspace_point_tensor, visibility_filter, radii, render_pkg

    def train(self):
        self.viewpoint_stack = None
        ema_loss_for_log = 0.0
        first_iter = 1
        progress_bar = tqdm(range(first_iter, self.opt.iterations + 1), desc="Training progress")

        for iteration in range(first_iter, self.opt.iterations + 1):

            loss, Ll1, viewspace_point_tensor, visibility_filter, radii, render_pkg = self.step(iteration)

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # Log and save
                #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                #                testing_iterations, scene, render, (pipe, background))


                # Densification
                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent,
                                                    size_threshold)

                    if iteration % self.opt.opacity_reset_interval == 0 or (
                            self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                #if (iteration in self.checkpoint_iterations):
                #    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #    torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                if iteration == self.saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.gaussians_to_save = self.gaussians.clone()

    #BN: called seperatly from AssetManager's extract_assets()
    def extract_mesh(self):
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        gaussExtractor = GaussianExtractorDepth(self.gaussians, render_depth, self.pipe, bg_color=bg_color)
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(self.scene.getTrainCameras())
        depth_trunc = (gaussExtractor.radius * 2.0) if self.opt.depth_trunc < 0 else self.opt.depth_trunc
        voxel_size = (depth_trunc / self.opt.mesh_res) if self.opt.voxel_size < 0 else self.opt.voxel_size
        sdf_trunc = 5.0 * voxel_size if self.opt.sdf_trunc < 0 else self.opt.sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        return mesh