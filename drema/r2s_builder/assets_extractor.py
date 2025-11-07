import copy
import os
import shutil
import cv2
import numpy as np
import scipy
import torch
import open3d as o3d

from PIL import Image

from drema.drema_scene.interactive_gaussian_model import InteractiveGaussianModel
from drema.r2s_builder.extractors.Urdf import URDFBuilder
from drema.utils.drema_camera_utils import read_pose_file
from drema.utils.point_cloud_utils import project_depth


class AssetsManager:
    def __init__(self, source_path, assets_path, optimizer, dataset, opt, pipe, gaussians_iterations, mesh_iterations):

        self.source_path = source_path
        self.assets_path = assets_path

        #BN: load data containers
        self.rgb_images = []
        self.depth_images = []
        self.camera_params = []
        self.masks_images = []

        self.poses_paths = []
        self.masks_files = []
        self.images_files = []
        self.depths_files = []
        self.names = []

        # filter gaussians data
        self.new_path = None
        self.box_max = np.array([-np.inf, -np.inf, -np.inf])
        self.box_min = np.array([np.inf, np.inf, np.inf])
        self.color_filter = 0.01
        self.kernel = np.ones((5, 5), np.uint8)

        # class to call to extract gaussians and meshes
        self.optimizer = optimizer

        # given parameters
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe

        # iterations
        self.gaussians_iterations = gaussians_iterations
        # if mesh iterations is less than gaussians iterations, change it to gaussians iterations and print a warning
        if mesh_iterations < gaussians_iterations:
            print("Warning: mesh_iterations should be greater than or equal to gaussians_iterations. "
                  "Changing mesh_iterations to gaussians_iterations")
            self.mesh_iterations = gaussians_iterations
        else:
            self.mesh_iterations = mesh_iterations

        # table
        self.table_mesh = None
        self.table_position = None
        self.table_normal = None
        self.delaunay = None
        self.hull_points = None

        # URDF
        self.urdf_builder = URDFBuilder(self.assets_path)

    def load_data(self):
        images_dir_path = os.path.join(self.source_path, "images")
        depth_dir_path = os.path.join(self.source_path, "depth_scaled")
        masks_dir_path = os.path.join(self.source_path, "object_mask")
        poses_dir_path = os.path.join(self.source_path, "poses") #BN: camera poses

        self.poses_paths = [f for f in os.listdir(poses_dir_path) if os.path.isfile(os.path.join(poses_dir_path, f))]
        self.poses_paths.sort()
        # load rgb images

        self.masks_files = [f for f in os.listdir(masks_dir_path) if os.path.isfile(os.path.join(masks_dir_path, f))]
        self.images_files = [f for f in os.listdir(images_dir_path) if os.path.isfile(os.path.join(images_dir_path, f))]
        self.depths_files = [f for f in os.listdir(depth_dir_path) if os.path.isfile(os.path.join(depth_dir_path, f))]

        self.masks_files.sort()
        self.images_files.sort()
        self.depths_files.sort()

        # read camera params
        camera_params = {}
        for pose in self.poses_paths:
            if "_near_far.txt" in pose:
                continue
            name = pose.split(".")[0]
            #BN: extract pose folder (it must be same as README because it is hard coded extraction)
            rotation, translation, intrinsics = read_pose_file(os.path.join(poses_dir_path, pose)) 
            camera_params[name] = (translation, rotation, intrinsics)

        for i, name in enumerate(self.masks_files):
            file_name = name.split(".")[0]
            self.names.append(file_name)

            mask = self.masks_files[i]
            image = self.images_files[self.images_files.index(file_name + ".png")]
            depth = self.depths_files[self.depths_files.index(file_name + ".npy")]

            mask_image = cv2.imread(os.path.join(masks_dir_path, mask), cv2.IMREAD_UNCHANGED)
            rgb_image = np.array(Image.open(os.path.join(images_dir_path, image)))
            depth_image = np.load(os.path.join(depth_dir_path, depth))

            self.masks_images.append(mask_image)
            self.rgb_images.append(rgb_image)
            self.depth_images.append(depth_image)
            self.camera_params.append(camera_params[file_name])

    def load_table(self):
        self.table_mesh = o3d.io.read_triangle_mesh(os.path.join(self.assets_path, "flat_surface", "flat_surface_mesh.obj"))

        # load center
        self.table_position = np.load(os.path.join(self.assets_path, "flat_surface", "position.npy"))

        # bring the table to the original position
        self.table_mesh = self.table_mesh.translate(self.table_position)

        # Get points from convex hull
        self.hull_points = np.asarray(self.table_mesh.vertices)

        # Create a Delaunay triangulation
        self.delaunay = scipy.spatial.Delaunay(self.hull_points)

    def extract_asset(self, id, extract_mesh=False, extract_urdf=False):
        # filter data
        self.filter_input_data(id)

        # set dataset source path to new path
        self.dataset.source_path = self.new_path

        # train gaussians
        gs_to_save, gs_to_mesh, trainer = self.train_gaussians() #BN: Wrapper around Trainer

        # restore source path
        shutil.rmtree(self.new_path)
        self.dataset.source_path = self.source_path

        # filter gaussians
        self.filter_gaussians(gs_to_save)
        self.filter_gaussians(gs_to_mesh)

        # save gaussians
        if len(gs_to_save.get_xyz) > 50:
            output_path_ply = os.path.join(self.assets_path, "objects_ply")
            os.makedirs(output_path_ply, exist_ok=True)
            gs_to_save.save_ply(os.path.join(output_path_ply, str(id) + ".ply"))

        if extract_mesh:
            mesh = trainer.extract_mesh()#BN: Trainer method
            self.filter_mesh(mesh)
            mesh_path = os.path.join(self.assets_path, "meshes")
            os.makedirs(mesh_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(mesh_path, str(id) + ".obj"), mesh)

            if extract_urdf:
                self.urdf_builder.build_urdf_object(mesh, str(id))

    def extract_environment(self, extract_mesh=False):
        self.dataset.source_path = self.source_path

        # train gaussians
        gs_to_save, gs_to_mesh, trainer = self.train_gaussians()

        if len(gs_to_save.get_xyz) > 50:
            output_path_ply = os.path.join(self.assets_path, "objects_ply")
            os.makedirs(output_path_ply, exist_ok=True)
            gs_to_save.save_ply(os.path.join(output_path_ply, "gaussians_before_removal.ply"))

        if extract_mesh: #BN: not used by default since this env will filtered
            mesh = trainer.extract_mesh()#BN: Trainer method
            mesh_path = os.path.join(self.assets_path, "meshes")
            os.makedirs(mesh_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(mesh_path, "environment.obj"), mesh)

    def train_gaussians(self, extract_mesh=False):
        trainer = self.optimizer(self.dataset, self.opt, self.pipe, self.gaussians_iterations)
        trainer.train()

        # get guassians to save and gaussians to mesh
        gaussians_to_save = trainer.gaussians_to_save
        gaussians_for_mesh = trainer.gaussians

        return gaussians_to_save, gaussians_for_mesh, trainer

    #BN: IMPORTANT table extraction does not depend on gaussian splat it creates
    #BN: the surface directy from depth images via piont cloud and ransac
    def extract_table(self, table_id):
        # read segmentation masks and depth maps to extract table points
        table_points = []
        for k, name in enumerate(self.names):
            mask = self.masks_images[k]
            depth_image = self.depth_images[k]

            gs_translation, gs_rotation, intrinsics = self.camera_params[k]
            #BN: create the point cloud data based on camera frame only for table mask
            points = project_depth(depth_image, intrinsics)[(mask == table_id).reshape(-1)]
            #BN: transform point cloud to world frame 
            points = np.dot(gs_rotation, points.T).T + gs_translation

            table_points.append(points)
        
        table_points = np.concatenate(table_points, axis=0)#BN: merge all world frame points
        assert table_points.shape[0] > 0, "No table points found."

        # downsample the points
        if table_points.shape[0] > 50000:
            table_points = table_points[np.random.choice(table_points.shape[0], 50000, replace=False), :]

        # create point cloud
        table = o3d.geometry.PointCloud()
        table.points = o3d.utility.Vector3dVector(table_points)

        # remove outliers
        table, _ = table.remove_radius_outlier(nb_points=16, radius=0.05)

        # compute plane from the point cloud
        plane_coordinates, plane_index_points = table.segment_plane(0.01, 3, 1000) #BN: uses ransac

        # create a point cloud of the plane
        plane_cloud = table.select_by_index(plane_index_points)#BN: filter plane ponts
        center = np.mean(np.array(plane_cloud.points), axis=0) #BN: find center of table
        plane_cloud = plane_cloud.translate(-center)#BN: move table so its in (0,0,0)

        # compute hull of the plane
        hull, _ = plane_cloud.compute_convex_hull()#BN: basically mesh creation
        #hull_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(hull)

        self.table_mesh = hull
        self.table_position = center
        self.table_normal = plane_coordinates

        self.urdf_builder.build_urdf_flat_surface(hull, center)

        # Get points from convex hull
        self.hull_points = np.asarray(self.table_mesh.vertices)#BN: get the vertices of table

        # move the table to the original position
        self.hull_points = self.hull_points + center #BN: put table old position

        # Create a Delaunay triangulation
        self.delaunay = scipy.spatial.Delaunay(self.hull_points)#BN:create fast lookup structure

    def filter_mesh(self, mesh):
        # Get points from the original point cloud
        pcd_points = np.asarray(mesh.vertices)

        # Find which points are inside the convex hull
        inside_mask = self.delaunay.find_simplex(pcd_points) >= 0

        # Find points below the table surface
        below_table_mask = pcd_points[:, 2] < np.max(self.hull_points[:, 2])

        mesh.remove_vertices_by_mask(inside_mask | below_table_mask)

    def filter_input_data(self, id):
        self.new_path = os.path.join(self.source_path, "filtered_" + str(id))
        print("Filtering data for object: ", id)
        print("New path: ", self.new_path)

        # initial folders
        self.box_max = np.array([-np.inf, -np.inf, -np.inf])
        self.box_min = np.array([np.inf, np.inf, np.inf])
        self.color_filter = np.inf

        # new folders
        new_images_path = os.path.join(self.new_path, "images")
        new_depth_path = os.path.join(self.new_path, "depth_scaled")
        new_masks_path = os.path.join(self.new_path, "object_mask")
        new_poses_path = os.path.join(self.new_path, "poses")

        # create new folders
        os.makedirs(self.new_path, exist_ok=True)
        os.makedirs(new_images_path, exist_ok=True)
        os.makedirs(new_depth_path, exist_ok=True)
        os.makedirs(new_masks_path, exist_ok=True)
        os.makedirs(new_poses_path, exist_ok=True)

        # create a copy of rgb, depth, mask and camera params
        rgb_images = copy.deepcopy(self.rgb_images) #self.rgb_images.copy()
        depth_images =  copy.deepcopy(self.depth_images)#self.depth_images.copy()
        masks_images =  copy.deepcopy(self.masks_images)#self.masks_images.copy()
        camera_params = copy.deepcopy(self.camera_params)#self.camera_params.copy()

        # filter data
        for k, name in enumerate(self.names):
            mask = masks_images[k]

            # create a filter mask
            filter_mask = np.zeros(mask.shape, dtype=np.uint8)
            filter_mask[mask != id] = 1
            filter_mask = cv2.dilate(filter_mask, self.kernel)
            filter_mask = filter_mask.astype(bool)

            # update bounding box params
            gs_translation, gs_rotation, intrinsics = camera_params[k]
            points = project_depth(depth_images[k], intrinsics)[~filter_mask.reshape(-1)]

            if len(points) > 0:
                points = np.dot(gs_rotation, points.T).T + gs_translation
                self.box_max = np.maximum(self.box_max, np.max(points, axis=0))
                self.box_min = np.minimum(self.box_min, np.min(points, axis=0))

            # save filtered images
            if len(rgb_images[k][~filter_mask]) > 0:

                # filter rgb and depth images
                rgb_images[k][filter_mask] = 0
                depth_images[k][filter_mask] = 0
                masks_images[k][filter_mask] = 0

                # update color filter
                if len(rgb_images[k][~filter_mask]) > 0:
                       self.color_filter = min(self.color_filter, np.min(np.linalg.norm(rgb_images[k][~filter_mask]/255, axis=1))*0.7)

                # save data
                cv2.imwrite(os.path.join(new_masks_path, name + ".png"), masks_images[k])
                Image.fromarray(rgb_images[k]).save(os.path.join(new_images_path, name + ".png"))
                np.save(os.path.join(new_depth_path, name + ".npy"), depth_images[k])
                shutil.copy(os.path.join(self.source_path, "poses", name + ".txt"), os.path.join(new_poses_path, name + ".txt"))

    @torch.no_grad()
    def filter_gaussians(self, gaussians):

        # create a mask from the centers of the gaussians and the bounding box
        min_coords = torch.tensor(self.box_min, device=gaussians.get_xyz.device)
        max_coords = torch.tensor(self.box_max, device=gaussians.get_xyz.device)
        mask_box = (gaussians.get_xyz > min_coords) & (gaussians.get_xyz < max_coords)
        mask_box = mask_box.all(dim=1)

        # filter gaussians outside the bounding box
        gaussians.filter_by_mask(mask_box)

        # use dbscan to detect clusters
        points_after_filter = gaussians.get_xyz.cpu().detach().numpy()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_after_filter)
        labels = np.array(point_cloud.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
        label_list, counts = np.unique(labels, return_counts=True)
        counts[label_list == -1] = 0 # do not count noise points
        main_cluster = label_list[np.argmax(counts)]
        gaussinas_to_keep = labels == main_cluster
        mask_cluster = np.zeros(len(labels), dtype=bool)
        mask_cluster[gaussinas_to_keep] = True

        # filter gaussians outside the cluster
        gaussians.filter_by_mask(mask_cluster)

        # filter gaussians by color
        gaussians.filter_by_color(self.color_filter)

    @torch.no_grad()
    def filter_environment(self):
        objcts_path = os.path.join(self.assets_path, "objects_ply")
        # load environment
        environment = InteractiveGaussianModel(self.dataset.sh_degree)
        environment.load_ply(os.path.join(objcts_path, "gaussians_before_removal.ply"))

        # objects to remove
        objects = [f for f in os.listdir(objcts_path) if os.path.isfile(os.path.join(objcts_path, f))]
        for obj in objects:
            if obj == "gaussians_before_removal.ply":
                continue
            gs_object = InteractiveGaussianModel(self.dataset.sh_degree)
            gs_object.load_ply(os.path.join(objcts_path, obj))

            # find close gaussians
            dists, indices = environment.get_close_gaussians(gs_object.get_xyz, 50)
            min_dists, _ = torch.min(dists, dim=-1)

            # filter environment
            mask = min_dists >= 0.01
            environment.filter_by_mask(mask)

        # save environment
        environment.save_ply(os.path.join(objcts_path, "gaussians.ply"))