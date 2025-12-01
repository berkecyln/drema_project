import os
import numpy as np
import yaml

from scipy.spatial.transform import Rotation as R

from drema.environment.base_environment import Environment
from drema.environment.robot_envirionment import RobotEnvironment
from drema.environment.robot.trajectory import RobotTrajectory
from drema.environment.observer.camera import CameraManager
from drema.utils.coppelia_utils import read_labels


class Builder:
    def __init__(self, args):
        self.args = args

        # crete paths
        self.objects_path = os.path.join(self.args.data.assets_path, self.args.simulation.objects.urdf_dir)
        self.labels_path = os.path.join(self.args.data.source_path, self.args.simulation.environment.labels_file)
        self.gaussians_path = os.path.join(self.args.data.assets_path, self.args.simulation.objects.gaussians_dir)
        self.flat_surface_path = os.path.join(self.args.data.assets_path, self.args.simulation.objects.flat_surface_dir)
        if self.args.simulation.objects.environment_file is not None:
            self.environment_path = os.path.join(self.args.data.assets_path, self.args.simulation.objects.gaussians_dir, self.args.simulation.objects.environment_file)
        else:
            self.environment_path = None

        # objects list
        self.objects_list = [x for x in os.listdir(self.objects_path) if os.path.isdir(os.path.join(self.objects_path, x))]

        # read the labels
        self.labels = read_labels(self.labels_path, filter_labels=self.args.simulation.environment.filter_labels)

        # if the object_list is different from ["All"] keep only the objects in the list
        if self.args.simulation.objects.objects_list != ["All"]:
            # for each label
            for name, value in self.labels.items():
                # if the label is not in the objects list
                if name not in self.args.simulation.objects.objects_list:
                    # remove the label
                    del self.labels[name]
                    # remove the object from the objects list
                    self.objects_list.remove(str(value))

        # get not collidable objects
        self.not_collidable_objects = []
        for name, value in self.labels.items():
            if name in self.args.simulation.objects.not_collidable_objects:
                self.not_collidable_objects.append(value)

        # get fixed objects
        self.fixed_objects = []
        for name, value in self.labels.items():
            if name in self.args.simulation.objects.fixed_objects:
                self.fixed_objects.append(value)

    def create_pipe_args(self):
        """
        Create the pipe arguments
        :return: a dictionary with the pipe arguments
        """

        # create a class object with the pipe arguments
        class PipeArgs:
            def __init__(self, args):
                self.convert_SHs_python = args.training.pipeline.convert_SHs_python
                self.compute_cov3D_python = args.training.pipeline.compute_cov3D_python
                self.depth_ratio = args.training.pipeline.depth_ratio
                self.debug = args.training.pipeline.debug

        """
        pipe_args = {
            "convert_SHs_python": self.args.training.pipeline.convert_SHs_python,
            "compute_cov3D_python": self.args.training.pipeline.compute_cov3D_python,
            "debug": self.args.training.pipeline.debug
        }
        """

        return PipeArgs(self.args)


    def create_robot_configuration(self):
        """
        Create the robot configuration and return a dictionary with it
        :return: a dictionary with the robot configuration
        """

        # read the links from the folder containing the gaussians
        link_files = [f for f in os.listdir(self.args.simulation.robot.robot_gaussians_dir)]

        # sort the links base on the ending number
        link_files = sorted(link_files, key=lambda x: int(x[4:].split(".")[0]))
        links_names = [f.split(".")[0] for f in link_files]
        gaussian_files = [os.path.join(self.args.simulation.robot.robot_gaussians_dir, f) for f in link_files]

        robot = {
            "urdf_path": self.args.simulation.robot.robot_urdf,
            "gaussians_path": gaussian_files,
            "connected_joints": self.args.simulation.robot.connected_joints,
            "initial_position": self.args.simulation.robot.initial_position,
            "initial_orientation": np.array([0, 0, 0, 1]),
            "initial_joint_positions": self.args.simulation.robot.initial_joint_positions,
            "initial_gripper_joint_positions": self.args.simulation.robot.initial_gripper_joint_positions,
            "end_effector_rotation": R.from_euler("xyz", self.args.simulation.robot.end_effector_rotation, degrees=True),
            "right_finger_index": self.args.simulation.robot.right_finger_index,
            "left_finger_index": self.args.simulation.robot.left_finger_index,
            "grasp_index": self.args.simulation.robot.grasp_index,
            "wrist_camera_index": self.args.simulation.robot.wrist_camera_index,
            "fixed": True

        }

        return robot

    def create_objets_configurations(self):
        """
        Create the objects configurations and return a dictionary with them
        :return:
        """

        objects = {}
        for obj in self.objects_list:
            # create the object path
            object_path = os.path.join(self.objects_path, obj)

            # if the configuration yaml file exists
            if os.path.exists(os.path.join(object_path, obj + ".yaml")):

                # load the configuration
                with open(os.path.join(object_path, obj + ".yaml"), "r") as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)
            else:
                # create the configuration

                # read the initial position in the file ending with .npy
                initial_position = np.load(os.path.join(object_path, "mesh_coordinates" + obj + ".npy"))

                config = {
                    "initial_position": initial_position,
                    "initial_orientation": np.array([0, 0, 0, 1]),
                    "mass": 4,
                    "lateral_friction": 0.5,
                    "spinning_friction": 1,
                    "rolling_friction": 1
                }

            config["urdf_path"] = os.path.join(self.objects_path, obj + ".urdf")
            config["gaussians_path"] = os.path.join(self.gaussians_path, obj + ".ply")

            # get the key the dictionary self.labels based on the value
            """
            label_value = None
            for key, value in self.labels.items():
                if str(value) == obj:
                    label_name = key
                    break
            """

            # set the collidable flag
            if int(obj) in self.not_collidable_objects:
                config["collidable"] = False
            else:
                config["collidable"] = True

            # set the fixed flag
            if int(obj) in self.fixed_objects:
                config["fixed"] = True
            else:
                config["fixed"] = False

            objects[obj] = config

        return objects

    def create_flat_surface_configuration(self):
        """
        Create the flat surface configuration and return a dictionary with it
        :return: a dictionary with the flat surface configuration
        """
        initial_position = np.load(os.path.join(self.flat_surface_path, "position.npy"))

        flat_surface = {
            "urdf_path": os.path.join(self.flat_surface_path, self.args.simulation.objects.flat_surface_file),
            "gaussians_path": None,
            "initial_position": initial_position,
            "initial_orientation": np.array([0, 0, 0, 1])
        }

        return flat_surface

    def create_cameras(self, trajectory: RobotTrajectory):
        """
        Create the cameras
        :param camera_manager: the camera manager
        :return: None
        """
        camera_manager = CameraManager()

        if self.args.simulation.visualization.visualize:
            if self.args.simulation.visualization.visualize_training_cameras:
                camera_manager.load_cameras_from_directory(self.args.data.source_path, scale=self.args.simulation.visualization.scale,
                                                           image_dir=self.args.training.model.images, simulation=False, visualization=True)
            if self.args.simulation.visualization.visualize_trajectory_cameras:
                camera_manager.load_cameras_from_trajectory(trajectory, scale=self.args.simulation.visualization.scale, simulation=False, visualization=True)

        if self.args.simulation.generation.generate_data:
            camera_manager.load_cameras_from_trajectory(trajectory, scale=self.args.simulation.generation.scale, simulation=True, visualization=False)

        return camera_manager

    def load_trajectory(self):
        """
        Load the trajectory
        :return: trajectory
        """
        if self.args.simulation.trajectory.load_trajectory:
            trajectory_path = os.path.join(self.args.data.source_path, self.args.simulation.trajectory.trajectory_file)
            trajectory = RobotTrajectory(trajectory_path, self.args.simulation.trajectory.offset,
                                         self.args.simulation.trajectory.remove_initial_part_trajectory,
                                         self.args.simulation.trajectory.remove_initial_part_trajectory_steps)

        else:
            trajectory = None
        return trajectory

    def create_environment(self, trajectory=None, objects=True):
        """
        Create the environment
        :return: the environment
        """

        # create the pipe arguments
        pipe_args = self.create_pipe_args()

        # create the objects configurations
        if self.args.simulation.environment.simulate_objects:
            objects = self.create_objets_configurations()
        else:
            objects = None

        # create the flat surface configuration
        flat_surface = self.create_flat_surface_configuration()

        if self.args.simulation.robot.simulate_robot:
            # create the robot configuration
            robot = self.create_robot_configuration()
            environment = RobotEnvironment(objects, flat_surface, pipe_args, robot,
                                           sh_degrees=self.args.training.model.sh_degree,
                                           trajectory=trajectory,
                                           max_waypoint_steps=self.args.simulation.trajectory.max_waypoint_steps,
                                           waypoint_threshold=self.args.simulation.trajectory.waypoint_threshold,
                                           keypoint_threshold=self.args.simulation.trajectory.keypoint_threshold,
                                           environment_path=self.environment_path,
                                           visualize=self.args.simulation.visualization.visualize)
        else:
            # create the environment
            environment = Environment(objects, flat_surface, pipe_args, sh_degrees=self.args.training.model.sh_degree,
                                      environment_path=self.environment_path,
                                      visualize=self.args.simulation.visualization.visualize)

        return environment

