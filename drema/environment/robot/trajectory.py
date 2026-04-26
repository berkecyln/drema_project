import copy
import pickle

import numpy as np

from drema.utils.trajectory_utils import compute_keypoints


class RobotTrajectory:
    def __init__(self, path, offset=np.zeros(3), remove_initial_part=False, number_of_steps=0):
        self.path = path
        self.offset = offset
        self.remove_initial_part = remove_initial_part
        self.number_of_steps = number_of_steps

        self.demo = [] # list of dictionaries containing the trajectory
        self.keypoints = []
        self.cameras = []
        self.indexes_after_first_keypoint = []
        self.demo_after_first_keypoint = []

        self.start_joint_positions = []
        self.start_waypoint = {}

        self.load()
        if self.remove_initial_part:
            self.remove_initial_part_trajectory(self.number_of_steps)

        self.compute_keypoints()

    def load(self):
        """
        Load the trajectory data from the path and popolate the class attributes
        :return:
        """

        file = open(self.path, "rb")
        self.demo = pickle.load(file)
        file.close()

        # extract starting joint positions
        self.start_joint_positions = np.concatenate((self.demo[0]["joint_positions"], self.demo[0]["gripper_joint_positions"]))
        self.start_waypoint = copy.deepcopy(self.demo[0])

        # add an offset if needed
        for i in range(len(self.demo)):
            self.demo[i]["gripper_pose"][:3] += self.offset

    def reset(self):
        """
        Reset the trajectory
        :return:
        """
        self.load()
        if self.remove_initial_part:
            self.remove_initial_part_trajectory(self.number_of_steps)
        self.compute_keypoints()

    def save(self, path):
        """
        Save the trajectory on the given path
        :return:
        """
        file = open(path, "wb")
        pickle.dump(self.demo, file)
        file.close()

    def compute_keypoints(self):
        """
        Compute the keypoints, the episodes after the first keypoint and the whole demo after it
        :return:
        """

        self.keypoints, self.indexes_after_first_keypoint, self.demo_after_first_keypoint = compute_keypoints(self.demo)


    def remove_initial_part_trajectory(self, number_of_steps):
        """

        :return:
        """

        self.demo = self.demo[number_of_steps:]
        self.compute_keypoints()


    def is_keypoint(self, waypoint_index):
        """
        Check if the waypoint_index is one of the keypoints
        :return: boolean value
        """
        return waypoint_index in self.keypoints

    def get_cameras(self, waypoint_index=0):
        """
        Get the cameras for the given trajectory step
        :param waypoint_index: integer index of the trajectory step
        :return:
        """
        step = self.demo[waypoint_index]

        # check the camera names in the step
        keys_with_camera_intrinsics = [key for key in step.keys() if "_camera_intrinsics" in key]

        # create the list of camera names to render
        camera_names = [key.split("_camera_intrinsics")[0] for key in keys_with_camera_intrinsics]

        camera_params = {}
        for camera_name in camera_names:
            w = step.get(camera_name + "_camera_width", None)
            h = step.get(camera_name + "_camera_height", None)
            crop_top = step.get(camera_name + "_camera_crop_top", 0)
            camera_params[camera_name] = (camera_name, step[camera_name + "_camera_extrinsics"], step[camera_name + "_camera_intrinsics"], step[camera_name + "_camera_far"], step[camera_name + "_camera_near"], w, h, crop_top)

        return camera_params

    def get_target_pose(self, waypoint_index):
        """
        Get the target pose for the given trajectory step
        :param waypoint_index: integer index of the trajectory step
        :return:
        """
        return copy.deepcopy(self.demo[waypoint_index]["gripper_pose"])

    def get_trajectory_positions(self):
        """
        Get the trajectory positions
        :return: a list of positions
        """
        return copy.deepcopy([waypoint["gripper_pose"][:3] for waypoint in self.demo])

    def get_gripper_open(self, waypoint_index):
        """
        Get the gripper open for the given trajectory step
        :param waypoint_index: integer index of the trajectory step
        :return:
        """
        return copy.deepcopy(self.demo[waypoint_index]["gripper_open"])

    def set_gripper_open(self, gripper_open, waypoint_index):
        """
        Set the gripper open for the current trajectory step
        :param gripper_open: boolean value of the gripper open
        :param waypoint_index: integer index of the trajectory step
        :return:
        """
        self.demo[waypoint_index]["gripper_open"] = gripper_open

    def set_gripper_pose(self, gripper_pose, waypoint_index):
        """
        Set the gripper pose for the current trajectory step
        :param gripper_pose:
        :param waypoint_index: integer index of the trajectory step
        :return:
        """
        self.demo[waypoint_index]["gripper_pose"] = gripper_pose

    def set_joint_positions(self, joint_positions, waypoint_index):
        """
        Set the joint positions for the current trajectory step
        :param joint_positions: integer index of the trajectory step
        :return:
        """
        self.demo[waypoint_index]["joint_positions"] = joint_positions

    def set_gripper_joint_positions(self, gripper_joint_positions, waypoint_index):
        """
        Set the gripper joint positions for the current trajectory step
        :param gripper_joint_positions:
        :param waypoint_index:
        :return:
        """
        self.demo[waypoint_index]["gripper_joint_positions"] = gripper_joint_positions

    def __len__(self):
        return len(self.demo)

    def __copy__(self):
        return RobotTrajectory(self.path, self.offset)

