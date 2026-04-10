import math
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from robot_io.utils.utils import quat_to_euler, euler_to_quat

from robot_scanner.movements.base import BaseMover


class RandomMover(BaseMover):

    def generate_path(self, cfg):
        return []

    def execute(self, cfg, task_name):
        task_dir = os.path.join(self.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        s = cfg.random_pose
        _, neutral_orn = self.robot.get_tcp_pos_orn()
        n = cfg.num_waypoints
        success = 0
        attempts = 0

        print(f"Random sampler: {n} poses, centered={s.centered}")

        while success < n and attempts < n * 2:
            attempts += 1
            pos, orn = self._centered_sample(s.centered_sampler) if s.centered else self._box_sample(s, neutral_orn)
            try:
                self.robot.move_cart_pos_abs_ptp(pos, orn)
                time.sleep(0.5)
                self._capture(success + 1, task_dir)
                success += 1
            except Exception as e:
                print(f"[SKIP] Unreachable pose: {e}")

        print(f"Done. {success}/{n} frames saved to: {task_dir}")

    def _box_sample(self, cfg, neutral_orn):
        pos = np.array([
            np.random.uniform(*cfg.x_limits),
            np.random.uniform(*cfg.y_limits),
            np.random.uniform(*cfg.z_limits),
        ])
        neutral_euler = quat_to_euler(neutral_orn)
        noise = np.radians(cfg.max_rotation_noise_deg)
        euler = np.array([
            math.pi + np.random.uniform(-noise, noise),
            np.random.uniform(-noise, noise),
            neutral_euler[2] + np.random.uniform(-noise, noise),
        ])
        return pos, euler_to_quat(euler)

    def _centered_sample(self, cfg):
        theta = np.random.uniform(*cfg.theta_limits)
        vec = np.array([np.cos(theta), np.sin(theta), 0]) * np.random.uniform(*cfg.r_limits)
        trans = np.cross(np.array([0.0, 0.0, 1.0]), vec) * np.random.uniform(*cfg.trans_limits)
        height = np.array([0.0, 0.0, 1.0]) * np.random.uniform(*cfg.h_limits)
        pos = np.array(cfg.initial_pos) + vec + trans + height
        euler = np.array([
            math.pi + np.random.uniform(*cfg.pitch_limit),
            np.random.uniform(*cfg.roll_limit),
            np.random.uniform(*cfg.yaw_limits) + theta,
        ])
        return pos, euler_to_quat(euler)
