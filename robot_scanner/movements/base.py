import os
import time
from abc import ABC, abstractmethod

from robot_scanner.core.frame_saver import save_frame


class BaseMover(ABC):
    STEP_SLEEP = 1.5

    def __init__(self, robot, cam_manager, T_tcp_cam, output_dir):
        self.robot = robot
        self.cam_manager = cam_manager
        self.T_tcp_cam = T_tcp_cam
        self.output_dir = output_dir

    @abstractmethod
    def generate_path(self, cfg) -> list:
        """Return list of (position, orientation) waypoints."""
        ...

    def _capture(self, index, task_dir):
        pos, orn = self.robot.get_tcp_pos_orn()
        imgs = self.cam_manager.get_images()
        save_frame(
            {"rgb": imgs["rgb_gripper"], "depth": imgs["depth_gripper"], "tcp_pos": pos, "tcp_orn": orn},
            index, task_dir, self.cam_manager, self.T_tcp_cam,
        )
        self.cam_manager.render()

    def execute(self, cfg, task_name):
        path = self.generate_path(cfg)
        task_dir = os.path.join(self.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        print(f"Moving to start...")
        # self.robot.move_cart_pos_abs_lin(path[0][0], path[0][1])
        self.robot.move_cart_pos_abs_ptp(path[0][0], path[0][1])
        time.sleep(2)

        for idx, (pos, orn) in enumerate(path, 1):
            # self.robot.move_cart_pos_abs_lin(pos, orn)
            self.robot.move_cart_pos_abs_ptp(pos, orn)
            time.sleep(self.STEP_SLEEP)
            self._capture(idx, task_dir)

        print(f"Done. Saved to: {task_dir}")
