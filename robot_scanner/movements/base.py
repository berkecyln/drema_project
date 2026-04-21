import os
import time
from abc import ABC, abstractmethod

try:
    from franky import ControlException as FrankyControlException
except ImportError:
    FrankyControlException = None

from robot_scanner.core.frame_saver import save_frame


class BaseMover(ABC):
    STEP_SLEEP = 1.0
    MOVE_RETRIES = 2
    RETRY_DELAY = 1.0

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

    def _move(self, waypoint):
        """Move to waypoint, retrying on ControlException. Returns True on success.

        waypoint can be:
            - np.ndarray of shape (7,): joint positions = move_joint_pos
            - tuple (pos, orn): Cartesian pose = move_cart_pos_abs_lin
        """
        exc_type = FrankyControlException if FrankyControlException is not None else Exception
        for attempt in range(1 + self.MOVE_RETRIES):
            try:
                import numpy as np
                if isinstance(waypoint, np.ndarray):
                    self.robot.move_joint_pos(waypoint)
                else:
                    pos, orn = waypoint
                    self.robot.move_cart_pos_abs_lin(pos, orn)
                return True
            except exc_type as e:
                if attempt < self.MOVE_RETRIES:
                    print(f"  ControlException (attempt {attempt + 1}): {e} - retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    print(f"  ControlException after {1 + self.MOVE_RETRIES} attempts: {e} - skipping frame.")
                    return False

    def execute(self, cfg, task_name):
        path = self.generate_path(cfg)
        task_dir = os.path.join(self.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        print(f"Moving to start...")
        self._move(path[0])
        time.sleep(2)

        skipped = 0
        for idx, waypoint in enumerate(path, 1):
            if not self._move(waypoint):
                skipped += 1
                continue
            time.sleep(self.STEP_SLEEP)
            self._capture(idx, task_dir)

        print(f"Done. Saved to: {task_dir}" + (f" ({skipped} frames skipped)" if skipped else ""))
