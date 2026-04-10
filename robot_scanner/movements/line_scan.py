import numpy as np
from scipy.spatial.transform import Rotation

from robot_scanner.movements.base import BaseMover


class LineScanMover(BaseMover):
    STEP_SLEEP = 1.0

    def generate_path(self, cfg):
        start = np.array(cfg.line_scan.start_point)
        end = np.array(cfg.line_scan.end_point)
        _, orn = self.robot.get_tcp_pos_orn()

        rot_deg = cfg.line_scan.get("rot_deg", None)
        if rot_deg is not None:
            orn = (Rotation.from_quat(orn) * Rotation.from_euler("z", rot_deg, degrees=True)).as_quat()

        print(f"Line scan: {np.round(start, 3)} → {np.round(end, 3)}, steps={cfg.num_waypoints}")

        path = []
        for i in range(cfg.num_waypoints):
            t = i / (cfg.num_waypoints - 1)
            path.append(((1 - t) * start + t * end, orn))
        return path
