import numpy as np
from scipy.spatial.transform import Rotation

from robot_scanner.movements.base import BaseMover
from robot_scanner.core.geometry import get_orientation


class ArchMover(BaseMover):

    def _bezier_setup(self, arch_offset, flip):
        pos, orn = self.robot.get_tcp_pos_orn()
        if flip:
            # Note: flipping can cause joint limit issues on some configurations
            orn = (Rotation.from_quat(orn) * Rotation.from_euler("z", 180, degrees=True)).as_quat()
        start = pos + np.array(arch_offset.start_offset)
        end = pos + np.array(arch_offset.end_offset)
        corner = pos + np.array(arch_offset.corner_offset)
        return start, end, corner, orn

    def generate_path(self, cfg):
        arch_type = cfg.arch_type
        if arch_type == "parallel":
            arch_offset = cfg.arch_parallel
        elif arch_type == "skewed_clockwise":
            arch_offset = cfg.arch_skewed_clockwise
        elif arch_type == "skewed_counterclockwise":
            arch_offset = cfg.arch_skewed_counterclockwise
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")

        start, end, corner, initial_orn = self._bezier_setup(arch_offset, cfg.flip_orientation)
        n = cfg.num_waypoints
        is_skewed = abs(start[0] - end[0]) > 0.05
        max_tilt = 25.0 if is_skewed else 35.0

        print(f"Arch path: {arch_type}, waypoints={n}, max_tilt={max_tilt}°")
        print(f"  Start={np.round(start, 3)}  End={np.round(end, 3)}  Corner={np.round(corner, 3)}")

        path = []
        for i in range(n):
            t = i / (n - 1)
            pos = np.array([
                (1 - t)**2 * start[j] + 2 * (1 - t) * t * corner[j] + t**2 * end[j]
                for j in range(3)
            ])
            orn = get_orientation(pos, start, end, corner, max_tilt, initial_orn)
            path.append((pos, orn))
        return path
