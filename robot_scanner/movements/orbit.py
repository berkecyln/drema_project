import numpy as np

from robot_scanner.movements.base import BaseMover
from robot_scanner.core.geometry import look_at_center


class OrbitMover(BaseMover):

    def generate_path(self, cfg):
        initial_pos, _ = self.robot.get_tcp_pos_orn()
        p = cfg.orbit
        center = initial_pos + np.array(p.center_offset)

        print(f"Orbit: R={p.radius}, H={p.height}, {p.start_angle}°→{p.end_angle}°")

        path = []
        for theta in np.linspace(np.radians(p.start_angle), np.radians(p.end_angle), cfg.num_waypoints):
            pos = center + np.array([p.radius * np.cos(theta), p.radius * np.sin(theta), p.height])
            path.append((pos, look_at_center(pos, center)))
        return path
