import os
import time
import numpy as np

from robot_scanner.movements.base import BaseMover
from robot_scanner.core.geometry import look_at_center


class GridScanMover(BaseMover):
    STEP_SLEEP = 1.0

    def generate_path(self, cfg):
        return []  # not used — execute() is overridden

    def execute(self, cfg, task_name):
        task_dir = os.path.join(self.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        g = cfg.grid_scan

        # Derive scene center from neutral pose — same approach as arch.
        # Robot is at neutral here (recover_to_center ran before execute).
        # Use neutral XY as center, drop Z slightly to point at table surface level.
        neutral_pos, _ = self.robot.get_tcp_pos_orn()
        scene_center = neutral_pos + np.array([0.0, 0.0, g.get("scene_center_z_offset", -0.1)])
        print(f"Scene center: {np.round(scene_center, 3)}")

        y_base = np.linspace(g.y_start, g.y_end, cfg.num_waypoints_per_pass)

        # z descending (high first = safest), x ascending (close first)
        passes = [
            (x, z)
            for z in sorted(g.z_heights, reverse=True)
            for x in sorted(g.x_positions)
        ]
        total = len(passes)
        frame_idx = 1

        for pass_idx, (x, z) in enumerate(passes):
            # snake pattern: alternate Y direction each pass to minimize travel
            y_vals = y_base if pass_idx % 2 == 0 else y_base[::-1]
            direction = "→" if pass_idx % 2 == 0 else "←"
            print(f"\nPass {pass_idx + 1}/{total}: X={x:.2f}  Z={z:.2f}  {direction}")

            path = [
                (np.array([x, y, z]), look_at_center(np.array([x, y, z]), scene_center))
                for y in y_vals
            ]

            self.robot.move_cart_pos_abs_ptp(path[0][0], path[0][1])
            time.sleep(1.5)

            for pos, orn in path:
                self.robot.move_cart_pos_abs_ptp(pos, orn)
                time.sleep(self.STEP_SLEEP)
                self._capture(frame_idx, task_dir)
                frame_idx += 1

        print(f"\nGrid scan complete. {frame_idx - 1} frames saved to: {task_dir}")
