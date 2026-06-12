import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # resolve before chdir

ROBOT_IO_PATH = "/home/ceylanb/robot/robot_io"  # set this for your machine
if ROBOT_IO_PATH not in sys.path:
    sys.path.insert(0, ROBOT_IO_PATH)
os.chdir(ROBOT_IO_PATH)

import datetime
import numpy as np
import hydra
from omegaconf import DictConfig

from robot_scanner.core.robot_utils import recover_to_center, print_robot_position
from robot_scanner.core.video_recorder import KinectFrameRecorder
from robot_scanner.movements.arch import ArchMover
from robot_scanner.movements.orbit import OrbitMover
from robot_scanner.movements.line_scan import LineScanMover
from robot_scanner.movements.random_pose import RandomMover
from robot_scanner.movements.gello_replay import GelloMover

DREMA_PROJECT_PATH = os.path.dirname(SCRIPT_DIR)
CONFIGS_PATH = os.path.join(SCRIPT_DIR, "configs")

MOVERS = {
    "arch": ArchMover,
    "orbit": OrbitMover,
    "line_scan": LineScanMover,
    "random_pose": RandomMover,
    "gello_replay": GelloMover,
}


@hydra.main(config_path=CONFIGS_PATH, config_name="data_gathering", version_base=None)
def main(cfg: DictConfig):
    robot = cam_manager = None
    try:
        robot = hydra.utils.instantiate(cfg.robot)
        cam_manager = hydra.utils.instantiate(cfg.cams)
        T_tcp_cam = np.load(cfg.calibration_file)

        recover_to_center(robot)
        print_robot_position(robot, "Initial")

        mover_cls = MOVERS.get(cfg.movement.type)
        if mover_cls is None:
            raise ValueError(f"Unknown movement type: '{cfg.movement.type}'. Choose from: {list(MOVERS)}")

        output_dir = cfg.get("output_dir", os.path.join(DREMA_PROJECT_PATH, "input"))
        mover = mover_cls(robot, cam_manager, T_tcp_cam, output_dir,
                          save_frames=cfg.get("save_frames", True))

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"task_{cfg.movement.type}_{ts}"

        recorder = None
        if cfg.get("record_frames", False):
            task_dir = os.path.join(output_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            recorder = KinectFrameRecorder(cam_manager.static_cam, os.path.join(task_dir, "overhead"))
            recorder.start()

        try:
            mover.execute(cfg.movement, task_name)
        finally:
            if recorder is not None:
                recorder.stop()

        print_robot_position(robot, "Final")
        recover_to_center(robot)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        if robot is not None:
            try:
                recover_to_center(robot)
            except Exception:
                pass
    finally:
        if robot is not None:
            try:
                robot.abort_motion()
                print("Robot stopped.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
