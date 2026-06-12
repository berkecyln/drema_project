import numpy as np

from robot_scanner.movements.base import BaseMover


def subsample_by_distance(trajectory, n_frames):
    """
    Subsample to n_frames waypoints evenly spaced by arc length.
    Index-based subsampling would over-sample where the recording moved slowly;
    distance-based gives uniform spatial coverage regardless of recording speed.
    """
    positions = np.array([f["tcp_pos"] for f in trajectory])  # (N, 3)

    diffs = np.diff(positions, axis=0)                          # (N-1, 3)
    step_lengths = np.linalg.norm(diffs, axis=1)               # (N-1,)
    arc = np.concatenate([[0], np.cumsum(step_lengths)])        # (N,)
    total_length = arc[-1]

    if total_length < 0.01:
        raise ValueError(
            f"Trajectory total arc length is {total_length:.4f}m (<1cm). "
            "Too short to subsample meaningfully. Record a longer path."
        )

    targets = np.linspace(0, total_length, n_frames)
    indices = [int(np.argmin(np.abs(arc - t))) for t in targets]

    return [trajectory[i]["joint_positions"] for i in indices]


class GelloMover(BaseMover):
    """
    Replays a GELLO-recorded trajectory (.npy from gello_teleop.py),
    subsampled to n_frames waypoints. Config: gello_replay.yaml.
    """

    def generate_path(self, cfg) -> list:
        traj = np.load(cfg.trajectory_file, allow_pickle=True)
        print(f"Loaded trajectory: {len(traj)} frames from {cfg.trajectory_file}")
        path = subsample_by_distance(traj, cfg.n_frames)
        print(f"Subsampled to {len(path)} waypoints.")
        return path
