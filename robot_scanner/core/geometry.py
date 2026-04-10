import numpy as np
from scipy.spatial.transform import Rotation


def calculate_center(start, end):
    return np.array([(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, start[2]])


def get_orientation(position, start, end, corner, max_tilt, initial_orn):
    center = calculate_center(start, end)
    look_dir = center - position
    look_dir /= np.linalg.norm(look_dir)

    initial_rot = Rotation.from_quat(initial_orn)
    R = initial_rot.as_matrix()
    initial_x, initial_z = R[:, 0], R[:, 2]

    dot = np.clip(np.dot(initial_z, look_dir), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    max_rad = np.radians(max_tilt)

    if angle_rad > max_rad:
        axis = np.cross(initial_z, look_dir)
        if np.linalg.norm(axis) > 0.001:
            axis /= np.linalg.norm(axis)
            return (Rotation.from_rotvec(axis * max_rad) * initial_rot).as_quat()
        return initial_orn

    z = look_dir
    x = initial_x - np.dot(initial_x, z) * z
    if np.linalg.norm(x) < 0.01:
        x = R[:, 1] - np.dot(R[:, 1], z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    return Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()


def look_at_center(current_pos, target_pos):
    z = target_pos - current_pos
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
