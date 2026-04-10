import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def save_frame(data, index, task_dir, cam_manager, T_tcp_cam):
    for d in ("images", "depth_scaled", "poses"):
        os.makedirs(os.path.join(task_dir, d), exist_ok=True)

    cv2.imwrite(os.path.join(task_dir, "images", f"{index:04d}.png"), data["rgb"][:, :, ::-1])
    np.save(os.path.join(task_dir, "depth_scaled", f"{index:04d}.npy"), data["depth"])

    T_base_tcp = np.eye(4)
    T_base_tcp[:3, 3] = data["tcp_pos"]
    T_base_tcp[:3, :3] = Rotation.from_quat(data["tcp_orn"]).as_matrix()
    T_base_cam = T_base_tcp @ T_tcp_cam

    intr = cam_manager.gripper_cam.get_intrinsics()
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])

    with open(os.path.join(task_dir, "poses", f"{index:04d}.txt"), "w") as f:
        np.savetxt(f, T_base_cam, fmt="%.6f")
        f.write("\n")
        np.savetxt(f, K, fmt="%.6f")

    if index == 1:
        np.savetxt(os.path.join(task_dir, "distortion_coeffs.txt"), cam_manager.gripper_cam.get_dist_coeffs())

    gripper_cam = cam_manager.gripper_cam
    if getattr(gripper_cam, "save_ir", False) and gripper_cam.last_ir_left is not None:
        for sub, frame in [("images_ir_left", gripper_cam.last_ir_left), ("images_ir_right", gripper_cam.last_ir_right)]:
            os.makedirs(os.path.join(task_dir, sub), exist_ok=True)
            cv2.imwrite(os.path.join(task_dir, sub, f"{index:04d}.png"), frame)
        if index == 1:
            ir_meta = {
                "intrinsics": gripper_cam.get_ir_intrinsics(),
                "baseline_m": gripper_cam.get_stereo_baseline(),
                "flipped": gripper_cam.flipped_cam,
            }
            with open(os.path.join(task_dir, "ir_metadata.json"), "w") as f:
                json.dump(ir_meta, f, indent=2)
