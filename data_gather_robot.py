"""
Data Gathering Robot Script for DreMa Using Franka Emika Panda with robot_io
robot_io repository: https://github.com/acl21/robot_io

This script automates RGB-D data collection by moving the Panda robot
in an arch (semicircle) pattern across the workspace.


Usage:
    conda activate robotio
    cd /home/ceylanb/DreMa/drema_project
    python data_gather_robot.py
"""

import sys
import os

# Add robot_io package to Python path
ROBOT_IO_PATH = "/home/ceylanb/robot/robot_io"
if ROBOT_IO_PATH not in sys.path:
    sys.path.insert(0, ROBOT_IO_PATH)

os.chdir(ROBOT_IO_PATH)

import time
import cv2
import datetime
import numpy as np
import hydra
import math
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
from robot_io.utils.utils import quat_to_euler, euler_to_quat

DREMA_PROJECT_PATH = "/home/ceylanb/DreMa/drema_project"
INPUT_PATH = os.path.join(DREMA_PROJECT_PATH, "input")


# MOVEMENT FUNCTIONS
def recover_to_center(robot):
    """Move the robot back to the neutral (center) position.
    
    Uses the neutral_pose defined in the robot configuration file.
    This is a safe recovery position.
    
    Args:
        robot: PandaFrankYInterface instance
    """
    print("====================")
    print("MOVING CENTER")
    print("====================")
    
    print("Moving to neutral pose...")
    robot.move_to_neutral()

    final_pos, _ = robot.get_tcp_pos_orn()
    print(f"Position after recovery: X={final_pos[0]:.3f}, Y={final_pos[1]:.3f}, Z={final_pos[2]:.3f}")

def create_bezier_points(robot, flip:bool = False, arch_offset=None) -> tuple:
    """
    Create anchot points for the Bézier curve based on the robot's initial position.
    
    Args:
        robot: PandaFrankYInterface instance
        flip: bool, whether to flip the initial orientation 180° around Z-axis
        arch_offset: offsets for start, end, and corner points of the arch
    
    Returns:
        start: np.array, start point of the arch
        end: np.array, end point of the arch
        corner: np.array, control point for the arch
        initial_pos: list, initial position of the robot TCP
        initial_orn: list, initial orientation of the robot TCP
    """
    initial_pos, initial_orn = robot.get_tcp_pos_orn()

    # Flip initial orientation 180° around Z-axis if FLIP is enabled 
    # Has some  issues it crashes to limit need to check
    if flip:
        initial_rot = Rotation.from_quat(initial_orn)
        flip_rotation = Rotation.from_euler('z', 180, degrees=True)
        flipped_rot = initial_rot * flip_rotation
        initial_orn = flipped_rot.as_quat()
        print(f"Flipped initial orientation 180° around Z-axis")

    start_pos = initial_pos.copy()
    end_pos = initial_pos.copy()
    corner_pos = initial_pos.copy()

    offset_start = np.array(arch_offset.start_offset)
    offset_end = np.array(arch_offset.end_offset)
    offset_corner = np.array(arch_offset.corner_offset)

    start = start_pos + offset_start
    end = end_pos + offset_end
    corner = corner_pos + offset_corner

    return start, end, corner, initial_pos , initial_orn

def calculate_center(start, end) -> np.array:
    """Calculate the center point of the table.
    
    Returns:
        center: np.array, center point
    """
    center_x = (start[0] + end[0]) / 2
    center_y = (start[1] + end[1]) / 2
    center_z = start[2]
    center = np.array([center_x, center_y, center_z])

    #print(f"Calculated center point: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")

    return center

def get_orientation(position, t, start, end, corner, max_tilt, initial_orn) -> np.array:
    """Get orientation quaternion that tilts toward center while maintaining gripper roll.
    
    This creates a "nodding" motion (tilt up/down and left/right) without twisting
    the gripper around its axis.
    
    Args:
        position: Current position [x, y, z]
        t: Parameter along curve (0 to 1)
        start: Start point of the arch
        end: End point of the arch
        corner: Control point of the arch
        max_tilt: Maximum tilt angle in degrees
        initial_orn: Initial orientation quaternion [x, y, z, w]
    
    Returns:
        orn: Orientation quaternion [x, y, z, w]
    """
    center = calculate_center(start, end)
    
    # Calculate direction to center
    look_direction = center - position
    look_direction = look_direction / np.linalg.norm(look_direction)
    
    # Get initial gripper orientation 
    initial_rot = Rotation.from_quat(initial_orn)
    initial_matrix = initial_rot.as_matrix()
   
    # Extract initial X-axis, Y-axis and Z-axis
    initial_x = initial_matrix[:, 0]
    initial_y = initial_matrix[:, 1]
    initial_z = initial_matrix[:, 2]

    # Calculate the angle between initial Z and desired look direction
    dot = np.clip(np.dot(initial_z, look_direction), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    max_tilt_deg=max_tilt

    # If angle exceeds limit, interpolate toward the limit
    max_tilt_rad = np.radians(max_tilt_deg)
    if angle_rad > max_tilt_rad:
        # Scale down the rotation
        scale = max_tilt_rad / angle_rad
        
        # Find rotation axis
        rotation_axis = np.cross(initial_z, look_direction)
        if np.linalg.norm(rotation_axis) > 0.001:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Create limited rotation
            limited_rotation = Rotation.from_rotvec(rotation_axis * max_tilt_rad)
            final_rotation = limited_rotation * initial_rot
            return final_rotation.as_quat()
        else:
            # Vectors are parallel, use initial orientation
            return initial_orn

    
    # Z axis points toward center
    z_axis = look_direction
    
    # Keep X-axis as close to initial as possible
    # Project initial X onto plane perpendicular to new Z
    x_axis = initial_x - np.dot(initial_x, z_axis) * z_axis
    
    # Handle edge case
    if np.linalg.norm(x_axis) < 0.01:
        # Use initial Y instead
        x_axis = initial_y - np.dot(initial_y, z_axis) * z_axis
    
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y-axis completes the frame
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build rotation matrix
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    rotation = Rotation.from_matrix(rot_matrix)
    
    quat = rotation.as_quat()
    
    return quat

def generate_arch_path(robot, start, end, corner, num_waypoints, initial_orn) -> tuple:
    """
    Generate arch path based on Quadratic Bézier Curve from start to end point.

    Args:
        robot: PandaFrankYInterface instance
        start: np.array, start point of the arch
        end: np.array, end point of the arch
        corner: np.array, control point for the arch
        num_waypoints: int, number of waypoints along the arch
        initial_orn: list, initial orientation of the robot TCP
    
    Returns:
        path: list of np.array, points along the arch path
    
    """

    # P(t) = (1-t)^2 * Start + 2(1-t)t * Corner + t^2 * End
    path = []
    waypoints = num_waypoints

    is_skewed = abs(start[0] - end[0]) > 0.05
    max_tilt = 25.0 if is_skewed else 35.0
    
    print(f"Path type: {'Skewed' if is_skewed else 'Parallel'}, max_tilt: {max_tilt}°")

    print("Generating arch path:")
    print(f"  Start: {start}")
    print(f"  End:   {end}")
    print(f"  Corner:{corner}")
    print(f"  Number of waypoints: {waypoints}")
    for i in range(waypoints):
        t = i / (waypoints -1)
        x_t = (1 - t)**2 * start[0] + 2 * (1 - t) * t * corner[0] + t**2 * end[0]
        y_t = (1 - t)**2 * start[1] + 2 * (1 - t) * t * corner[1] + t**2 * end[1]
        z_t = (1 - t)**2 * start[2] + 2 * (1 - t) * t * corner[2] + t**2 * end[2]
        orientation = get_orientation(np.array([x_t, y_t, z_t]), t, start, end, corner, max_tilt, initial_orn)
        
        #orientation = initial_orn  # Keep fixed orientation (no tilt)
        path.append(([x_t, y_t, z_t], orientation))
    
    print("Arch path generation completed.")
    for p in path: print(f"Pos: {np.round(p[0], 3)}, Ori: {p[1]}")
    
    return path

def save_frame(data, index, task_dir, cam_manager, T_tcp_cam):
    """
    Saves a single frame of data using Camera Manager and Calibration.
    """
    # Save RGB Image
    rgb_dir = os.path.join(task_dir, "images")
    os.makedirs(rgb_dir, exist_ok=True)
    
    # RGB is usually RGB in dictionary, convert to BGR for OpenCV
    bgr_image = data['rgb'][:, :, ::-1]
    cv2.imwrite(os.path.join(rgb_dir, f"{index:04d}.png"), bgr_image)

    # Save Depth Image
    depth_scaled_dir = os.path.join(task_dir, "depth_scaled")
    depth_dir = os.path.join(task_dir, "depth")
    os.makedirs(depth_scaled_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    np.save(os.path.join(depth_dir, f"{index:04d}.npy"), data['depth'])
    depth_mm = data['depth'] * 10  # scale with 10
    np.save(os.path.join(depth_scaled_dir, f"{index:04d}.npy"), depth_mm)

    # Save Poses (Camera -> World, i.e., C2W format expected by DreMa)
    pose_dir = os.path.join(task_dir, "poses")
    os.makedirs(pose_dir, exist_ok=True)
    # Get Robot Pose (Base -> TCP)
    tcp_pos = data['tcp_pos']
    tcp_orn = data['tcp_orn'] # [x, y, z, w]
    
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, 3] = tcp_pos
    T_base_tcp[:3, :3] = Rotation.from_quat(tcp_orn).as_matrix()

    # Calculate Camera Pose (Base -> Camera) using loaded calibration
    # T_base_cam = T_base_tcp * T_tcp_cam
    # This gives Camera-to-World (C2W) transform - camera position/orientation in robot base frame
    T_base_cam = T_base_tcp @ T_tcp_cam

    # Get Camera Intrinsics (positive focal lengths - standard OpenCV convention)
    intrinsics = cam_manager.gripper_cam.get_intrinsics()
    K = np.eye(3)
    K[0, 0] = intrinsics['fx']
    K[1, 1] = intrinsics['fy']
    K[0, 2] = intrinsics['cx']
    K[1, 2] = intrinsics['cy']

    # Write to text file
    pose_file = os.path.join(pose_dir, f"{index:04d}.txt")
    with open(pose_file, "w") as f:
        # Save C2W pose (camera pose in world coordinates) - DreMa expects this format
        np.savetxt(f, T_base_cam, fmt='%.6f') 
        f.write("\n")
        np.savetxt(f, K, fmt='%.6f')

def arch_move(robot, cam_manager=None, T_tcp_cam=None, arch_config=None, arch_type="parallel"):
    """Move the robot along an arch path defined by a Bezier curve.
    
    Args:
        robot: PandaFrankYInterface instance
        cam_manager: CameraManager instance for capturing images
        T_tcp_cam: np.array, transformation matrix from TCP to camera
        arch_config: configuration for arch movement
        arch_type: type of arch - "parallel", "skewed_clockwise", or "skewed_counterclockwise"
    """

    flip = arch_config.flip_orientation
    num_waypoints = arch_config.num_waypoints
    
    if arch_type == "parallel":
        arch_offset = arch_config.arch_parallel
    elif arch_type == "skewed_clockwise":
        arch_offset = arch_config.arch_skewed_clockwise
    elif arch_type == "skewed_counterclockwise":
        arch_offset = arch_config.arch_skewed_counterclockwise
    else:
        raise ValueError(f"Invalid arch_type: {arch_type}")

    start, end, corner, initial_pos, initial_orn = create_bezier_points(robot, flip, arch_offset)

    path = generate_arch_path(robot, start, end, corner, num_waypoints, initial_orn)

    positions = [point for point, _ in path]
    orientations = [orn for _, orn in path]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"task_{arch_type}_{timestamp}"
    task_dir = os.path.join(INPUT_PATH, task_name)
    print(f"===== Starting Capturing =====")
    print(f"    Task Name: {task_name}")
    print(f"    Data will be saved to: {task_dir}")

    print("Moving to Start point of the arch...")
    robot.move_cart_pos_abs_lin(positions[0], orientations[0])
    curr_pos, curr_orn = robot.get_tcp_pos_orn()
    print(f"    Start point reached: X={curr_pos[0]:.3f}, Y={curr_pos[1]:.3f}, Z={curr_pos[2]:.3f}, Ori={curr_orn}")

    time.sleep(2)

    print("Starting arch movement for capturing images...")
    for idx, (position, orientation) in enumerate(zip(positions[1:], orientations[1:]), 1):
        #print(f"    Moving to waypoint {idx + 1}/{len(path)}: X={position[0]:.3f}, Y={position[1]:.3f}, Z={position[2]:.3f}, Ori={orientation}")
        robot.move_cart_pos_abs_lin(position, orientation)
        time.sleep(0.5)
        if cam_manager is not None:
            # Match calibration approach: use actual robot pose after movement (in NE frame)
            # Calibration does: robot.get_tcp_pose() which returns NE frame
            actual_tcp_pos, actual_tcp_orn = robot.get_tcp_pos_orn()
            
            data = cam_manager.get_images()
            data_point = {
                'rgb': data['rgb_gripper'],
                'depth': data['depth_gripper'],
                'tcp_pos': actual_tcp_pos,
                'tcp_orn': actual_tcp_orn
            }
            
            save_frame(data_point, idx, task_dir, cam_manager, T_tcp_cam)
            
            cam_manager.render()
            
    #robot.move_cart_waypoints(positions[1:], orientations[1:])
    print("Arch movement completed.")

def sample_pose(cfg_sampler, neutral_euler):
    """
    Sample a random pose.
    Args:
        cfg_sampler: Configuration for the pose sampler
        neutral_euler: Neutral orientation in Euler angles
    Returns:
        target_pos: Sampled target position
        target_orn: Sampled target orientation quaternion
    """
    # Get Limits
    x = np.random.uniform(*cfg_sampler.x_limits)
    y = np.random.uniform(*cfg_sampler.y_limits)
    z = np.random.uniform(*cfg_sampler.z_limits)
    noise_deg = cfg_sampler.max_rotation_noise_deg
    
    target_pos = np.array([x, y, z])

    noise_rad = np.radians(noise_deg)
    
    # generate noises
    r_noise = np.random.uniform(-noise_rad, noise_rad) # Roll noise
    p_noise = np.random.uniform(-noise_rad, noise_rad) # Pitch noise
    y_noise = np.random.uniform(-noise_rad, noise_rad) # Yaw noise
    
    # get base yaw
    base_yaw = neutral_euler[2]

    target_euler = np.array([math.pi + r_noise,  p_noise, base_yaw + y_noise])
    
    target_orn = euler_to_quat(target_euler)
    
    return target_pos, target_orn

    print(f"Line scan completed. Saved to {task_dir}")

def line_scan_move(robot, cam_manager, T_tcp_cam, scan_config):
    """
    Move in a straight line for cleaner data verification.
    """
    # Get parameters from config
    start_pos = np.array(scan_config.start_point)
    end_pos = np.array(scan_config.end_point)
    num_steps = scan_config.num_waypoints
    
    # Neutral orientation
    _, neutral_orn = robot.get_tcp_pos_orn()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"task_line_scan_{timestamp}"
    task_dir = os.path.join(INPUT_PATH, task_name)
    
    print(f"===== Starting Line Scan =====")
    print(f"    Task Name: {task_name}")
    print(f"    Start: {start_pos}")
    print(f"    End:   {end_pos}")
    
    print("Moving to Start...")
    robot.move_cart_pos_abs_lin(start_pos, neutral_orn)
    time.sleep(2.0)
    
    print("Starting Scan...")
    for i in range(num_steps):
        # Linear Interpolation (Lerp)
        t = i / (num_steps - 1)
        target_pos = (1 - t) * start_pos + t * end_pos
        
        robot.move_cart_pos_abs_lin(target_pos, neutral_orn)
        time.sleep(1.5) # Generous wait for settling
        
        if cam_manager:
            actual_pos, actual_orn = robot.get_tcp_pos_orn()
            data = cam_manager.get_images()
            data_point = {
                'rgb': data['rgb_gripper'],
                'depth': data['depth_gripper'],
                'tcp_pos': actual_pos,
                'tcp_orn': actual_tcp_orn
            }
            save_frame(data_point, i+1, task_dir, cam_manager, T_tcp_cam)
            cam_manager.render()
            
    print(f"Line scan completed. Saved to {task_dir}")

def random_move(robot, cam_manager, T_tcp_cam, sampler_config):
    """
    Moves the robot to random points using the safe sampler.
    """
    num_poses = sampler_config.num_poses
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"task_random_{timestamp}"
    task_dir = os.path.join(INPUT_PATH, task_name)
    
    _, neutral_orn = robot.get_tcp_pos_orn()
    
    print(f"===== Starting Random Sampler =====")
    print(f"    Task Name: {task_name}")
    print(f"    Poses to collect: {num_poses}")
    print(f"    Box Limits: X{sampler_config.x_limits} Y{sampler_config.y_limits} Z{sampler_config.z_limits}")

    success_count = 0
    attempts = 0
    max_attempts = num_poses * 2 

    while success_count < num_poses and attempts < max_attempts:
        attempts += 1
        print(f"Sampling attempt {attempts} (Collected {success_count}/{num_poses})...")
        
        target_pos, target_orn = sample_pose(sampler_config, neutral_orn)
        
        try:
            # Move Robot
            robot.move_cart_pos_abs_ptp(target_pos, target_orn)
            
            time.sleep(0.5) 
            
            # Capture & Save
            if cam_manager is not None:
                # Match calibration approach: use actual robot pose after movement (in NE frame)
                actual_tcp_pos, actual_tcp_orn = robot.get_tcp_pos_orn()
                
                data = cam_manager.get_images()
                data_point = {
                    'rgb': data['rgb_gripper'],
                    'depth': data['depth_gripper'],
                    'tcp_pos': actual_tcp_pos,
                    'tcp_orn': actual_tcp_orn
                }
                save_frame(data_point, success_count + 1, task_dir, cam_manager, T_tcp_cam)
                cam_manager.render()
                
            success_count += 1
            
        except Exception as e:
            print(f"[WARNING] Pose unreachable. Skipping. Error: {e}")
            continue
            
    print(f"Random sampler completed. Captured {success_count} frames.")

# DEBUG/UTILITY FUNCTIONS
def print_robot_position(robot, label: str = "Current"):
    """Print the current robot TCP position and orientation."""
    pos, orn = robot.get_tcp_pos_orn()
    print(f"\n{label} Position:")
    print(f"  X = {pos[0]:.4f} m")
    print(f"  Y = {pos[1]:.4f} m")
    print(f"  Z = {pos[2]:.4f} m")
    print(f"  Orientation (quaternion): [{orn[0]:.3f}, {orn[1]:.3f}, {orn[2]:.3f}, {orn[3]:.3f}]")
    return pos, orn

# MAIN EXECUTION
@hydra.main(
    config_path="/home/ceylanb/DreMa/drema_project/configs/data_gathering",
    config_name="data_gather_task",
    version_base=None
)
def main(cfg: DictConfig):
    """
    Main entry point for data gathering.
    """
    robot = None
    cam_manager = None
    T_tcp_cam = None

    arch_config = cfg.movement
    sampler_config = cfg.pose_sampler
    line_scan_config = cfg.line_scan
    movement_type = cfg.movement_type # Configure on configs/data_gathering/data_gather_task.yaml
    
    try:
        # Initialize robot
        print("Initializing robot")
        robot = hydra.utils.instantiate(cfg.robot)
        print("Robot initialized successfully")

        print("Initializing Camera Manager from config...")
        cam_manager = hydra.utils.instantiate(cfg.cams)
        print("Camera Manager initialized.")

        print("Loading calibration...")
        T_tcp_cam = np.load(cfg.calibration_file)
        print("Calibration Matrix Loaded Successfully.")

        print("Robot positioned to neutral pose")
        recover_to_center(robot)
        

        print("Data Gathering starting")
        
        print_robot_position(robot, "Initial")
        if movement_type == "pose_sampler":
            # Random pose sampling start
            time.sleep(1)
            random_move(robot, cam_manager, T_tcp_cam, sampler_config)
            time.sleep(1)
             # Random pose sampling end
        elif movement_type == "line_scan":
            # Line scan start
            time.sleep(1)
            line_scan_move(robot, cam_manager, T_tcp_cam, line_scan_config)
            time.sleep(1)
            # Line scan end
        if movement_type == "arch_parallel":
            arch_type = "parallel"
            # Movement sequence start
            time.sleep(1)
            arch_move(robot, cam_manager, T_tcp_cam, arch_config, arch_type=arch_type)
            time.sleep(1)
            # Movement sequence end
        elif movement_type == "arch_skewed_clockwise":
            arch_type = "skewed_clockwise"
            # Movement sequence start
            time.sleep(1)
            arch_move(robot, cam_manager, T_tcp_cam, arch_config, arch_type=arch_type)
            time.sleep(1)
            # Movement sequence end
        elif movement_type == "arch_skewed_counterclockwise":
            arch_type = "skewed_counterclockwise"
            # Movement sequence start
            time.sleep(1)
            arch_move(robot, cam_manager, T_tcp_cam, arch_config, arch_type=arch_type)
            time.sleep(1)
            # Movement sequence end
        
        print_robot_position(robot, "Final")

        print("Data Gathering completed")

        print("Robot positioned to neutral pose")
        recover_to_center(robot)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        try:
            print("Moving robot to safe position...")
            recover_to_center(robot)
        except Exception as recovery_error:
            print(f"Recovery error: {recovery_error}")
    finally:
        if robot is not None:
            try:
                robot.abort_motion()
                print("Robot stopped and recovered safely.")
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

if __name__ == "__main__":
    main()