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
from dataclasses import dataclass
import numpy as np
import hydra
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation


from robot_io.robot_interface.panda_franky_interface import NE_T_EE 

# CONFIGURATION PARAMETERS

@dataclass
class MovementConfig:
    """Configuration for robot movement parameters.
    
    Adjust these values based on your workspace and safety requirements.
    All distances are in meters.
    """
    # Movement amount
    MOVE: float = 0.05  # 5cm to each side
    
    # Arch movement parameters
    START, END, CORNER = None, None, None # Parallel to y axis arch
    START2, END2 = None, None # Skewed reverseclockwise arch
    START3, END3 = None, None # Skewed clockwise arch
    INITIAL_POS, INITIAL_ORN = None, None

    FLIP = False          # Whether to flip orientation 180 degrees around Y-axis
    NUM_WAYPOINTS: int = 150        # Number of points along the arch
# Global config instance
CONFIG = MovementConfig()

# MOVEMENT FUNCTIONS

def move(robot, direction: str):
    """Move the robot in X, Y, Z axis of the workspace.
    X -> +X for forward, -X for backward
    Y -> +Y for right, -Y for left
    Z -> +Z for up, -Z for down
    
    Args:
        robot: PandaFrankYInterface instance
        direction: 'left', 'right', 'up', 'down', 'forward', 'backward'
    """
    if direction.lower() not in ['left', 'right', 'up', 'down', 'forward', 'backward']:
        print(f"Invalid direction: {direction}")
        return False

    print("====================")
    print(f"MOVING {direction.upper()}")
    print("====================")
    
    # Get current position and orientation
    current_pos, current_orn = robot.get_tcp_pos_orn()
    print(f"Current position: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
    
    # Calculate target position based on direction
    target_pos = current_pos.copy()
    if direction == 'forward':
        target_pos[0] += CONFIG.MOVE
    elif direction == 'backward':
        target_pos[0] -= CONFIG.MOVE
    elif direction == 'left':
        target_pos[1] += CONFIG.MOVE    
    elif direction == 'right':
        target_pos[1] -= CONFIG.MOVE
    elif direction == 'up':
        target_pos[2] += CONFIG.MOVE
    elif direction == 'down':
        target_pos[2] -= CONFIG.MOVE

    print(f"Target position:  X={target_pos[0]:.3f}, Y={target_pos[1]:.3f}, Z={target_pos[2]:.3f}")
    
    # Execute movement
    print("Executing movement...")
    robot.move_cart_pos_abs_ptp(target_pos, current_orn)

    print("Movement completed")

def gripper(robot, action: str):
    """Control the robot gripper to open or close.
    
    Args:
        robot: PandaFrankYInterface instance
        action: 'open' or 'close'
    """
    if action not in ['open', 'close']:
        print(f"Invalid gripper action: {action}")
        return
    
    print("====================")
    print(f"GRIPPER {action.upper()}")
    print("====================")

    # Execute gripper action
    print(f"Executing gripper action: {action}...")
    print(f"Initial Width: {robot.gripper.width*1000:.1f}mm")
    if action == 'open':
        robot.open_gripper(blocking=True)
    elif action == 'close':
        robot.close_gripper(blocking=True)
    print(f"End Width: {robot.gripper.width*1000:.1f}mm")
    
    print("Gripper action completed")   

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

def create_bezier_points(robot) -> tuple:
    """
    Create anchot points for the Bézier curve based on the robot's initial position.
    
    Args:
        robot: PandaFrankYInterface instance
    
    Returns:
        START: np.array, start point of the arch
        END: np.array, end point of the arch
        CORNER: np.array, control point for the arch
        initial_pos: list, initial position of the robot TCP
        initial_orn: list, initial orientation of the robot TCP
    """
    initial_pos, initial_orn = robot.get_tcp_pos_orn()

    # Flip initial orientation 180° around Y-axis if FLIP is enabled 
    # Has some  issues it crashes to limit need to check
    if CONFIG.FLIP:
        initial_rot = Rotation.from_quat(initial_orn)
        flip_rotation = Rotation.from_euler('y', 180, degrees=True)
        flipped_rot = initial_rot * flip_rotation
        initial_orn = flipped_rot.as_quat()
        print(f"Flipped initial orientation 180° around Y-axis")
    
    start_pos = initial_pos.copy()
    START = np.array([start_pos[0] , start_pos[1] - 0.4, start_pos[2] - 0.3])  # Start point of the arch (X, Y, Z)
    end_pos = initial_pos.copy()
    END = np.array([end_pos[0], end_pos[1] + 0.3, end_pos[2] - 0.3]) # End point of the arch (X, Y, Z)
    corner_pos = initial_pos.copy()
    CORNER = np.array([corner_pos[0]- 0.1, corner_pos[1], corner_pos[2] + 0.4])  # Puller point for the arch (X, Y, Z)

    
    START2 = np.array([start_pos[0] + 0.1, start_pos[1] - 0.3, start_pos[2] - 0.3])
    END2 = np.array([end_pos[0] - 0.1, end_pos[1] + 0.3, end_pos[2] - 0.3])             

    START3 = np.array([start_pos[0] +  0.1, start_pos[1] + 0.3, start_pos[2] - 0.3]) 
    END3 = np.array([end_pos[0] - 0.1, end_pos[1] - 0.4, end_pos[2] - 0.3])

    return START, END, CORNER, START2, END2, START3, END3, initial_pos , initial_orn

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

def calculate_tangent(t: float, start: np.array, corner: np.array, end: np.array) -> np.array:
    """Calculate the tangent vector at parameter t on a Quadratic Bézier Curve.
        We are taking derivative of:
        P(t) = (1-t)^2 * Start + 2(1-t)t * Corner + t^2 * End
        Which gives us:
        P'(t) = 2(1-t)(Corner - Start) + 2 t(End - Corner)
        
    Args:
        t: float, parameter along the curve (0 <= t <= 1)
        start: np.array, start point of the curve
        corner: np.array, control point of the curve
        end: np.array, end point of the curve
    
    Returns:
        tangent: np.array, normilized tangent vector at point t
    """
    tangent = 2 * (1 - t) * (corner - start) + 2 * t * (end - corner)
    tangent_norm = tangent / np.linalg.norm(tangent)

    print(f"Calculated tangent at t={t}: [{tangent_norm[0]:.3f}, {tangent_norm[1]:.3f}, {tangent_norm[2]:.3f}]")

    return tangent_norm

def get_orientation(position, t, start, end, corner, max_tilt) -> np.array:
    """Get orientation quaternion that tilts toward center while maintaining gripper roll.
    
    This creates a "nodding" motion (tilt up/down and left/right) without twisting
    the gripper around its axis.
    
    Args:
        position: Current position [x, y, z]
        t: Parameter along curve (0 to 1)
    
    Returns:
        orn: Orientation quaternion [x, y, z, w]
    """
    center = calculate_center(start, end)
    
    # Calculate direction to center
    look_direction = center - position
    look_direction = look_direction / np.linalg.norm(look_direction)
    
    # Get initial gripper orientation 
    initial_rot = Rotation.from_quat(CONFIG.INITIAL_ORN)
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
            return CONFIG.INITIAL_ORN

    
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

def generate_arch_path(robot, start, end, corner) -> tuple:
    """
    Generate arch path based on Quadratic Bézier Curve from start to end point.

    Args:
        robot: PandaFrankYInterface instance
    
    Returns:
        path: list of np.array, points along the arch path
    
    """

    # P(t) = (1-t)^2 * Start + 2(1-t)t * Corner + t^2 * End
    path = []
    waypoints = CONFIG.NUM_WAYPOINTS

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
        orientation = get_orientation(np.array([x_t, y_t, z_t]), t, start, end, corner, max_tilt)
        
        #orientation = CONFIG.INITIAL_ORN
        path.append(([x_t, y_t, z_t], orientation))
    
    print("Arch path generation completed.")
    for p in path: print(f"Pos: {np.round(p[0], 3)}, Ori: {p[1]}")
    
    return path

def arch_move(robot, start=None, end=None, corner=None):
    """Move the robot along an arch path defined by a Bezier curve.
    
    Args:
        robot: PandaFrankYInterface instance
    """
    path = generate_arch_path(robot, start, end, corner)
    positions = [point for point, _ in path]
    orientations = [orn for _, orn in path]

    

    print("Moving to Start point of the arch...")
    robot.move_cart_pos_abs_ptp(positions[0], orientations[0])
    curr_pos, curr_orn = robot.get_tcp_pos_orn()
    print(f"Start point reached: X={curr_pos[0]:.3f}, Y={curr_pos[1]:.3f}, Z={curr_pos[2]:.3f}, Ori={curr_orn}")

    time.sleep(2)

    print("Starting arch movement...")
    #for idx, (point, orientation) in enumerate(path):
        #print(f"Moving to waypoint {idx + 1}/{len(path)}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}, Ori={orientation}")
        #robot.move_cart_pos_abs_ptp(point, orientation)
    robot.move_cart_waypoints(positions[1:], orientations[1:])
    print("Arch movement completed.")

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

def debug_transformations(robot):
    """Debug the transformation matrices."""
    print("\n" + "="*60)
    print("TRANSFORMATION DEBUG")
    print("="*60)
    
    # Get F_T_NE from robot
    F_T_NE = robot.robot.state.F_T_NE.matrix
    print("\nF_T_NE (Flange to Nominal End-Effector):")
    print(F_T_NE)
    
    # Get NE_T_EE from the interface file
    print("\nNE_T_EE (Nominal End-Effector to End-Effector):")
    print(NE_T_EE.matrix)
    
    # Get current TCP pose
    current_pose = robot.get_tcp_pose()
    print("\nCurrent TCP Pose:")
    print(current_pose)
    
    # Get current orientation
    pos, orn = robot.get_tcp_pos_orn()
    print(f"\nCurrent Orientation (quaternion): {orn}")
    
    # Check if IK solver has the right F_T_NE
    if hasattr(robot, 'ik_solver'):
        if hasattr(robot.ik_solver, 'NE_T_NE_ikfast'):
            print("\nIK Solver NE_T_NE_ikfast:")
            print(robot.ik_solver.NE_T_NE_ikfast)
    
    print("="*60 + "\n")

# MAIN EXECUTION

@hydra.main(
    config_path="/home/ceylanb/robot/robot_io/robot_io/conf",
    config_name="replay_recorded_trajectory",
    version_base=None
)
def main(cfg: DictConfig):
    """
    Main entry point for data gathering.
    """
    robot = None
    
    try:
        # Initialize robot
        print("Initializing robot")
        robot = hydra.utils.instantiate(cfg.robot)
        print("Robot initialized successfully")

        print("Robot positioned to neutral pose")
        recover_to_center(robot)
        

        print("Movement sequence starting")
        
        print_robot_position(robot, "Initial")
        
        #debug_transformations(robot)
        
        # Movement sequence start
        CONFIG.START, CONFIG.END, CONFIG.CORNER, CONFIG.START2, CONFIG.END2, CONFIG.START3, CONFIG.END3, CONFIG.INITIAL_POS, CONFIG.INITIAL_ORN = create_bezier_points(robot)
        time.sleep(1)

        # Parallel to y axis arch
        arch_move(robot, CONFIG.START, CONFIG.END, CONFIG.CORNER)
        time.sleep(1)

        recover_to_center(robot)
        time.sleep(1)

        # Skewed reverseclockwise arch
        arch_move(robot, CONFIG.START2, CONFIG.END2, CONFIG.CORNER)
        time.sleep(1)

        recover_to_center(robot)
        time.sleep(1)
        # Skewed clockwise arch
        arch_move(robot, CONFIG.START3, CONFIG.END3, CONFIG.CORNER)

        time.sleep(1)
        # Movement sequence end
    
        #debug_transformations(robot)

        #print_robot_position(robot, "Final")

        print("Movement sequence completed")

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