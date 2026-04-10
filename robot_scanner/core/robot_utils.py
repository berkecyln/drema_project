import time


def recover_to_center(robot):
    print("Moving to neutral pose...")
    robot.move_to_neutral()
    robot.get_tcp_pos_orn()
    print("Neutral pose reached.")
    time.sleep(1)


def print_robot_position(robot, label="Current"):
    pos, orn = robot.get_tcp_pos_orn()
    print(f"\n{label} Position:")
    print(f"  X={pos[0]:.4f}  Y={pos[1]:.4f}  Z={pos[2]:.4f}")
    print(f"  Orientation: {[round(float(v), 3) for v in orn]}")
    return pos, orn
