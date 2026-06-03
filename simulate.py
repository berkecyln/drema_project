import hydra
import cv2
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf

from drema.environment.builder import Builder
from drema.utils.utils import prepare_depth, LoopFrequencyLogger
import drema.utils.keylistner_utils as key_listener


# use hydra for configuration
@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config_real")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # create the builder from the configuration
    builder = Builder(cfg)

    # load trajectory
    trajectory = builder.load_trajectory() 

    # create the cameras
    camera_manager = builder.create_cameras(trajectory)

    # create the environment
    env = builder.create_environment(trajectory)

    # bu the environment
    env.build_environment()

    frequency_logger = LoopFrequencyLogger(log_interval=1.0)

    orbit_yaw = 0
    orbit_writer = None
    if cfg.simulation.visualization.orbit_camera and cfg.simulation.visualization.record_orbit_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        orbit_writer = cv2.VideoWriter(
            cfg.simulation.visualization.orbit_video_path,
            fourcc, 30, (640, 480)
        )

    #objs, bodies = visualize_trajectory(env.client, env.trajectory)
    key_listener.start_listener()

    # simulation loop
    while True:

        if cfg.simulation.visualization.pybullet_camera:
            rotation, translation = env.get_pybullet_camera(np.array(cfg.simulation.visualization.transform_matrix))
            camera_manager.update_visualization_camera_extrinsics(rotation, translation)

        # get the pressed key
        key = key_listener.get_pressed_key()
        if key:
            if key == "Key.esc":  # Exit on 'esc'
                print("Exiting...")
                break
            elif key == 'r':
                print("Resetting environment...")
                env.reset()

            if cfg.simulation.visualization.visualize and not cfg.simulation.visualization.pybullet_camera:

                if key in ['Key.right', 'd', 'Key.up']:
                    camera_manager.get_next_visualization_camera()
                elif key in ['Key.down', 'Key.left', 'a']:
                    camera_manager.get_previous_visualization_camera()


        # step the simulation
        execution_results = env.step()

        if execution_results == 1:

            # step the simulator as the base environment
            for _ in range(20):
                env.step()

            env.next_waypoint()

        # update the environment
        env.update_state()

        if cfg.simulation.visualization.orbit_camera:
            import pybullet
            orbit_yaw += cfg.simulation.visualization.orbit_camera_speed
            view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cfg.simulation.visualization.orbit_camera_target,
                distance=cfg.simulation.visualization.orbit_camera_distance,
                yaw=orbit_yaw,
                pitch=cfg.simulation.visualization.orbit_camera_pitch,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = pybullet.computeProjectionMatrixFOV(
                fov=60, aspect=640/480, nearVal=0.01, farVal=10.0
            )
            _, _, orbit_rgb, _, _ = pybullet.getCameraImage(
                640, 480, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=pybullet.ER_TINY_RENDERER
            )
            orbit_frame = cv2.cvtColor(
                np.array(orbit_rgb, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3],
                cv2.COLOR_RGB2BGR
            )
            cv2.imshow("Orbit View", orbit_frame)
            cv2.waitKey(1)
            if orbit_writer is not None:
                orbit_writer.write(orbit_frame)
            if orbit_yaw >= 360 * cfg.simulation.visualization.orbit_loops:
                orbit_writer.release()
                print(f"Orbit recording done. Saved to: {cfg.simulation.visualization.orbit_video_path}")
                orbit_writer = None
                break

        # if the simulation is visualized
        if cfg.simulation.visualization.visualize:

            if cfg.simulation.trajectory.update_wrist_camera and cfg.simulation.robot.simulate_robot:
                # get the wrist camera position on the robot
                translation, rotation = env.get_wrist_camera_extrinsics()
                camera_manager.update_camera_extrinsics("wrist", rotation, translation)

            # get visualization camera
            camera = camera_manager.get_visualization_camera()

            # render the environment
            rgb, depth = env.render_cameras([camera], filter_depth=cfg.simulation.output.filter_depth, radius_filter=cfg.simulation.output.radius_filter, threshold=cfg.simulation.output.threshold)

            # invert the rbg channels and prepare depth for visualization
            rgb = cv2.cvtColor(rgb[0], cv2.COLOR_BGR2RGB)
            depth = prepare_depth(depth[0])

            # show the rgb image
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth)
            cv2.waitKey(1)

        # log the frequency
        frequency_logger.log_frequency()

    if orbit_writer is not None:
        orbit_writer.release()

if __name__ == "__main__":
    main()
