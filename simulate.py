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
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # create the builder from the configuration
    builder = Builder(cfg)

    # load trajectory
    #BN: loads singles shot recording recoreded by us
    #BN: can be disabled by making simulation.trajectory.load_trajectory: False in config file
    trajectory = builder.load_trajectory() 

    # create the cameras
    camera_manager = builder.create_cameras(trajectory)

    # create the environment
    env = builder.create_environment(trajectory)

    # bu the environment
    env.build_environment()

    frequency_logger = LoopFrequencyLogger(log_interval=1.0)

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

        # if the simulation is visualized
        if cfg.simulation.visualization.visualize:

            if cfg.simulation.trajectory.update_wrist_camera:
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

if __name__ == "__main__":
    main()
