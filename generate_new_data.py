import itertools
import os.path
from array import array

import hydra
import cv2
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf

from drema.environment.augmentations import AugmentationManager
from drema.environment.builder import Builder
from drema.utils.generation_utils import save_images
from drema.utils.utils import prepare_depth, LoopFrequencyLogger



def simulation_loop(environment, camera_manager, cfg, frequency_logger):
    images = []

    while True:
        # step the simulation
        execution_results = environment.step()

        if execution_results != 0:
            if execution_results == -1 and environment.current_waypoint_is_keypoint():
                objects_positions = [position for position, orientation in environment.observe_state()]

                return objects_positions, images

            # step the simulator as the base environment
            for _ in range(20):
                environment.step()

            # if save the images then render the images
            if cfg.simulation.generation.save_images:
                if cfg.simulation.trajectory.update_wrist_camera:
                    # get the wrist camera position on the robot
                    translation, rotation = environment.get_wrist_camera_extrinsics()
                    camera_manager.update_camera_extrinsics("wrist", rotation, translation)

                    # write per-step T_w2c into the trajectory so saved pkl has correct extrinsics
                    R_w2c = rotation.T
                    t_w2c = -R_w2c @ translation
                    T_w2c_step = np.eye(4)
                    T_w2c_step[:3, :3] = R_w2c
                    T_w2c_step[:3, 3] = t_w2c
                    environment.trajectory.demo[environment.waypoint_index]["wrist_camera_extrinsics"] = T_w2c_step
                
                # get the camera names and objects
                names, cameras = camera_manager.get_simulation_cameras()
                names = [name for name in names]

                rgbs, depths = environment.render_cameras(cameras, filter_depth=cfg.simulation.output.filter_depth, radius_filter=cfg.simulation.output.radius_filter, threshold=cfg.simulation.output.threshold)
                images.append([names, rgbs, depths])

            # update the trajectory
            #environment.update_trajectory_with_current_data()

            environment.next_waypoint()

            # if the trajectory waypoint is 0, break the loop
            if environment.waypoint_index == 0:

                objects_positions = [position for position, orientation in environment.observe_state()]

                return objects_positions, images


        # update the environment
        environment.update_state()

        # if the simulation is visualized
        if cfg.simulation.visualization.visualize:

            if cfg.simulation.trajectory.update_wrist_camera:
                # get the wrist camera position on the robot
                translation, rotation = environment.get_wrist_camera_extrinsics()
                camera_manager.update_camera_extrinsics("wrist", rotation, translation)

            # get visualization camera
            camera = camera_manager.get_visualization_camera()

            # render the environment
            rgb, depth = environment.render_cameras([camera], filter_depth=cfg.simulation.output.filter_depth, radius_filter=cfg.simulation.output.radius_filter, threshold=cfg.simulation.output.threshold)

            # invert the rbg channels and prepare depth for visualization
            rgb = cv2.cvtColor(rgb[0], cv2.COLOR_BGR2RGB)
            depth = prepare_depth(depth[0])

            # show the rgb image
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth)

            # wait for a key
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key in [82, 83, 100]:
                camera_manager.get_next_visualization_camera()
            elif key in [81, 84, 97]:
                camera_manager.get_previous_visualization_camera()
            # if a press the "r" key, reset the environment
            elif key == ord('r'):
                environment.reset()

        # log the frequency
        frequency_logger.log_frequency()




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

    # build the environment
    env.build_environment()

    # create the augmentation manager
    augmentation = AugmentationManager(env)

    frequency_logger = LoopFrequencyLogger(log_interval=1.0)

    # execute original trajectory
    final_positions_original_trajectory, images = simulation_loop(env, camera_manager, cfg, frequency_logger)

    print("Final positions: ", final_positions_original_trajectory)
    base_name = os.path.basename(cfg.data.source_path)
    output_path = os.path.join(cfg.simulation.generation.generated_data_path, base_name + "_episode0_start")
    save_images(os.path.join(output_path, "episode0"), images, cfg.data.source_path)
    # save the augmented trajectory
    trajectory.save(os.path.join(output_path, "episode0", "generated_trajectory.pkl"))

    # augment the data

    # save the images
    episode = 1
    if cfg.simulation.generation.translate_environment:
        # remove the 0 translation
        translation_values = cfg.simulation.generation.translation_values
        xy_translations = np.array(list(itertools.product(translation_values, repeat=2)))
        xy_translations = xy_translations[1:]
        translations = np.concatenate((xy_translations, np.zeros((xy_translations.shape[0], 1))), axis=1)

        for translation in translations:
            env.reset()
            final_positions_augmented_trajectory = augmentation.translate_environment(translation, final_positions_original_trajectory)
            final_positions, images = simulation_loop(env, camera_manager, cfg, frequency_logger)
            if np.max(np.linalg.norm(np.array(final_positions) - np.array(final_positions_augmented_trajectory), axis=1)) < cfg.simulation.generation.threshold:
                print("Saving episode: ", episode)
                save_images(os.path.join(output_path, "episode" +str(episode)), images, cfg.data.source_path)

                # save the augmented trajectory
                trajectory.save(os.path.join(output_path, "episode" + str(episode), "generated_trajectory.pkl"))

                episode += 1
            else:
                print("Error in the final positions")
                print(np.linalg.norm(np.array(final_positions) - np.array(final_positions_augmented_trajectory), axis=1))

    episode += 1000
    if cfg.simulation.generation.rotate_environment:

        rot_min = cfg.simulation.generation.get('rotate_environment_min', 0)
        rot_max = cfg.simulation.generation.get('rotate_environment_max', 360)
        rotations = range(rot_min, rot_max, cfg.simulation.generation.rotate_environment_step)

        for rotation in rotations:

            if rotation == 0:
                continue

            env.reset()
            final_positions_augmented_trajectory = augmentation.rotate_environment(np.array(cfg.simulation.generation.rotation_center), rotation, final_positions_original_trajectory)
            final_positions, images = simulation_loop(env, camera_manager, cfg, frequency_logger)
            if np.max(np.linalg.norm(np.array(final_positions) - np.array(final_positions_augmented_trajectory), axis=1)) < cfg.simulation.generation.threshold:
                print("Saving episode: ", episode)
                save_images(os.path.join(output_path, "episode" +str(episode)), images, cfg.data.source_path)

                # save the augmented trajectory
                trajectory.save(os.path.join(output_path, "episode" +str(episode), "generated_trajectory.pkl"))

                episode += 1
            else:
                print("Error in the final positions")
                print(np.linalg.norm(np.array(final_positions) - np.array(final_positions_augmented_trajectory), axis=1))

    episode += 1000
    if cfg.simulation.generation.rotate_objects:

        rotations = range(-180, 180, cfg.simulation.generation.rotation_objects_step)

        for rotation in rotations:

            if rotation == 0:
                continue

            env.reset()
            final_positions_augmented_trajectory = augmentation.rotate_objects(rotation, final_positions_original_trajectory)
            final_positions, images = simulation_loop(env, camera_manager, cfg, frequency_logger)
            if np.max(np.linalg.norm(np.array(final_positions) - np.array(final_positions_augmented_trajectory), axis=1)) < cfg.simulation.generation.threshold:
                print("Saving episode: ", episode)
                save_images(os.path.join(output_path, "episode" +str(episode)), images, cfg.data.source_path)

                # save the augmented trajectory
                trajectory.save(os.path.join(output_path, "episode" + str(episode), "generated_trajectory.pkl"))

                episode += 1
            else:
                print("Error in the final positions")
                print(np.linalg.norm(np.array(final_positions) - np.array(final_positions_augmented_trajectory), axis=1))

if __name__ == "__main__":
    main()