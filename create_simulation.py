import os.path

import hydra

from omegaconf import DictConfig, OmegaConf

from drema.environment.assets.object import Object
from drema.r2s_builder.assets_extractor import AssetsManager
from drema.r2s_builder.gaussians_optimizers.depth_optimizer import DepthTrainer
from drema.r2s_builder.gaussians_optimizers.surf_depth_optimizer import SurfDepthTrainer
from drema.r2s_builder.gaussians_optimizers.surf_optimizer import SurfTrainer
from drema.r2s_builder.gaussians_optimizers.base_optimizer import BaseTrainer
from drema.utils.coppelia_utils import read_labels

#BN: go over labels.txt and seperate robot, table and object labels
def prepare_labels(path):
    labels = read_labels(path, filter_labels=False) #BN: create dict [name: number]

    #BN: this filtering occurs if we genereate data over RLBench scenes
    #BN: if we bring our own scenes, we create labels.txt manually so these will be not here
    filter_names = ["DefaultCamera", "ResizableFloor", "workspace", "Wall"]
    # remove labels containing the filter names
    labels = {k: v for k, v in labels.items() if not any(name in k for name in filter_names)}

    counter = 1
    print(labels)
    Panda_labels = {}
    Object_labels = {}
    Table_labels = {}
    # iterate over the labels
    for label_name, value in labels.items():
        # check if the label contain the Panda string
        if "Panda" in label_name:
            label_name = label_name.split("_")[1]
            if label_name == "gripper":
                label_name = "link8"
            elif label_name == "leftfinger":
                label_name = "link10"
            elif label_name == "rightfinger":
                label_name = "link9"
            Panda_labels[label_name] = value
        elif label_name in ["table", "diningTable_visible"]:
            Table_labels[label_name] = value
        else:
            # check if label_name is in keys
            if label_name in Table_labels.keys():
                label_name = label_name + "_"+str(counter)
                counter += 1
            Object_labels[label_name] = value

    return Panda_labels, Object_labels, Table_labels




@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Building the simulation")

    source_path = cfg.data.source_path
    assets_path = cfg.data.assets_path

    dataset = cfg.training.model #BN: configs/training/coppelia_params.yaml
    pipeline = cfg.training.pipeline
    optimization = cfg.training.optimization

    assets = cfg.training.assets

    gaussians_iterations = assets.gaussians_iterations
    mesh_iterations = optimization.iterations

    #BN: Basically we cannot use non-depth trainers since table extraction requires depth
    if assets.use_original_guassians:
        if assets.use_depth:
            trainer = DepthTrainer #BN: Original Gaussian with depth
        else:
            trainer = BaseTrainer #BN: Original Gaussian
    else:
        if assets.use_depth:
            trainer = SurfDepthTrainer #BN: 
        else:
            trainer = SurfTrainer

    assets_manager = AssetsManager(source_path, assets_path, trainer, dataset, optimization, pipeline, gaussians_iterations, mesh_iterations)

    # Load the data
    assets_manager.load_data()

    # Extract Environment
    if assets.extract_gaussians_environment:
        assets_manager.extract_environment(assets.extract_mesh_environment)

    # get labels
    panda_labels, object_labels, table_labels = prepare_lables(os.path.join(source_path, "labels.txt"))

    # if it needs to extract the table
    if assets.extract_table:
        # extract the table
        # TODO: extract table even without depth
        assert assets.use_depth, "Extracting objects requires depth to extract the table in the current implementation"
        # get first value of the table labels
        table_labels = list(table_labels.values())[0]
        assets_manager.extract_table(table_labels)
    else:
        # load the table
        assets_manager.load_table()

    if assets.extract_gaussians_objects: #BN: <======== I STAYED HERE
        for label_name, value in object_labels.items():
            # extract the object
            assets_manager.extract_asset(value, extract_mesh=assets.extract_mesh_objects, extract_urdf=assets.extract_urdf_objects)

    if assets.extract_gaussians_robot:
        for label_name, value in panda_labels.items():
            # extract the object
            assets_manager.extract_asset(value, extract_mesh=False, extract_urdf=False)

            # move the output to the correct folder
            src = os.path.join(source_path, "output", "objects_ply", str(value) + ".ply")
            dst = os.path.join(source_path, "output", "robot", label_name + ".ply")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)

    if assets.filter_objects_gaussians_environment:
        assets_manager.filter_environment()


if __name__ == "__main__":
    main()
