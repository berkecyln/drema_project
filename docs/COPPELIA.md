
# PerAct data generation
Here you can find the information about the data generation process for the PerAct dataset.

## Installation
You can use a different environment for this part. We suggest to use a different directory to clone this external code and using a different environment than DreMa's since it may have compatibility issues with the versions of the libraries.
(e.g. you can crete a third_party directory and clone the following code there)

This part was tested on Ubuntu 20.04 with Python 3.8. The code may work with other versions of Python and other operating systems, but it was not tested.

### 1 Download Coppelia 4.1

Download CoppeliaSim Edu 4.1 from the following link: https://www.coppeliarobotics.com/previousVersions#

You need to set the following environment variable:
```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```
you can also add them to the .bashrc file to make them permanent (remember to source your bashrc -> source ~/.bashrc) .

### 2 install PyRep

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
cd ..
```

### 3 Install the forked version of RLBench

```bash
git clone https://github.com/leobarcellona/RLBench.git
cd RLBench
pip install -r requirements.txt
python setup.py develop
cd ..
```

### 4 Install the forked version of YARR

```bash
git clone https://github.com/leobarcellona/YARR.git
cd YARR
pip install -r requirements.txt
python setup.py develop
pip install packaging==21.3 dotmap pyhocon wandb chardet opencv-python-headless gpustat ipdb visdom sentencepiece termcolor
```

## Generate the data of the tasks
For generating the training data for the tasks, you can use the following command:
```bash
python RLBench/tools/prepare_data_for_drema.py --tasks=TASK \
                            --save_path="PATH_WHERE_TO_SAVE_DATA" \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=5 \
                            --processes=1 \
                            --all_variations=True

```

The code will generate two directories in the desired path:
- <b>data_data</b>: containing the data structured needed by DreMa
- <b>TASK</b>: containing the data structured needed by PerAct

<b>We tested the following tasks</b>: close_jar, insert_onto_square_peg_larger, pick_and_lift, pick_up_cup, place_shape_in_shape_sorter,
place_wine_at_rack_location, put_groceries_in_cupboard, new_slide_block_to_color_target, stack_blocks

## (Optional) Visualize the generated data

Install additional requirements
```bash
pip install matplotlib
pip install open3d
```
Use the following code to visualize the generated data:
```commandline
python RLBench/tools/visualize_data.py --data_path="PATH_TO_PERACT_DATA" --task="TASK" --episode=0
```

## 5 (Optional) install PerAct

To use the data generation to train PerAct, you can install PerAct following the instructions in the original repository.
https://github.com/peract/peract
If you use the same environment for PerAct ad DreMa there could be compatibility issues with the versions of the libraries.
In the experiment we used different environments for PerAct and DreMa.

### Prepare the data for PerAct
If you generate new demonstrations with Drema, you can use the following command to prepare the data for PerAct:
```bash
python RLBench/tools/prepare_data_for_peract.py \
                            --original_path "PATH_TO_RLBENCH_DATA" \
                            --generated_path "PATH_TO_GENERATED_DATA" \
                            --output_path "PATH_WHERE_TO_SAVE_DATA" \
                            --scenes TASKS \
```

### Validation and test data
Here is a temporary link to download the validation and test data used in the paper:

[Validation and test data](https://amsuni-my.sharepoint.com/:f:/g/personal/l_barcellona_uva_nl/Et9um-8BkAFHsdl9JqVrSFoB5EMBQ1tAAYPv4eSnUK4fSA?e=vH6sqv)

If the link is not working, please contact us by email.
