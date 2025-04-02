## Install

### Setup repository and submodules

```bash
git clone https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar.git
git submodule init
git submodule update
```

### Setup environment

Using miniforge is required for setting up ROS. Follow instructions in [miniforge repo].(https://github.com/conda-forge/miniforge) to install miniforge.

Then do the following:

``` bash
conda create --name mapping_ros2 python=3.11
conda activate mapping_ros2
```

To make ROS accessible in the environment, install ROS/ROS2 using robostack.

```bash

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-jazzy

# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

# Install ros-noetic into the environment (ROS1)
mamba install ros-jazzy-desktop

conda deactivate
conda activate mapping_ros2

# mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

Install dependencies:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install cuda -c nvidia/label/cuda-12.4.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd external/Grounded-SAM-2/
pip install grounding_dino/
pip install .
pip install transformers

# install bytetracker
cd ../
pip install byte_track cython_bbox

pip install supervision open3d spacy rerun-sdk
python3 -m spacy download en_core_web_sm

cd ../
pip install . # Install semantic_mapping package
```

# Tests
```bash
# mecanum simulation
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_sim.yaml

# mecanum real
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_real_general.yaml
```