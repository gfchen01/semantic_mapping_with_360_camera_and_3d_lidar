git submodule init
git submodule update

conda create --name dino_ros1 python=3.11

conda activate dino_ros1

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# if you want to use Jazzy, also add its own channel:
conda config --env --add channels robostack-jazzy

# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

# Install ros-noetic into the environment (ROS1)
mamba install ros-noetic-desktop

conda deactivate
conda activate dino_ros1

mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

<!-- git clone https://github.com/chadwick-yao/Grounded-SAM-2.git -->

<!-- pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 -->
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install cuda -c nvidia/label/cuda-12.4.0
export CUDA_HOME=$(path_to_your_conda_environment)

pip install grounding_dino

pip install .

pip install git+https://github.com/huggingface/transformers

# install bytetracker
cd ..
pip install -e byte_track

<!-- git clone git@github.com:valentin-fngr/cython_bbox.git
cd cython_bbox -->
## before installing, change the source code first
# in cython_bbox repo, in file src/cython_bbox.pyx change lines 12 and 13 to be:
# DTYPE = np.float32
# ctypedef np.float32_t DTYPE_t
pip install -e cython_bbox

pip install supervision open3d spacy rerun-sdk
python3 -m spacy download en_core_web_sm


# Tests
```bash

# mecanum simulation
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_sim.yaml

```