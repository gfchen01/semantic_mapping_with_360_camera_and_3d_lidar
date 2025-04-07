## Introduction

This is a 3D open-vocabulary instance-level semantic mapping module. The system takes stamped panoramic images, 3D LiDAR scans, and odometry as inputs. The outputs are 3D instance-level object centroids and 3D bounding box estimates. The system uses [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for open-vocabulary object detection and segmentation on images, manages the 3D object clustering, tracking, and filtering to provide a 3D object-level map, aiming to provide a real-time and easily adaptable spatial representation for language navigation and other upstream tasks.

<p float="center">
  <img src="images/mecanum_wheel.jpg" height="300" />
  <img src="images/wheelchair.jpg" height="300" />
</p>

Left: Livox-Mid360 + Ricoh Theta Z1 Camera; Right: Velodyne VLP-16 + Ricoh Theta Z1 Camera.

## Repository Setup

This repository contains the code for semantic mapping with 360 (panoramic) camera and 3D LiDAR. The system has been tested with the following robot/sensor setups: [wheelchair_platform](https://github.com/jizhang-cmu/cmu_vla_challenge_unity) and
[mecanum_wheel_platform](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform).

For the ros1 branch, we will use the wheelchair platform as an example in the instructions, and for the ros2 branch, we will use the mecanum wheel platform as an example. This branch is for ROS2 and should be set up on **Ubuntu 22.04** or **Ubuntu 24.04**.

### Requirements

1. GPU with >16GB memory
2. Miniforge install is required for setting up ROS. [Follow the install instructions here](https://github.com/conda-forge/miniforge).

### Setup the environment

First, clone the repo and create the environment:

``` bash
git clone https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar.git
git checkout ros2

conda create --name mapping_ros2 python=3.11
conda activate mapping_ros2
```

To make ROS accessible in the environment, install ROS/ROS2 using robostack. 

If you are using Ubuntu 22.04, install ros-humble with mamba:

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-humble
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

# Install ros-humble into the environment (ROS2)
mamba install ros-humble-desktop

conda deactivate
conda activate mapping_ros2
```

If you are using Ubuntu 24.04, install ros-jazzy with mamba:

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-jazzy
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

# Install ros-jazzy into the environment (ROS2)
mamba install ros-jazzy-desktop

conda deactivate
conda activate mapping_ros2
```

Install Grounded-SAM-2:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install cuda -c nvidia/label/cuda-12.4.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd external/
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install Grounded-SAM-2/grounding_dino/
pip install Grounded-SAM-2
pip install transformers
```

Install the customized byte_track and cython_bbox in `external` folder:
```bash
# install customized external libraries
pip install byte_track cython_bbox
```


Install other dependencies
```bash
pip install supervision open3d spacy rerun-sdk
python3 -m spacy download en_core_web_sm
```

Finally, install the semantic mapping package:
```bash
cd ../
pip install . # Install semantic_mapping package
```

## Run
Our system subscribes to the following topics:

| Topic Name        | Message Type            | Note                                      |
| ----------------- | ----------------------- | ----------------------------------------- |
| /registered_scan  | sensor_msgs/PointCloud2 | Registered point cloud in the world frame |
| /camera/image     | sensor_msgs/Image       | 360 Images                                |
| /state_estimation | nav_msgs/Odometry       | Odometry of the LiDAR frame               |

If you need to change the topic names you subscribe to, please check [the topic subscriptions here](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar/blob/ros2/semantic_mapping/mapping_ros2_node.py#L129-L155).

**Before running**, [check the configs](./config/). Here is an example config:

```yaml
platform: mecanum # name of the platform. Implies extrinsics.

use_lidar_odom: true # If set true, the LiDAR odometry named /aft_mapped_to_init_incremental should be provided, which is a more accurate low-frequency odometry from the LiDAR SLAM

detection_linear_state_time_bias: -0.05 # Compensation on the camera odometry estimate of linear states

detection_angular_state_time_bias: 0.0 # Compensation on the camera odometry estimate of angular states

image_processing_interval: 0.5 # Period in which the semantic inference and mapping are performed

visualize: true # If set true, a Rerun window spawns for visualization

vis_interval: 0.5 # Period in which map is visualized

annotate_image: false # if set true, 2D images with annotated semantic masks and LiDAR projections will be saved

# prompts 
prompts:
    chair:
    prompts: # prompts that actually goes to DINO
      - chair
    is_instance: true # Whether the object should be treated as instance-level
  sofa:
    prompts:
      - sofa
    is_instance: true
  table:
    prompts:
      - table
    is_instance: true
  wall:
    prompts:
      - wall
      - white wall
      - blue wall
      - black wall
    is_instance: false
```

You can either run in simulation or real-world system (rosbag).

### Run with simulation
 
 First, set up the simulated base autonomy system following the instructions here: [mecanum_wheel_platform](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform.git).
 
 Then start the base autonomy system with `system_simulation.sh` or `system_simulation_with_route_planner.sh`, you should see an Rviz and the unity simulation like this:
 
 <p float="center">
 <img src="./images/mecanum_sim.png">
 </p>
 
 Then start the semantic mapping module:
 ```bash
 python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_sim.yaml
 ```
 When the ros node runs stably, you can start operating the simulation. You should see a rerun visualization:
 
 ![](./images/ros2_sim_vis.png)

### Run in real-world

An example bag recorded with [mecanum_wheel_platform](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform.git) on real-world system [can be downloaded here](https://drive.google.com/file/d/1upuVfn2NmJdryzzpOvqMzZbjqAoekk00/view?usp=sharing). After downloading the bag, start the ros node with:

```bash
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_real.yaml
```

Then play the rosbag with `rosbag play sqh_2.bag`. You should be able to see the visualized map in Rerun window.


<figure style="text-align: center;">
  <img src="./images/mecanum_real_3d_vis.png" width="90%"/>
  <figcaption style="text-align: center;">Rerun visualization of Mapping Result in 3D</figcaption>
</figure>


<figure style="text-align: center;">
  <img src="./images/mecanum_real_2d_vis.png" width="90%"/>
  <figcaption style="text-align: center;">One frame of aligned Depth+Image+Semantics Input (set `annotate_image: true` in config). The LiDAR scan points are projected to the image frame, with darker color representing farther points. Detection and segmentation results are drawn in masks and bounding boxes, with colors representing different instances.
</figcaption>
</figure>


## Credits

Contributors: [Guofei Chen](https://gfchen01.github.io), [Changwei Yao](https://github.com/chadwick-yao), [Wenshan Wang](http://www.wangwenshan.com/), [Ji Zhang](https://frc.ri.cmu.edu/~zhangji/).

[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [byte_track](https://github.com/ifzhang/ByteTrack), and [cython_bbox](https://github.com/samson-wang/cython_bbox) are from open-source releases.

