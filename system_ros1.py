#!/usr/bin/env python
# coding: utf-8

import json
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import supervision as sv
from supervision.draw.color import ColorPalette

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import box_convert
from bytetrack.byte_tracker import BYTETracker
from types import SimpleNamespace

import cv2

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

def load_models(
    dino_id="IDEA-Research/grounding-dino-base", sam2_id="facebook/sam2-hiera-large"
):
    mask_predictor = SAM2ImagePredictor.from_pretrained(sam2_id, device=device)
    grounding_processor = AutoProcessor.from_pretrained(dino_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(
        device
    )

    return mask_predictor, grounding_processor, grounding_model

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
import open3d as o3d

import yaml
import sys
from pathlib import Path
import time

captioner_not_found = False
try:
    captioner_src_path = Path(__file__).resolve().parents[1] / "ai_module" / "src" / "captioner" / "src"
    sys.path.append(str(captioner_src_path))
    from captioning_backend import Captioner
except ModuleNotFoundError:
    captioner_not_found = True
    print(f"Captioner not found. Fall back to no captioning version.")

from semantic_mapping.mapping_ros1_node import MappingNode

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {args.config} not found.")
        exit(1)

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.float16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    mask_predictor, grounding_processor, grounding_model = load_models()

    # args for BYTETracker
    args = SimpleNamespace(
        **{
            "track_thresh": 0.2, # +0.1 = thresh for adding new tracklet
            "track_buffer": 5, # number of seconds to delete a tracklet
            "match_thresh": 0.85, # 
            "mot20": False,
            "min_box_area": 100,
        }
    )
    tracker = BYTETracker(args)
    
    node = MappingNode(config, mask_predictor, grounding_processor, grounding_model, tracker, device=device)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down ROS node.")
