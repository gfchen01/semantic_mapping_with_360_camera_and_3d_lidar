import numpy as np
import open3d as o3d
import os
import cv2
import supervision as sv
from supervision.draw.color import ColorPalette
import time
from semantic_mapping import scannet_utils
from functools import partial
from tqdm import tqdm

from semantic_mapping.cloud_image_fusion import CloudImageFusion

FRAME_RATE = 10

if __name__=='__main__':
    # ================== #
    # Load data
    # ================== #

    PATH_PREFIX = '/media/all/easystore/dataset/semantic_mapping/scannetv2/scene0000_00_data/'
    depth_dir = PATH_PREFIX + 'depth'
    odom_dir = PATH_PREFIX + 'pose'
    image_dir = PATH_PREFIX + 'color'
    
    cnt = 0
    valid_cnt = 1
    cloud_stack = []
    adjacency_by_label = {}

    box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
    image_length = len(os.listdir(image_dir))
    rgb_intrinsics = scannet_utils.read_intrinsics(PATH_PREFIX + 'intrinsic/intrinsic_color.txt')
    depth_intrinsics = scannet_utils.read_intrinsics(PATH_PREFIX + 'intrinsic/intrinsic_depth.txt')

    rgb_shape = (968, 1296)
    depth_shape = (480, 640)

    scannet_projector = partial(scannet_utils.project_pointcloud_to_image, camera_intrinsics=rgb_intrinsics, image_shape=rgb_shape)
    # ================== #
    # setup camera model
    # ================== #

    cloud_image_fusion = CloudImageFusion(platform='scannet')

    outdir = 'output/projection/scannet'
    os.makedirs(outdir, exist_ok=True)

    start_time = time.time()
    for cnt in tqdm(range(image_length), desc='Processing frames'):
        depth_file = os.path.join(depth_dir, f"{cnt}.png")
        odom_file = os.path.join(odom_dir, f"{cnt}.txt")
        image_file = os.path.join(image_dir, f"{cnt}.jpg")
        
        stamp = start_time + cnt / FRAME_RATE
        # process odom
        SE3 = scannet_utils.read_pose(odom_file)
        t_b2w = SE3[:3, 3]
        R_b2w = SE3[:3, :3]
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        
        # process mask
        cv_image = cv2.imread(image_file)

        # process depth
        depth_image = scannet_utils.read_depth(depth_file)
        cloud_world = scannet_utils.depth_to_pointcloud(depth_image, depth_intrinsics, SE3)
        cloud_body = scannet_utils.depth_to_pointcloud(depth_image, depth_intrinsics, np.eye(4))

        dummy_masks = np.ones([1, cv_image.shape[0], cv_image.shape[1]])
        dummy_label = ['dummy']
        dummy_confidence = np.array([1.0])

        obj_clouds_world = cloud_image_fusion.generate_seg_cloud(cloud_body, dummy_masks, dummy_label, dummy_confidence, R_b2w, t_b2w, image_src=cv_image)

        cv2.imwrite(os.path.join(outdir, f"{cnt}.png"), cv_image)
