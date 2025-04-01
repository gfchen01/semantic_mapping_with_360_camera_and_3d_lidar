from semantic_mapping.cloud_image_fusion import CloudImageFusion
from semantic_mapping.utils import find_closest_stamp
import pytest
import numpy as np
import cv2
import os
import pickle
import open3d as o3d

from scipy.spatial.transform import Rotation, Slerp

from tqdm import tqdm

if __name__ == "__main__":
    PATH_PREFIX = "/media/all/easystore/dataset/semantic_mapping/cic_mcanum/cic_1217_data/"
    cloud_dir = os.path.join(PATH_PREFIX, 'registered_scan')
    image_dir = os.path.join(PATH_PREFIX, 'image')
    odom_dir = os.path.join(PATH_PREFIX, 'odom')
    lidar_odom_dir = os.path.join(PATH_PREFIX, 'lidar_odom')
    detection_dir = os.path.join(PATH_PREFIX, 'original_annotations')

    cloud_img_fusion = CloudImageFusion(platform='mecanum')

    cloud_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(cloud_dir)])
    image_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(image_dir)])
    odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(odom_dir)])
    lidar_odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(lidar_odom_dir)])

    dummy_mask = np.zeros((1, 640, 1920))
    dummy_labels = ['dummy']
    dummy_confidences = [1.0]

    global_cloud_stack = np.empty((0, 3))

    for i in tqdm(range(len(cloud_stamps[:300]))):
        cloud_file = os.path.join(cloud_dir, f"{cloud_stamps[i]}.bin")

        cloud = np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 3)

        global_cloud_stack = np.vstack((global_cloud_stack, cloud))

    last_odom_stamp = find_closest_stamp(lidar_odom_stamps, cloud_stamps[300])
    last_odom_index = lidar_odom_stamps.index(last_odom_stamp)

    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_cloud_stack)
    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.02)

    global_cloud = np.asarray(global_pcd.points)

    for i in tqdm(range(0, last_odom_index, 10)):
        odom_file = os.path.join(lidar_odom_dir, f"{lidar_odom_stamps[i]}.pkl")

        odom = pickle.load(open(odom_file, 'rb'))

        R_b2w = Rotation.from_quat(odom['orientation']).as_matrix()
        t_b2w = np.array(odom['position'])
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        cloud_body = global_cloud @ R_w2b.T + t_w2b

        _, normalized_depth = cloud_img_fusion.generate_seg_cloud_v2(cloud_body, dummy_mask, dummy_labels, dummy_confidences, R_b2w, t_b2w)

        cv2.imwrite(f"output/depth_{i}.png", normalized_depth)
        # cv2.imshow('normalized_depth', normalized_depth)
        # cv2.waitKey(100)
