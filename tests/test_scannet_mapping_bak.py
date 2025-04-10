import cv2
import numpy as np
import open3d as o3d
import os

from docutils.nodes import target
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import pickle
import pandas as pd
import supervision as sv
from supervision.draw.color import ColorPalette
from matplotlib import pyplot as plt
from single_object import SingleObject, AdjacencyGraph
import time
from semantic_mapping import scannet_utils

from functools import partial
from tqdm import tqdm

from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

FRAME_RATE = 10

label_list = []

single_obj_list = []

# params
voxel_size = 0.01
voting_thres = 3
merge_thres = 0.1
diversity_thres = 0
diversity_diff = 3
confidence_thres = 0.50

odom_move_dist_thres = 0.1
cloud_to_odom_dist_thres = 7.0
ground_z_thres = -0.4

num_angle_bin = 20
percentile_thresh = 0.4

clear_outliers_cycle = 5

label_filter = []

if __name__=='__main__':
    PATH_PREFIX = '/media/luke/easystore/dataset/semantic_mapping/scannetv2/scene0000_00_data/'

    depth_dir = os.path.join(PATH_PREFIX, 'depth')
    odom_dir = os.path.join(PATH_PREFIX, 'pose')
    mask_dir = os.path.join(PATH_PREFIX, 'mask_tracked')
    image_dir = os.path.join(PATH_PREFIX, 'color')
    
    cnt = 0
    valid_cnt = 1
    
    cloud_stack = []

    annotated_length = len(os.listdir(mask_dir))

    rgb_intrinsics = scannet_utils.read_intrinsics(PATH_PREFIX + 'intrinsic/intrinsic_color.txt')
    depth_intrinsics = scannet_utils.read_intrinsics(PATH_PREFIX + 'intrinsic/intrinsic_depth.txt')

    rgb_shape = (968, 1296)
    depth_shape = (480, 640)

    scannet_projector = partial(scannet_utils.project_pointcloud_to_image, camera_intrinsics=rgb_intrinsics, image_shape=rgb_shape)
    
    start_time = time.time()
    pool_exec = ProcessPoolExecutor(max_workers=8)

    for cnt in tqdm(range(annotated_length), desc='Processing frames'):
        depth_file = os.path.join(depth_dir, f"{cnt}.png")
        odom_file = os.path.join(odom_dir, f"{cnt}.txt")
        mask_file = os.path.join(mask_dir, f"{cnt}.npz") # mask file name is based on image stamp
        
        stamp = start_time + cnt / FRAME_RATE
        
        if not os.path.exists(mask_file):
            print(f"Missing files: {mask_file}")
            exit()
        
        # process odom
        SE3 = scannet_utils.read_pose(odom_file)
                
        t_b2w = SE3[:3, 3]
        R_b2w = SE3[:3, :3]
        
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        
        # process mask
        seg_result = np.load(mask_file)
        confidences = seg_result['confidences']
        confidences_mask = (confidences >= confidence_thres)
        confidences = confidences[confidences_mask]
        masks = seg_result['masks'][confidences_mask]
        labels = seg_result['labels'][confidences_mask]
        obj_ids = seg_result['ids'][confidences_mask]
        bboxes = seg_result['bboxes'][confidences_mask]

        for l in labels:
            if l not in adjacency_by_label:
                adjacency_by_label[l] = AdjacencyGraph()
                print(f"Found new label {l}")

        # process depth
        depth_image = scannet_utils.read_depth(depth_file)
        cloud_world = scannet_utils.depth_to_pointcloud(depth_image, depth_intrinsics, SE3)
        cloud_body = scannet_utils.depth_to_pointcloud(depth_image, depth_intrinsics, np.eye(4))

        # maintain adjacency graph
        if len(obj_ids) == 0:
            continue
        elif len(obj_ids) == 1:
            adjacency_by_label[labels[0]].add_vertex(obj_ids[0])
            # id_adjacency.add_vertex(obj_ids[0])
        else:
            for i in range(len(obj_ids)):
                for j in range(i, len(obj_ids)):
                    if labels[i] == labels[j]:
                        adjacency_by_label[labels[i]].add_edge(obj_ids[i], obj_ids[j])
                    # id_adjacency.add_edge(obj_ids[i], obj_ids[j])

        # obj_clouds_world = generate_seg_cloud(cloud_body, masks, labels, confidences, SE3[:3, :3], SE3[:3, 3], platform='scannet')

        obj_clouds_world = scannet_utils.masked_depth_to_pointcloud(masks, rgb_intrinsics, labels, depth_image, depth_intrinsics, SE3)

        # image_src = cv2.imread(os.path.join(image_dir, f"{cnt}.jpg"))
        # image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        # obj_clouds_world = scannet_utils.masked_depth_to_pointcloud(masks, rgb_intrinsics, labels, confidences, depth_image, depth_intrinsics, SE3, image_src=image_src)
        # image_src = cv2.cvtColor(image_src, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'reproject_debug_vis/{cnt}.png', image_src)

        for cloud_cnt, cloud in enumerate(obj_clouds_world):
            # # DEBUG
            # if labels[cloud_cnt] == "chair":
            #     chair_pcd = o3d.geometry.PointCloud()
            #     chair_pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
            #     chair_pcd.voxel_down_sample(voxel_size=voxel_size)
            #     chair_stack = np.concatenate((chair_stack, np.asarray(chair_pcd.points)), axis=0)

            cloud_to_odom_dist = np.linalg.norm(cloud[:, :3] - t_w2b, axis=1)
            dist_mask = (cloud_to_odom_dist < cloud_to_odom_dist_thres)
            dist_mask = dist_mask & (cloud[:, 2] > ground_z_thres)
            cloud = cloud[dist_mask]
            
            if cloud.shape[0] < 100:
                continue
            
            class_id = labels[cloud_cnt]
            obj_id = obj_ids[cloud_cnt]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

            # print(f'before: {cloud.shape[0]}, after: {np.asarray(pcd_downsampled.points).shape[0]}')
            # o3d.visualization.draw_geometries([pcd_downsampled, coord])
            
            # Match object cloud to existing object based on object id
            merged = False
            for single_obj in single_obj_list:
                if obj_id in single_obj.obj_id and single_obj.class_id == class_id:
                    single_obj.merge(np.array(pcd_downsampled.points), R_b2w, t_b2w, stamp)
                    single_obj.reproject_obs_angle(R_w2b, t_w2b, bboxes[cloud_cnt], scannet_projector)
                    merged = True
                    break

            if not merged:
                single_obj_list.append(SingleObject(class_id, obj_id, np.array(pcd_downsampled.points), voxel_size, \
                    R_b2w, t_b2w, masks[cloud_cnt], stamp, valid_vote_thres=voting_thres, num_angle_bin=num_angle_bin))
        
        # filter and merge objects in list each cycle
        for single_obj in single_obj_list:
            single_obj.life += 1
            merged_obj = False
            
            if single_obj.life < 100 and single_obj.vote_stat.voxels.shape[0] > 100:
                valid_voxels, votes = single_obj.retrieve_valid_voxels(diversity_percentile=percentile_thresh, require_certified=False)
                centroid = (valid_voxels * votes[:, None]).sum(axis=0) / votes.sum()
                # centroid = valid_voxels.sum(axis=0)[:2] / valid_voxels.shape[0]

                for same_class_obj in single_obj_list:
                    if len(valid_voxels) == 0:
                            continue
                    if same_class_obj.class_id == single_obj.class_id and same_class_obj != single_obj:
                        if not adjacency_by_label[same_class_obj.class_id].is_set_adjacent(single_obj.obj_id, same_class_obj.obj_id):
                        # if not id_adjacency.is_set_adjacent(single_obj.obj_id, same_class_obj.obj_id):
                            target_valid_voxels, votes = same_class_obj.retrieve_valid_voxels(diversity_percentile=percentile_thresh, require_certified=False)
                            if target_valid_voxels.shape[0] == 0:
                                continue
                            target_centroid = (target_valid_voxels * votes[:, None]).sum(axis=0) / votes.sum()
                            # target_centroid = target_valid_voxels.sum(axis=0)[:2] / target_valid_voxels.shape[0]

                            if np.linalg.norm(target_centroid - centroid) < 0.5:
                                print(f"Merge {single_obj.class_id}:{single_obj.obj_id} to {same_class_obj.class_id}:{same_class_obj.obj_id}, centroid1: {centroid}, centroid2: {target_centroid}")
                                same_class_obj.merge_object(single_obj)
                                single_obj_list.remove(single_obj)
                                merged_obj = True
                                break
                
            if not merged_obj and single_obj.life % clear_outliers_cycle == 0:
                # single_obj.clear_outliers()
                # if single_obj.life < 200 :
                #     single_obj.clustering_filter()
                pass
        
        if DEBUG:
            print(f"Frame {stamp} =====================")

            # class_cnt = {}
            # diversities = {}
            # votes = {}
            # for single_obj in single_obj_list:
            #     if single_obj.retrieve_valid_voxels(diversity_percentile=percentile_thresh, require_certified=False).shape[0] < 2:
            #         continue
            #     view_point_diversity = np.sum(single_obj.vote_stat.observation_angles, axis=1)
            #     obj_label = single_obj.class_id
            #     if obj_label not in class_cnt:
            #         class_cnt[obj_label] = []
            #     class_cnt[obj_label].append(single_obj.obj_id)
            #     diversities[f'{obj_label}_{len(class_cnt[obj_label])}'] =\
            #         view_point_diversity

            #     votes[f'{obj_label}_{len(class_cnt[obj_label])}'] = single_obj.vote_stat.vote
            # print(class_cnt)
            
            # image_file = os.path.join(image_dir, f"{stamp}.png")
            # image = cv2.imread(image_file)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # masked_image = image.copy()
            
            # detections = sv.Detections(xyxy=seg_result['bboxes'], mask=seg_result['masks'], class_id=seg_result['labels'])
            # mask_annotator = sv.MaskAnnotator()
            # masked_image = mask_annotator.annotate(masked_image, detections=detections)
            
            # plt.imshow(masked_image)
            # plt.axis("off")
            # plt.show()
            
            if valid_cnt % 100 == 0:

                # if len(diversities) > 0:
                #     fig, axes = plt.subplots(len(diversities), 1, figsize=(15, 5), sharey=True)  # 1 row, 3 columns, shared y-axis
                    
                #     plot_cnt = 0
                #     bins = np.arange(1, num_angle_bin)
                #     for k, v in diversities.items():
                #         axes[plot_cnt].hist(v, bins=bins)
                #         axes[plot_cnt].set_title(k)
                #         plot_cnt += 1
                    
                #     # plt.tight_layout()
                #     plt.show()

                # if len(votes) > 0:
                #     fig, axes = plt.subplots(len(votes), 1, figsize=(15, 5), sharey=True)  # 1 row, 3 columns, shared y-axis
                    
                #     plot_cnt = 0
                #     for k, v in votes.items():
                #         axes[plot_cnt].hist(v, bins=np.arange(1, v.max()))
                #         axes[plot_cnt].set_title(k)
                #         plot_cnt += 1
                    
                #     # plt.tight_layout()
                #     plt.show()

                vis_list = [coord]
                tree_cnt = 0
                colors_list = generate_colors(len(single_obj_list))
                for single_obj in single_obj_list:
                    if len(label_filter) > 0 and single_obj.class_id not in label_filter:
                        continue
                    
                    print(f"Object {single_obj.class_id}:{single_obj.obj_id} has {single_obj.vote_stat.voxels.shape[0]} voxels")
                    obj_points, votes = single_obj.retrieve_valid_voxels(diversity_percentile=percentile_thresh, require_certified=False)                 
                    if len(obj_points) == 0:
                        continue
                    
                    obj_centroid = (obj_points * votes[:, None]).sum(axis=0) / votes.sum()

                    # Draw the object center
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    sphere.translate(obj_centroid)  # Move the sphere to the desired location
                    sphere.paint_uniform_color([0, 1, 0])  # Set the color to red (RGB)
                    vis_list.append(sphere)

                    sphere_non_weighted = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    sphere_non_weighted.translate(obj_points.mean(axis=0))  # Move the sphere to the desired location
                    sphere_non_weighted.paint_uniform_color([0, 0, 1])  # Set the color to red (RGB)
                    vis_list.append(sphere_non_weighted)

                    # Draw the object points
                    point = obj_points
                    color = np.array([colors_list[tree_cnt]] * obj_points.shape[0])

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point)
                    pcd.colors = o3d.utility.Vector3dVector(color)
                    vis_list.append(pcd)

                    aabb = pcd.get_axis_aligned_bounding_box()
                    aabb.color = colors_list[tree_cnt]
                    vis_list.append(aabb)
                    
                    tree_cnt += 1
                    
                odom_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=t_b2w)
                odom_coord.rotate(R_b2w, center=t_b2w)
                vis_list.append(odom_coord)
                
                o3d.visualization.draw_geometries(vis_list)

                # chair_pcd = o3d.geometry.PointCloud()
                # chair_pcd.points = o3d.utility.Vector3dVector(chair_stack)
                # o3d.visualization.draw_geometries([chair_pcd, coord, odom_coord])

        last_odom_t = t_b2w
        valid_cnt += 1

    # clustering_labels = np.array(pcd_downsampled.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))
    # max_label = clustering_labels.max()

    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(clustering_labels / (max_label if max_label > 0 else 1))
    # colors[clustering_labels < 0] = 0
    # pcd_downsampled.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd_downsampled],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    