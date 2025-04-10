import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch
from rclpy.time import Time
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import serialize_message
from bytetrack.byte_tracker import BYTETracker

from .cloud_image_fusion import CloudImageFusion
from .single_object import SingleObject, AdjacencyGraph
from .tools import ros2_bag_utils as ros2_bag_utils
from .utils import generate_colors, extract_meta_class, get_corners_from_box3d_torch, find_nearby_points
from .visualizer import VisualizerRerun

from captioner.captioning_backend import Captioner

# from line_profiler import profile

import copy
# from pytorch3d.ops import box3d_overlap

def serialize_objs_to_bag(writer, obj_mapper, stamp: float, raw_cloud=None, odom=None):
    seconds = int(stamp)
    nanoseconds = int((stamp - seconds) * 1e9)

    marker_array = []
    delete_marker_time = stamp - 1e-4
    delete_marker_seconds = int(delete_marker_time)
    delete_marker_nanoseconds = int((delete_marker_time - delete_marker_seconds) * 1e9)
    clear_marker = Marker()
    clear_marker.header.frame_id = 'map'
    clear_marker.header.stamp = Time(seconds=delete_marker_seconds, nanoseconds=delete_marker_nanoseconds).to_msg()
    clear_marker.action = Marker.DELETEALL

    marker_array.append(clear_marker)

    map_vis_msgs = obj_mapper.to_ros2_msgs(stamp)
    
    for msg in map_vis_msgs:
        if isinstance(msg, PointCloud2):
            writer.write('obj_points', serialize_message(msg), int(stamp * 1e9))
        elif isinstance(msg, Marker):
            marker_array.append(msg)

    if len(marker_array) > 1:
        marker_array_msg = MarkerArray()
        marker_array_msg.markers = marker_array
        writer.write('obj_boxes', serialize_message(marker_array_msg), int(stamp * 1e9))

    if raw_cloud is not None:
        if raw_cloud.shape[0] > 1e5:
            downsampled_cloud = raw_cloud[np.random.choice(raw_cloud.shape[0], int(1e5), replace=False)]
        else:
            downsampled_cloud = raw_cloud

        ros_raw_pcd = ros2_bag_utils.create_point_cloud(downsampled_cloud, seconds, nanoseconds, frame_id='map')
        writer.write('registered_scan', serialize_message(ros_raw_pcd), int(stamp * 1e9))
    
    if odom is not None:
        odom_msg = ros2_bag_utils.create_odom_msg(odom, seconds, nanoseconds)
        tf_transform = ros2_bag_utils.create_tf_msg(odom, seconds, nanoseconds, 'map', 'sensor')

        writer.write('state_estimation', serialize_message(odom_msg), int(stamp * 1e9))
        writer.write('tf', serialize_message(tf_transform), int(stamp * 1e9))

INSTANCE_LEVEL_OBJECTS = [
    # 'chair', 
    # 'table', 
    # 'sofa', 
    # 'garbagebin', 
    # 'cabinet', 
    # 'microwave', 
    # 'door', 
    # 'refrigerator', 
    # 'sign', 
    # 'pottedplant', 
    # 'light',
    # 'vehicle',
    # 'painting',
    # 'box',
]

OMIT_OBJECTS = [
    "window",
    "door",
]

BACKGROUND_OBJECTS = [
]

VERTICAL_OBJECTS = [
    "door", 'painting'
]



class ObjMapper():
    def __init__(
        self,
        tracker: BYTETracker,
        cloud_image_fusion: CloudImageFusion,
        label_template,
        captioner: Captioner = None,
        visualize=False,
        log_info=print
    ):
        self.single_obj_list: list[SingleObject] = []
        self.background_obj_list = []

        self.adjacency_graph = AdjacencyGraph()
        self.cloud_stack = []
        self.stamp_stack = []
        self.valid_cnt = 1

        self.tracker = tracker
        self.cloud_image_fusion = cloud_image_fusion

        self.label_template = label_template
        self.do_visualize = visualize
        if visualize:
            self.rerun_visualizer = VisualizerRerun()
        else:
            self.rerun_visualizer = None

        self.log_info = log_info

        self.captioner = captioner

        # params
        self.voxel_size = 0.05
        self.confidence_thres = 0.30
        self.cloud_to_odom_dist_thres = 6.0
        self.ground_height = -0.5
        self.num_angle_bin = 20
        self.percentile_thresh = 0.8
        self.clear_outliers_cycle = 1

        # # possibly useful params
        # self.odom_move_dist_thres = 0.1

        for label, val in self.label_template.items():
            if val["is_instance"] and label not in INSTANCE_LEVEL_OBJECTS:
                INSTANCE_LEVEL_OBJECTS.append(label)
            self.label_template[label] = val['prompts']
        
        self.log_info(f"Instance level objects: {INSTANCE_LEVEL_OBJECTS}")
        self.log_info(f"label template: {self.label_template}")

    def track_objects(self, det_bboxes, det_labels, det_confidences, detection_odom):
        labels_mask = np.ones_like(det_labels).astype(bool)

        unmatched = False
        det_labels_orig = copy.deepcopy(det_labels)

        det_background_bboxes = []
        det_background_labels = []
        det_background_confidences = []

        for i, label in enumerate(det_labels):
            det_labels[i] = extract_meta_class(label, self.label_template)
            if det_labels[i] == 'None' or det_labels[i] in OMIT_OBJECTS:
                labels_mask[i] = False
            elif det_labels[i] not in INSTANCE_LEVEL_OBJECTS:
                det_background_bboxes.append(det_bboxes[i])
                det_background_labels.append(det_labels[i])
                det_background_confidences.append(det_confidences[i])
                labels_mask[i] = False
                if det_labels[i] not in BACKGROUND_OBJECTS:
                    BACKGROUND_OBJECTS.append(det_labels[i])
            if det_labels[i] == 'None':
                # self.log_info(f'check: {label}')
                unmatched = True
        
        det_labels = det_labels[labels_mask]
        det_bboxes = det_bboxes[labels_mask]
        det_confidences = det_confidences[labels_mask]

        det_tracked = {'bboxes': [], 'confidences': [], 'labels': [], 'ids': []}
        
        R_b2w = Rotation.from_quat(detection_odom['orientation']).as_matrix()
        t_b2w = np.array(detection_odom['position'])
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        if len(det_bboxes) > 0:
            detection_data = np.hstack((det_bboxes, det_confidences.reshape(-1, 1)))
            # reproject objects in the map to assist tracking
            update_tracklet_state = []
            update_tracklet_id = []
            for tracklet in self.tracker.tracked_stracks:
                if tracklet.class_name in INSTANCE_LEVEL_OBJECTS:
                    pred_state = self.predict_new_tracklet_state(tracklet, R_w2b, t_w2b, self.single_obj_list)
                    update_tracklet_state.append(pred_state)
                    update_tracklet_id.append(tracklet.track_id)
            for tracklet in self.tracker.lost_stracks:
                if tracklet.class_name in INSTANCE_LEVEL_OBJECTS:
                    pred_state = self.predict_new_tracklet_state(tracklet, R_w2b, t_w2b, self.single_obj_list)
                    update_tracklet_state.append(pred_state)
                    update_tracklet_id.append(tracklet.track_id)
            self.tracker.compensate_with_3d(update_tracklet_state, update_tracklet_id)
            online_targets = self.tracker.update(detection_data, det_labels)

            bboxes = []
            confidences = []
            labels = []
            obj_ids = []
            for target in online_targets:
                bboxes.append(target.curr_bbox)
                labels.append(target.class_name)
                obj_ids.append(target.track_id)
                confidences.append(target.score)
                if labels[-1] in BACKGROUND_OBJECTS:
                    # obj_ids[-1] = -BACKGROUND_OBJECTS.index(labels[-1]) - 1 # boarder case 0
                    raise ValueError(f"Background object {labels[-1]} should not be tracked")
            
            for bg_bbox, bg_label, bg_confidence in zip(det_background_bboxes, det_background_labels, det_background_confidences):
                bboxes.append(bg_bbox)
                labels.append(bg_label)
                obj_ids.append(-BACKGROUND_OBJECTS.index(bg_label) - 1)
                confidences.append(bg_confidence)
            
            det_tracked['bboxes'] = np.array(bboxes)
            det_tracked['confidences'] = np.array(confidences)
            det_tracked['labels'] = np.array(labels)
            det_tracked['ids'] = np.array(obj_ids)

            # if None in obj_ids:
                # self.log_info(f'check: {obj_ids}; {det_tracked["ids"]}')
        
        return det_tracked, unmatched, det_labels_orig

    # @memory_profiler.profile
    # @profile
    def update_map(self, detections, detection_stamp, detection_odom, cloud, image=None):
        R_b2w = Rotation.from_quat(detection_odom['orientation']).as_matrix()
        t_b2w = np.array(detection_odom['position'])
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        cloud_body = cloud @ R_w2b.T + t_w2b

        confidences = np.array(detections['confidences'])
        confidences_mask = (confidences >= self.confidence_thres)
        confidences = confidences[confidences_mask]

        masks = [mask for mask, confidence in zip(detections['masks'], detections['confidences']) if confidence >= self.confidence_thres]
        labels = [label for label, confidence in zip(detections['labels'], detections['confidences']) if confidence >= self.confidence_thres]
        obj_ids = [obj_id for obj_id, confidence in zip(detections['ids'], detections['confidences']) if confidence >= self.confidence_thres]
        bboxes = [bbox for bbox, confidence in zip(detections['bboxes'], detections['confidences']) if confidence >= self.confidence_thres]

        # masks = np.array(detections['masks'])[confidences_mask]
        # labels = np.array(detections['labels'])[confidences_mask]
        # obj_ids = np.array(detections['ids'])[confidences_mask]
        # bboxes = np.array(detections['bboxes'])[confidences_mask]

        # masks = detections['masks']
        # labels = detections['labels']
        # obj_ids = detections['ids']
        # bboxes = detections['bboxes']

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for i in range(len(masks)):
            masks[i] = cv2.erode(masks[i].astype(np.uint8), kernel, iterations=5).astype(bool)

        # maintain adjacency graph
        if len(obj_ids) == 0:
            return
        elif len(obj_ids) == 1:
            self.adjacency_graph.add_vertex(obj_ids[0])
        else:
            for i in range(len(obj_ids)):
                for j in range(i, len(obj_ids)):
                    self.adjacency_graph.add_edge(obj_ids[i], obj_ids[j])
        
        # # DEBUG: Try using bounding box instead of SAM masks
        # bboxes_mask = np.zeros_like(masks)
        # for i, bbox in enumerate(bboxes):
        #     bbox = bbox.astype(int)
        #     bboxes_mask[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        # obj_clouds_world = generate_seg_cloud(cloud_body, bboxes_mask, labels, confidences, R_b2w, t_b2w, platform='diablo')
        
        obj_clouds_world = self.cloud_image_fusion.generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w)

        for cloud_cnt, cloud in enumerate(obj_clouds_world):
            cloud_to_odom_dist = np.linalg.norm(cloud[:, :3] - t_b2w, axis=1)
            dist_mask = (cloud_to_odom_dist < self.cloud_to_odom_dist_thres)
            # dist_mask = dist_mask & (cloud[:, 2] > self.ground_height)
            cloud = cloud[dist_mask]
            
            if cloud.shape[0] < 5:
                continue
            
            class_id = labels[cloud_cnt]
            obj_id = obj_ids[cloud_cnt]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            if pcd_downsampled.is_empty():
                continue

            # Match object cloud to existing object based on object id
            merged = False
            if obj_id < 0:
                for background_obj in self.background_obj_list:
                    if obj_id in background_obj.obj_id:
                        background_obj.merge(np.array(pcd.points), R_b2w, t_b2w, class_id, detection_stamp)
                        background_obj.inactive_frame = -1
                        merged = True
                        break
            else:
                for single_obj in self.single_obj_list:
                    if obj_id in single_obj.obj_id:
                        single_obj.merge(np.array(pcd.points), R_b2w, t_b2w, class_id, detection_stamp)
                        single_obj.reproject_obs_angle(R_w2b, t_w2b, masks[cloud_cnt], projection_func=self.cloud_image_fusion.scan2pixels)
                        single_obj.inactive_frame = -1
                        merged = True
                        break

            if not merged:
                if obj_id < 0:
                    self.background_obj_list.append(SingleObject(class_id, obj_id, np.array(pcd_downsampled.points), \
                        self.voxel_size, R_b2w, t_b2w, masks[cloud_cnt], detection_stamp, num_angle_bin=self.num_angle_bin))
                else:
                    self.single_obj_list.append(SingleObject(class_id, obj_id, np.array(pcd_downsampled.points), \
                        self.voxel_size, R_b2w, t_b2w, masks[cloud_cnt], detection_stamp, num_angle_bin=self.num_angle_bin))
        
        # ===================== update object crops in captioner =====================

        obj_ids_updated = []
        bboxes_2d = []
        centroids_3d = []
        bboxes_3d = []
        class_names = []

        if self.captioner is not None and image is not None:
            for i, obj_id in enumerate(obj_ids):
                if obj_id < 0:
                    continue

                for single_obj in self.single_obj_list:
                    if obj_id in single_obj.obj_id:
                        cent_3d = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                        bbox_3d = single_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
                        # bbox_3d = single_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)

                        if cent_3d is not None:
                            obj_ids_updated.append(single_obj.obj_id[0])
                            bboxes_2d.append(bboxes[i])
                            centroids_3d.append(cent_3d)
                            class_names.append(single_obj.get_dominant_label())
                            bboxes_3d.append(bbox_3d)
                        break
            
            

            self.captioner.update_object_crops(
                rgb=torch.from_numpy(image).cuda().flip((-1)),
                bboxes_2d=bboxes_2d,
                obj_ids_global=obj_ids_updated,
                centroids_3d=centroids_3d,
                class_names=class_names,
                bboxes_3d=bboxes_3d,
            )
        
        # ===================== associate objects in world =====================

        i = 0
        while i < len(self.single_obj_list):
            single_obj = self.single_obj_list[i]
            if single_obj.obj_id[0] >= 0: # not background object
                single_obj.life += 1

                if single_obj.inactive_frame > 20:
                    # self.single_obj_list.remove(single_obj)
                    # self.log_info(f"Remove {single_obj.class_id}:{single_obj.obj_id}")
                    i += 1
                    continue

                merged_obj = False
                swapped = False
                source_is_vertical = single_obj.get_dominant_label() in VERTICAL_OBJECTS

                if single_obj.life < 1000 and single_obj.life > 5:
                    if single_obj.valid_indices_regularized.shape[0] < 20 and single_obj.inactive_frame > 5: # TODO: voxel count thresh should be related to the object class
                        self.single_obj_list.remove(single_obj)
                        if self.captioner is not None:
                            self.captioner.remove_object(single_obj.obj_id[0])
                        continue
                    else:
                        centroid = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                        if centroid is None:
                            continue
                        else:
                            target_obj = None
                            target_index = -1
                            minimum_dist = 1e6
                            # j = i + 1
                            j = 0
                            while j < len(self.single_obj_list):
                                j += 1
                                same_class_obj = self.single_obj_list[j-1]
                                if same_class_obj.obj_id[0] < 0 or same_class_obj.get_dominant_label() != single_obj.get_dominant_label():
                                    continue
                                if i != (j-1):
                                # if i != (j-1) and not self.adjacency_graph.is_set_adjacent(single_obj.obj_id, same_class_obj.obj_id):
                                    target_is_vertical = same_class_obj.get_dominant_label() in VERTICAL_OBJECTS
                                    if target_is_vertical == source_is_vertical:
                                        target_centroid = same_class_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                                        if target_centroid is None:
                                            continue
                                        else:
                                            dist = np.linalg.norm(target_centroid - centroid)
                                            if dist < minimum_dist:
                                                minimum_dist = dist
                                                target_obj = same_class_obj
                                                target_index = j - 1

                            if target_obj is not None:
                                center_object, extent_object, q_object = single_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                                center_target, extent_target, q_target = target_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                                # avrg of half extent
                                if extent_object is None or extent_target is None:
                                    continue

                                dist_thresh = np.linalg.norm((extent_object/2 + extent_target/2)/2) * 0.5

                                # merge directly if the distance is small
                                if minimum_dist < dist_thresh or minimum_dist < 0.5:
                                    self.log_info(f"Merge {single_obj.class_id}:{single_obj.obj_id} to {target_obj.class_id}:{target_obj.obj_id} with dist thresh {dist_thresh}")
                                    if self.captioner is not None:
                                        centroid_target = target_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                                        bbox_3d_target = target_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
                                        self.captioner.merge_objects(single_obj.obj_id[0], target_obj.obj_id[0], centroid_target, bbox_3d_target)
                                    merged_obj = True
                                
                                # # if not merged, check the IOU of the nearest bounding box
                                # else:
                                #     # bbox3d_object_torch = torch.from_numpy(bbox3d_object).float().unsqueeze(0)
                                #     # bbox3d_target_torch = torch.from_numpy(bbox3d_target).float().unsqueeze(0)

                                #     _, _, angle_object = Rotation.from_quat(q_object).as_euler('xyz')
                                #     _, _, angle_target = Rotation.from_quat(q_target).as_euler('xyz')

                                #     bbox3d_object_corners = get_corners_from_box3d_torch(center_object, extent_object/2, angle_object).unsqueeze(0)
                                #     bbox3d_target_corners = get_corners_from_box3d_torch(center_target, extent_target/2, angle_target).unsqueeze(0)

                                #     bbox3d_object_vol = (extent_object[0] * extent_object[1] * extent_object[2]).item()
                                #     bbox3d_target_vol = (extent_target[0] * extent_target[1] * extent_target[2]).item()
                                #     inter_vol, iou_3d = box3d_overlap(bbox3d_object_corners, bbox3d_target_corners, eps=1e-5)

                                #     inter_vol = inter_vol.item()
                                #     iou_3d = iou_3d.item()

                                #     ratio_object = inter_vol / bbox3d_object_vol
                                #     ratio_target = inter_vol / bbox3d_target_vol

                                #     if iou_3d > 0.3 or (ratio_object > 0.5 and ratio_target > 0.5):
                                #         merged_obj = True
                                    
                                #     elif (ratio_object > 0.5 or ratio_target > 0.5):
                                #         if self.adjacency_graph.is_set_adjacent(single_obj.obj_id, target_obj.obj_id):
                                #             if ratio_object < ratio_target: # object is larger than target
                                #                 single_obj, target_obj = target_obj, single_obj
                                #                 swapped = True
                                            
                                #             obj_voxels = single_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)
                                #             target_voxels = target_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)

                                #             # HYPERPARAM: distance threshold for nearest neighbor
                                #             nearby_indices = find_nearby_points(obj_voxels, target_voxels, 0.05)
                                            
                                #             # single_obj_regularize_mask = single_obj.vote_stat.regualrized_voxel_mask
                                #             # single_obj_valid_mask = single_obj.valid_indices_regularized

                                #             target_obj_regularize_mask = target_obj.vote_stat.regularized_voxel_mask
                                #             target_obj_valid_indices_regularized = target_obj.valid_indices_regularized
                                            
                                #             voxel_exchange_mask_indices = np.where(target_obj_regularize_mask)[0][target_obj_valid_indices_regularized][nearby_indices]
                                #             voxel_exchange_mask = np.zeros_like(target_obj_regularize_mask)
                                #             voxel_exchange_mask[voxel_exchange_mask_indices] = 1
                                #             voxel_exchange_mask = ~voxel_exchange_mask.astype(bool)
                                            
                                #             voxels_exchange, obs_angle_exchange, votes_exchange = target_obj.pop(voxel_exchange_mask)
                                #             single_obj.add(voxels_exchange, obs_angle_exchange, votes_exchange)

                                #             # DEBUG
                                #             self.log_info(f"Voxels exchanged: {voxel_exchange_mask_indices.shape[0]} for {single_obj.get_dominant_label()}:{single_obj.obj_id} and {target_obj.get_dominant_label()}:{target_obj.obj_id}")
                                #             self.log_info(f'Before: {obj_voxels.shape[0]}, {target_voxels.shape[0]}')

                                #             new_obj_voxels = single_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)
                                #             new_target_voxels = target_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)
                                            
                                #             self.log_info(f'After: {new_obj_voxels.shape[0]}, {new_target_voxels.shape[0]}')

                                #             # if new_target_voxels.shape[0] < new_obj_voxels.shape[0]:
                                #             #     # import open3d as o3d
                                #             #     pcd_obj = o3d.geometry.PointCloud()
                                #             #     pcd_obj.points = o3d.utility.Vector3dVector(obj_voxels)
                                #             #     pcd_obj.paint_uniform_color([1, 0, 0])
                                #             #     pcd_target = o3d.geometry.PointCloud()
                                #             #     pcd_target.points = o3d.utility.Vector3dVector(target_voxels)
                                #             #     pcd_target.paint_uniform_color([0, 1, 0])
                                #             #     o3d.visualization.draw_plotly([pcd_obj, pcd_target], window_name="Object and Target")

                                #             #     pcd_obj.points = o3d.utility.Vector3dVector(new_obj_voxels)
                                #             #     pcd_obj.paint_uniform_color([1, 0, 0])
                                #             #     pcd_target.points = o3d.utility.Vector3dVector(new_target_voxels)
                                #             #     pcd_target.paint_uniform_color([0, 1, 0])
                                #             #     o3d.visualization.draw_plotly([pcd_obj, pcd_target], window_name="Object and Target New")
                                            
                                #             if new_obj_voxels.shape[0] / new_target_voxels.shape[0] < 0.2:
                                #                 merged_obj = True
                                #         else:
                                #             merged_obj = True

                                    # # DEBUG: make a test case for separation
                                    # if (ratio_object > 0.8 or ratio_target > 0.8):
                                    #     import os
                                    #     voxel_object = single_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)
                                    #     voxel_target = target_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)
                                    #     dir_name = f'output/separation_test/{detection_stamp}/{single_obj.get_dominant_label()}_{single_obj.obj_id[0]}_{target_obj.get_dominant_label()}_{target_obj.obj_id[0]}'
                                    #     os.makedirs(dir_name, exist_ok=True)
                                    #     np.save(f'{dir_name}/voxel_object.npy', voxel_object)
                                    #     np.save(f'{dir_name}/voxel_target.npy', voxel_target)
                                    #     self.log_info(f"Separation test case saved to {dir_name}")

                                    # self.log_info(f"IOU: {iou_3d}, inter_vol: {inter_vol}, inter/object: {ratio_object}, inter/target: {ratio_target}. {single_obj.get_dominant_label()}:{single_obj.obj_id} and {target_obj.get_dominant_label()}:{target_obj.obj_id}, dist: {minimum_dist}, thresh: {dist_thresh}")

                                if merged_obj:
                                    if target_index < i and not swapped:
                                            single_obj, target_obj = target_obj, single_obj
                                    single_obj.merge_object(target_obj)
                                    single_obj.inactive_frame = -1
                                    self.single_obj_list.remove(target_obj)
                                    del target_obj

                if not merged_obj:
                    single_obj.inactive_frame += 1
                    single_obj.regularize_shape(self.percentile_thresh)
                    i += 1
            else:
                i += 1
        

        self.valid_cnt += 1

    def predict_new_tracklet_state(self, tracklet, R_w2b, t_w2b, single_obj_list):
        def tracklet_state_to_xyxy(state):
            x, y, a, h = state[:4]
            w = h * a 
            x1 = x - w/2
            x2 = x + w/2
            y1 = y - h/2
            y2 = y + h/2
            return np.array([x1, y1, x2, y2])

        def xyxy_to_tracklet_state(xyxy):
            x1, y1, x2, y2 = xyxy
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            a = w / h
            return np.array([x, y, a, h])

        tracklet_id = tracklet.track_id
        tracklet_state = tracklet.mean # [x, y, a, h, vx, vy, va, vh]
        curr_p = xyxy_to_tracklet_state(tracklet.curr_bbox)
        tracklet_state[:4] = curr_p

        # DEBUG: percentile_thresh should be a parameter
        percentile_thresh = 0.8

        pred_state = tracklet_state.copy()
        for single_obj in single_obj_list:
            if tracklet_id in single_obj.obj_id:
                obj_center = single_obj.infer_centroid(diversity_percentile=percentile_thresh, regularized=True) # this might happen before any regularization
                if obj_center is not None:
                    obj_center_body = obj_center @ R_w2b.T + t_w2b
                    obj_center_body = obj_center_body.reshape(1, 3)
                    obj_center_pixel_idx = self.cloud_image_fusion.scan2pixels(obj_center_body)
                    # pred_state[:2] = obj_center_pixel_idx[0, :2]
                    pred_state[0] = obj_center_pixel_idx[0, 0]

                break
        
        return pred_state

    def open3d_vis(self, odom=None):
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis_list = [coord]
        tree_cnt = 0
        colors_list = generate_colors(len(self.single_obj_list))
        for single_obj in self.single_obj_list:
            obj_points = single_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, require_certified=True)                 
            if len(obj_points) == 0:
                continue
            
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

        self.log_info(f'Number of objects: {len(self.single_obj_list)}')
        self.log_info(f'Number of valid objects: {tree_cnt}')

        if odom is not None:
            R_b2w = Rotation.from_quat(odom['orientation']).as_matrix()
            t_b2w = np.array(odom['position'])
            odom_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=t_b2w)
            odom_coord.rotate(R_b2w, center=t_b2w)
            vis_list.append(odom_coord)
        
        o3d.visualization.draw_geometries(vis_list, window_name="mecanum_ros2")

    def serialize_map_to_dict(self, stamp):
        objects_dict = {}
        for single_obj in self.single_obj_list:
            obj_label = single_obj.get_dominant_label()
            obj_id = [int(x) for x in single_obj.obj_id]
            obj_center = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
            obj_bbox3d = single_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
            
            if obj_center is not None:
                obj_center = obj_center.tolist() if isinstance(obj_center, np.ndarray) else obj_center
                box_center = obj_bbox3d[0].tolist() if isinstance(obj_bbox3d[0], np.ndarray) else obj_bbox3d[0]
                box_extent = obj_bbox3d[1].tolist() if isinstance(obj_bbox3d[1], np.ndarray) else obj_bbox3d[1]
                box_rotation = obj_bbox3d[2].tolist() if isinstance(obj_bbox3d[2], np.ndarray) else obj_bbox3d[2]
                objects_dict[obj_id[0]] = {
                    'label': obj_label,
                    'id': obj_id,
                    'center': obj_center,
                    'bbox3d': {'center': box_center, 'extent': box_extent, 'rotation': box_rotation}
                }
        return objects_dict

    def to_ros2_msgs(self, stamp):
        tree_cnt = 0
        colors_to_choose = generate_colors(len(self.single_obj_list), is_int=False)
        seconds = int(stamp)
        nanoseconds = int((stamp - seconds) * 1e9)

        points_list = []
        colors_list = []
        bbox_msg_list = []
        text_msg_list = []
        for single_obj in self.single_obj_list:
            obj_points = single_obj.retrieve_valid_voxels(
                diversity_percentile=self.percentile_thresh,
                regularized=True
            )
            if len(obj_points) == 0:
                continue

            point = obj_points
            color = np.array([colors_to_choose[tree_cnt]] * obj_points.shape[0])
            points_list.append(point)
            colors_list.append(color)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            pcd.colors = o3d.utility.Vector3dVector(color)

            aabb = pcd.get_axis_aligned_bounding_box()


            obj_marker = ros2_bag_utils.create_wireframe_marker(
                center=aabb.get_center(),
                extent=aabb.get_extent(),
                yaw=0.0,
                ns=f'{single_obj.class_id}',
                box_id=f'{single_obj.obj_id[0]}',
                color=colors_to_choose[tree_cnt],
                seconds=seconds,
                nanoseconds=nanoseconds,
                frame_id='map'
            )


            text_msg = ros2_bag_utils.create_text_marker(
                center=aabb.get_center(),
                marker_id=single_obj.obj_id[0],
                text=single_obj.get_dominant_label(),
                color=colors_to_choose[tree_cnt],
                text_height=0.2,
                seconds=seconds,
                nanoseconds=nanoseconds,
                frame_id='map'
            )

            bbox_msg_list.append(obj_marker)
            text_msg_list.append(text_msg)

            tree_cnt += 1

        if len(points_list) != 0:
            points = np.concatenate(points_list, axis=0)
            colors = np.concatenate(colors_list, axis=0)
            ros_pcd = ros2_bag_utils.create_colored_point_cloud(
                points=points,
                colors=colors,
                seconds=seconds,
                nanoseconds=nanoseconds,
                frame_id='map'
            )
        else:
            ros_pcd = None

        return bbox_msg_list, text_msg_list, ros_pcd

    def rerun_vis(self, odom, regularized=True, show_bbox=False, debug=False, enforce=False):
        if self.do_visualize:
            if debug:
                self.rerun_visualizer.visualize_debug(self.single_obj_list, odom)
            else:
                self.rerun_visualizer.visualize(self.single_obj_list, odom, regularized=regularized, show_bbox=show_bbox)
        else:
            if enforce:
                self.rerun_visualizer = VisualizerRerun() if self.rerun_visualizer is None else self.rerun_visualizer
                self.rerun_visualizer.visualize(self.single_obj_list, odom, regularized=regularized, show_bbox=show_bbox)
            else:
                self.log_info("Visualizer is not enabled!!!")
    
    def print_obj_info(self):
        self.log_info('==== All Objects Info ====')
        for single_obj in self.single_obj_list:
            obj_str = single_obj.get_info_str()
            self.log_info(obj_str)



