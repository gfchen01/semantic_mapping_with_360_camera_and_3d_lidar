import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from types import SimpleNamespace
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
import supervision as sv
from supervision.draw.color import ColorPalette

from bytetrack.byte_tracker import BYTETracker
from semantic_mapping.utils import find_closest_stamp, find_neighbouring_stamps
from semantic_mapping.semantic_map import ObjMapper
from semantic_mapping.cloud_image_fusion import CloudImageFusion

import memory_profiler

# @memory_profiler.profile()
def main():
    # import argparse
    # args_parser = argparse.ArgumentParser()
    # args_parser.add_argument('--data_dir', type=str, required=True)
    # args_parser.add_argument('--visualize', action='store_true')
    # args_parser.add_argument('--annotate', action='store_true')
    # args = args_parser.parse_args()

    # PATH_PREFIX = args.data_dir
    # VISUALIZE = args.visualize
    # ANNOTATE = args.annotate

    PLATFORM = "mecanum"
    PATH_PREFIX = "/media/all/easystore/dataset/ros2/extracted"
    VISUALIZE = True
    ANNOTATE = False
    TO_JSON = True

    POST_FIX = ""
    cloud_dir = os.path.join(PATH_PREFIX, f'registered_scan{POST_FIX}')
    image_dir = os.path.join(PATH_PREFIX, 'image')
    odom_dir = os.path.join(PATH_PREFIX, f'odom{POST_FIX}')
    lidar_odom_dir = os.path.join(PATH_PREFIX, f'lidar_odom{POST_FIX}')
    detection_dir = os.path.join(PATH_PREFIX, 'original_annotations')

    if ANNOTATE:
        box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
        label_annotator = sv.LabelAnnotator(
            color=ColorPalette.DEFAULT,
            text_padding=4,
            text_scale=0.3,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.INDEX,
            smart_position=True,
        )
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
        ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'annotated_3d_in_loop')
        if os.path.exists(ANNOTATE_OUT_DIR):
            os.system(f"rm -r {ANNOTATE_OUT_DIR}")
        os.makedirs(ANNOTATE_OUT_DIR, exist_ok=True)

        VERBOSE_ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'verbose_3d_in_loop')
        if os.path.exists(VERBOSE_ANNOTATE_OUT_DIR):
            os.system(f"rm -r {VERBOSE_ANNOTATE_OUT_DIR}")
        os.makedirs(VERBOSE_ANNOTATE_OUT_DIR, exist_ok=True)


    cloud_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(cloud_dir)])
    image_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(image_dir)])
    odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(odom_dir)])
    lidar_odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(lidar_odom_dir)])
    
    cnt = 0
    image_linear_time_bias = -0.1
    image_angular_time_bias = 0.0
    cloud_stack = []
    stamp_stack = []

    import yaml
    with open(os.path.join(PATH_PREFIX, 'prompts.yaml'), "r") as file:
        label_template = yaml.safe_load(file)

    args = SimpleNamespace(
        **{
            "track_thresh": 0.2, # +0.1 = thresh for adding new tracklet
            "track_buffer": 5, # number of frames to delete a tracklet
            "match_thresh": 0.85, # 
            "mot20": False,
            "min_box_area": 100,
        }
    )
    tracker = BYTETracker(args)

    cloud_image_fusion = CloudImageFusion(platform=PLATFORM)

    mapper = ObjMapper(tracker=tracker, cloud_image_fusion=cloud_image_fusion, label_template=label_template, visualize=VISUALIZE)

    global_cloud = np.empty((0, 3))

    last_odom = None

    image_stamps = image_stamps[0:-1:4]

    for cnt, image_stamp in enumerate(tqdm(image_stamps)):
        image_linear_stamp = image_stamp + int(image_linear_time_bias * 1e9)
        image_angular_stamp = image_stamp + int(image_angular_time_bias * 1e9)

        lidar_stamp = find_closest_stamp(cloud_stamps, image_stamp)
        odom_stamp = find_closest_stamp(odom_stamps, image_angular_stamp)
        left_lidar_odom_stamp, right_lidar_odom_stamp = find_neighbouring_stamps(lidar_odom_stamps, image_linear_stamp)

        if image_stamp - left_lidar_odom_stamp > 1e9 or right_lidar_odom_stamp - image_stamp > 1e9:
            continue

        image_file = os.path.join(image_dir, f"{image_stamp}.png")
        cloud_file = os.path.join(cloud_dir, f"{lidar_stamp}.bin")
        left_lidar_odom_file = os.path.join(lidar_odom_dir, f"{left_lidar_odom_stamp}.pkl")
        right_lidar_odom_file = os.path.join(lidar_odom_dir, f"{right_lidar_odom_stamp}.pkl")
        odom_file = os.path.join(odom_dir, f"{odom_stamp}.pkl")
        detection_file = os.path.join(detection_dir, f"{image_stamp}.npz")
        
        # ================ interpolate between two odometry ===================
        # process linear and angular odometry separately
        left_linear_odom = pickle.load(open(left_lidar_odom_file, 'rb'))
        right_linear_odom = pickle.load(open(right_lidar_odom_file, 'rb'))
        angular_odom = pickle.load(open(odom_file, 'rb'))

        linear_left_ratio = (image_linear_stamp - left_lidar_odom_stamp) / (right_lidar_odom_stamp - left_lidar_odom_stamp) if right_lidar_odom_stamp != left_lidar_odom_stamp else 0.5

        odom = {}
        odom['position'] = np.array(right_linear_odom['position']) * linear_left_ratio + np.array(left_linear_odom['position']) * (1 - linear_left_ratio)
        odom['linear_velocity'] = np.array(right_linear_odom['linear_velocity']) * linear_left_ratio + np.array(left_linear_odom['linear_velocity']) * (1 - linear_left_ratio)
        # SLERP
        # rotations = Rotation.from_quat([left_linear_odom['orientation'], right_linear_odom['orientation']])
        # slerp = Slerp([0, 1], rotations)
        # odom['orientation'] = slerp(linear_left_ratio).as_quat()
        odom['orientation'] = angular_odom['orientation']
        odom['angular_velocity'] = angular_odom['angular_velocity']

        # ================= Find neighbouring clouds =================
        cloud_this_frame = np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 3)

        global_cloud = np.vstack((global_cloud, cloud_this_frame))

        cloud_stack.append(cloud_this_frame)
        stamp_stack.append(image_stamp)
        cloud = np.concatenate(cloud_stack, axis=0)
        while stamp_stack[-1] - stamp_stack[0] > 3e8:
            cloud_stack.pop(0)
            stamp_stack.pop(0)

        if last_odom is not None and np.linalg.norm(odom['position'] - last_odom['position']) < 0.1:
            continue
        last_odom = odom

        if abs(odom['angular_velocity'][2]) > 0.2:
            continue

        # ================== load detection ==================
        detections = np.load(detection_file)
        det_labels = detections['labels']
        det_bboxes = detections['bboxes']
        det_confidences = detections['confidences']
        det_masks = detections['masks']

        detections_tracked, unmatched, detection_orig = mapper.track_objects(det_bboxes, det_labels, det_confidences, odom)

        # if unmatched:
        #     print(f"frame {image_stamp} unmatched, det_labels: {detection_orig}")

        detections_tracked['masks'] = []
        for bbox in detections_tracked['bboxes']:
            err = np.linalg.norm(bbox - det_bboxes, axis=1)
            err_mask = (err < 0.1)
            mask_idx = err_mask.argmax()
            detections_tracked['masks'].append(det_masks[mask_idx])

        # ================== update map ==================
        
        mapper.update_map(detections_tracked, image_stamp, odom, cloud)

        # ================= visualize ==================
        
        if VISUALIZE:
            mapper.rerun_vis(odom, regularized=True, show_bbox=True, debug=True)

            # if cnt % 20 == 0:
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(
                global_cloud
            )

            voxel_size = 0.05
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
            global_cloud = np.asarray(merged_pcd.points)
            mapper.rerun_visualizer.visualize_global_pcd(global_cloud) 

        if TO_JSON:
            map_dict = mapper.serialize_map_to_dict(image_stamp)
            import json
            os.makedirs(f'../output/json_serialization_{PLATFORM}', exist_ok=True)
            with open(os.path.join(f'../output/json_serialization_{PLATFORM}', f"{image_stamp}.json"), 'w') as f:
                json.dump(map_dict, f)

        if ANNOTATE:
            image_anno = cv2.imread(image_file)
            image_verbose = image_anno.copy()

            bboxes = detections_tracked['bboxes']
            masks = detections_tracked['masks']
            labels = detections_tracked['labels']
            obj_ids = detections_tracked['ids']
            confidences = detections_tracked['confidences']

            if len(bboxes) > 0:
                image_anno = cv2.cvtColor(image_anno, cv2.COLOR_BGR2RGB)
                class_ids = np.array(list(range(len(labels))))
                annotation_labels = [
                    f"{class_name} {obj_id} {confidence:.2f}"
                    for class_name, obj_id, confidence in zip(
                        labels, obj_ids, confidences
                    )
                ]
                if None in obj_ids:
                    print(f"frame {image_stamp} obj_ids: {obj_ids}")

                detections = sv.Detections(
                    xyxy=np.array(bboxes),
                    mask=np.array(masks).astype(bool),
                    class_id=class_ids,
                )
                image_anno = box_annotator.annotate(scene=image_anno, detections=detections)
                image_anno = label_annotator.annotate(scene=image_anno, detections=detections, labels=annotation_labels)
                image_anno = mask_annotator.annotate(scene=image_anno, detections=detections)
                image_anno = cv2.cvtColor(image_anno, cv2.COLOR_RGB2BGR)

            if len(det_bboxes) > 0:
                image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_BGR2RGB)
                class_ids = np.array(list(range(len(det_labels))))
                annotation_labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence in zip(
                        det_labels, det_confidences
                    )
                ]
                detections = sv.Detections(
                    xyxy=np.array(det_bboxes),
                    class_id=class_ids,
                )
                image_verbose = box_annotator.annotate(scene=image_verbose, detections=detections)
                image_verbose = label_annotator.annotate(scene=image_verbose, detections=detections, labels=annotation_labels)
                image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_RGB2BGR)
                image_verbose = np.vstack((image_verbose, image_anno))

            # draw pcd
            R_b2w = Rotation.from_quat(odom['orientation']).as_matrix()
            t_b2w = np.array(odom['position'])
            R_w2b = R_b2w.T
            t_w2b = -R_w2b @ t_b2w
            cloud_body = cloud @ R_w2b.T + t_w2b

            cloud_image_fusion.generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w, image_src=image_anno)

            cv2.imwrite(os.path.join(ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image_anno)
            cv2.imwrite(os.path.join(VERBOSE_ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image_verbose)
    
    # if not VISUALIZE:
    mapper.rerun_vis(odom, regularized=True, show_bbox=True, debug=False, enforce=True)

def run():
    main()

if __name__=='__main__':
    run()