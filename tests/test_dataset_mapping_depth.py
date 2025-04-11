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
    ANNOTATE = True
    TO_JSON = True

    image_dir = os.path.join(PATH_PREFIX, 'image')
    image_odom_dir = os.path.join(PATH_PREFIX, 'image_odom')
    detection_dir = os.path.join(PATH_PREFIX, 'original_annotations')
    depth_dir = os.path.join(PATH_PREFIX, 'depth_image')

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


    image_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(image_dir)])
    image_odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(image_odom_dir)])
    
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
        if image_stamp in image_odom_stamps:
            image_odom_idx = image_odom_stamps.index(image_stamp)
        else:
            print('image not processed')
            continue

        image_file = os.path.join(image_dir, f"{image_stamp}.png")
        depth_file = os.path.join(depth_dir, f"{image_stamp}.tiff")
        detection_file = os.path.join(detection_dir, f"{image_stamp}.npz")
        image_odom_file = os.path.join(image_odom_dir, f"{image_stamp}.pkl")
        
        # ================ interpolate between two odometry ===================
        odom = pickle.load(open(image_odom_file, 'rb'))
        if last_odom is not None and np.linalg.norm(odom['position'] - last_odom['position']) < 0.1:
            continue
        else:
            last_odom = odom

        if abs(odom['angular_velocity'][2]) > 0.2:
            continue

        # ================= Find neighbouring clouds =================
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

        # global_cloud = np.vstack((global_cloud, cloud_this_frame))

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
        
        mapper.update_map_depth(detections_tracked, image_stamp, odom, depth)

        # ================= visualize ==================
        
        if VISUALIZE:
            mapper.rerun_vis(odom, regularized=True, show_bbox=True, debug=True)

            if cnt % 20 == 0:
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
            json_output_dir = f'{os.path.dirname(os.path.realpath(__file__))}/output/json_serialization_{PLATFORM}'
            os.makedirs(json_output_dir, exist_ok=True)
            with open(os.path.join(json_output_dir, f"{image_stamp}.json"), 'w') as f:
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

            cv2.imwrite(os.path.join(ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image_anno)
            cv2.imwrite(os.path.join(VERBOSE_ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image_verbose)
    
    # if not VISUALIZE:
    mapper.rerun_vis(odom, regularized=True, show_bbox=True, debug=False, enforce=True)

def run():
    main()

if __name__=='__main__':
    run()