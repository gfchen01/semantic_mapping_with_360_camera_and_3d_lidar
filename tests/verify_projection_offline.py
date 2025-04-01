import supervision as sv
from supervision.draw.color import ColorPalette
import os
from tqdm import tqdm
from types import SimpleNamespace
import utils
import pickle
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import cv2
from cloud_image_fusion import CloudImageFusion

def main():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_dir', type=str, required=True)
    args_parser.add_argument('--annotate', action='store_true')
    args_parser.add_argument('--use_lidar_odom', action='store_true')
    args_parser.add_argument('--platform', type=str, default='diablo')
    args = args_parser.parse_args()

    PATH_PREFIX = args.data_dir
    ANNOTATE = args.annotate
    use_lidar_odom = args.use_lidar_odom

    print(f"PATH_PREFIX: {PATH_PREFIX}")
    print(f"ANNOTATE: {ANNOTATE}")
    print(f"use_lidar_odom: {use_lidar_odom}")
    # PATH_PREFIX = "/media/all/easystore/dataset/ros2/extracted/"
    # VISUALIZE = False
    # ANNOTATE = False

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
        ANNOTATE_OUT_DIR = os.path.join('debug_proj', 'annotated_3d_in_loop')
        if os.path.exists(ANNOTATE_OUT_DIR):
            os.system(f"rm -r {ANNOTATE_OUT_DIR}")
        os.makedirs(ANNOTATE_OUT_DIR, exist_ok=True)

        VERBOSE_ANNOTATE_OUT_DIR = os.path.join('debug_mapper', 'verbose_3d_in_loop')
        ORIG_ANNOTATE_OUT_DIR = os.path.join('debug_mapper', 'orig_3d_in_loop')
        if os.path.exists(VERBOSE_ANNOTATE_OUT_DIR):
            os.system(f"rm -r {VERBOSE_ANNOTATE_OUT_DIR}")
        os.makedirs(VERBOSE_ANNOTATE_OUT_DIR, exist_ok=True)
        if os.path.exists(ORIG_ANNOTATE_OUT_DIR):
            os.system(f"rm -r {ORIG_ANNOTATE_OUT_DIR}")
        os.makedirs(ORIG_ANNOTATE_OUT_DIR, exist_ok=True)

    cloud_dir = os.path.join(PATH_PREFIX, 'registered_scan')
    image_dir = os.path.join(PATH_PREFIX, 'image')
    odom_dir = os.path.join(PATH_PREFIX, 'odom')
    detection_dir = os.path.join(PATH_PREFIX, 'annotations')
    if use_lidar_odom:
        lidar_odom_dir = os.path.join(PATH_PREFIX, 'lidar_odom')
    else:
        lidar_odom_dir = odom_dir

    cloud_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(cloud_dir)])
    image_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(image_dir)])
    odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(odom_dir)])
    lidar_odom_stamps = sorted([int(x.split('.')[0]) for x in os.listdir(lidar_odom_dir)])

    cnt = 0
    image_linear_time_bias = 0.2
    image_angular_time_bias = 0.0
    cloud_stack = []
    stamp_stack = []

    args = SimpleNamespace(
        **{
            "track_thresh": 0.2, # +0.1 = thresh for adding new tracklet
            "track_buffer": 5, # number of frames to delete a tracklet
            "match_thresh": 0.85, # 
            "mot20": False,
            "min_box_area": 100,
        }
    )

    cloud_img_fusion = CloudImageFusion(platform=args.platform)

    for cnt, image_stamp in enumerate(tqdm(image_stamps)):
        image_linear_stamp = image_stamp + int(image_linear_time_bias * 1e9)
        image_angular_stamp = image_stamp + int(image_angular_time_bias * 1e9)

        lidar_stamp = utils.find_closest_stamp(cloud_stamps, image_stamp)
        odom_stamp = utils.find_closest_stamp(odom_stamps, image_angular_stamp)
        left_lidar_odom_stamp, right_lidar_odom_stamp = utils.find_neighbouring_stamps(lidar_odom_stamps, image_linear_stamp)

        print(f"image_stamp: {image_stamp/1e9}, odom_stamp: {odom_stamp/1e9}, lidar_stamp: {lidar_stamp/1e9}, lidar_odom_stamp: {left_lidar_odom_stamp/1e9}, right_lidar_odom_stamp: {right_lidar_odom_stamp/1e9}")

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
        rotations = Rotation.from_quat([left_linear_odom['orientation'], right_linear_odom['orientation']])
        slerp = Slerp([0, 1], rotations)
        odom['orientation'] = slerp(linear_left_ratio).as_quat()
        odom['angular_velocity'] = angular_odom['angular_velocity']

        # ================= Find neighbouring clouds =================

        cloud_stack.append(np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 3))
        stamp_stack.append(image_stamp)
        cloud = np.concatenate(cloud_stack, axis=0)
        while stamp_stack[-1] - stamp_stack[0] > 1e9:
            cloud_stack.pop(0)
            stamp_stack.pop(0)

            if ANNOTATE:
                image = cv2.imread(image_file)
                # draw pcd
                R_b2w = Rotation.from_quat(odom['orientation']).as_matrix()
                t_b2w = np.array(odom['position'])
                R_w2b = R_b2w.T
                t_w2b = -R_w2b @ t_b2w
                cloud_body = cloud @ R_w2b.T + t_w2b

                dummy_masks = np.ones([1, image.shape[0], image.shape[1]])
                dummy_labels = ['dummy']
                dummy_confidences = np.array([0])

                cloud_img_fusion.generate_seg_cloud(cloud_body, dummy_masks, dummy_labels, dummy_confidences, R_b2w, t_b2w, image_src=image)
                cv2.imwrite(os.path.join(ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image)

        # if ANNOTATE:
        #     image_anno = cv2.imread(image_file)
        #     image_verbose = image_anno.copy()

        #     bboxes = detections_tracked['bboxes']
        #     masks = detections_tracked['masks']
        #     labels = detections_tracked['labels']
        #     obj_ids = detections_tracked['ids']
        #     confidences = detections_tracked['confidences']

        #     if len(bboxes) > 0:
        #         image_anno = cv2.cvtColor(image_anno, cv2.COLOR_BGR2RGB)
        #         class_ids = np.array(list(range(len(labels))))
        #         annotation_labels = [
        #             f"{class_name} {id} {confidence:.2f}"
        #             for class_name, id, confidence in zip(
        #                 labels, obj_ids, confidences
        #             )
        #         ]
        #         detections = sv.Detections(
        #             xyxy=np.array(bboxes),
        #             mask=np.array(masks).astype(bool),
        #             class_id=class_ids,
        #         )
        #         image_anno = box_annotator.annotate(scene=image_anno, detections=detections)
        #         image_anno = label_annotator.annotate(scene=image_anno, detections=detections, labels=annotation_labels)
        #         image_anno = mask_annotator.annotate(scene=image_anno, detections=detections)
        #         image_anno = cv2.cvtColor(image_anno, cv2.COLOR_RGB2BGR)

        #     if len(det_bboxes) > 0:
        #         image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_BGR2RGB)
        #         class_ids = np.array(list(range(len(det_labels))))
        #         annotation_labels = [
        #             f"{class_name} {confidence:.2f}"
        #             for class_name, confidence in zip(
        #                 det_labels, det_confidences
        #             )
        #         ]
        #         detections = sv.Detections(
        #             xyxy=np.array(det_bboxes),
        #             class_id=class_ids,
        #         )
        #         image_verbose = box_annotator.annotate(scene=image_verbose, detections=detections)
        #         image_verbose = label_annotator.annotate(scene=image_verbose, detections=detections, labels=annotation_labels)
        #         image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_RGB2BGR)
        #         image_verbose = np.vstack((image_verbose, image_anno))

        #     # draw pcd
        #     from generate_seg_cloud import generate_seg_cloud
        #     R_b2w = Rotation.from_quat(odom['orientation']).as_matrix()
        #     t_b2w = np.array(odom['position'])
        #     R_w2b = R_b2w.T
        #     t_w2b = -R_w2b @ t_b2w
        #     cloud_body = cloud @ R_w2b.T + t_w2b

        #     generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w, platform='mecanum', image_src=image_anno)

        #     cv2.imwrite(os.path.join(ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image_anno)
        #     cv2.imwrite(os.path.join(VERBOSE_ANNOTATE_OUT_DIR, f"{image_stamp}.png"), image_verbose)

if __name__=="__main__":
    main()