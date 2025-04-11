#!/usr/bin/env python
# coding: utf-8

import json
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
import numpy as np
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import supervision as sv
from supervision.draw.color import ColorPalette


from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from bytetrack.byte_tracker import BYTETracker
from types import SimpleNamespace


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

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String

import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
import open3d as o3d

from .utils import find_closest_stamp, find_neighbouring_stamps
from .semantic_map import ObjMapper
from .tools import ros2_bag_utils
from .cloud_image_fusion import CloudImageFusion

import yaml
import sys
from pathlib import Path

captioner_not_found = False
try:
    captioner_src_path = Path(__file__).resolve().parents[1] / "ai_module" / "src" / "captioner" / "src"
    sys.path.append(str(captioner_src_path))
    from captioning_backend import Captioner
except ModuleNotFoundError:
    captioner_not_found = True
    print(f"Captioner not found. Fall back to no captioning version.")

class MappingNode(Node):
    def __init__(self, config, mask_predictor, grounding_processor, grounding_model, tracker):
        super().__init__('semantic_mapping_node')

        # class global containers
        self.cloud_stack = []
        self.cloud_stamps = []
        self.odom_stack = []
        self.odom_stamps = []
        self.detections_stack = []
        self.detection_stamps = []
        self.rgb_stack = []
        self.lidar_odom_stack = []
        self.lidar_odom_stamps = []
        self.global_cloud = np.empty([0, 3])

        # class global last states
        self.new_detection = False
        self.new_rgb = False
        self.last_camera_odom = None
        self.last_vis_stamp = 0.0

        self.odom_cbk_lock = threading.Lock()
        self.lidar_odom_cbk_lock = threading.Lock()
        self.cloud_cbk_lock = threading.Lock()
        self.rgb_cbk_lock = threading.Lock()
        self.mapping_processing_lock = threading.Lock()

        # parameters
        self.platform = config.get('platform', 'mecanum')
        self.use_lidar_odom = config.get('use_lidar_odom', False)
        # time compensation parameters
        self.detection_linear_state_time_bias = config.get('detection_linear_state_time_bias', 0.0)
        self.detection_angular_state_time_bias = config.get('detection_angular_state_time_bias', 0.0)
        # image processing interval
        self.image_processing_interval = config.get('image_processing_interval', 0.5) # seconds
        # visualization settings
        self.vis_interval = config.get('vis_interval', 1.0) # seconds
        self.ANNOTATE = config['annotate_image']

        print(
            f'Platform: {self.platform}\n,\
                Use lidar odometry: {self.use_lidar_odom}\n,\
                Detection linear state time bias: {self.detection_linear_state_time_bias}\n,\
                Detection angular state time bias: {self.detection_angular_state_time_bias}\n,\
                Image processing interval: {self.image_processing_interval}\n,\
                Visualization interval: {self.vis_interval}\n,\
                Annotate image: {self.ANNOTATE}'
        )

        self.mask_predictor = mask_predictor
        self.grounding_processor = grounding_processor
        self.grounding_model = grounding_model

        self.label_template = config['prompts']
        self.text_prompt = []
        for value in self.label_template.values():
            self.text_prompt += value['prompts']
        self.text_prompt = " . ".join(self.text_prompt) + " ."
        print(f"Text prompt: {self.text_prompt}")

        # ROS2 subscriptions and publishers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/registered_scan',
            self.cloud_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        if self.use_lidar_odom:
            self.lidar_odom_sub = self.create_subscription(
                Odometry,
                '/aft_mapped_to_init_incremental',
                self.lidar_odom_callback,
                10,
                callback_group=MutuallyExclusiveCallbackGroup()
            )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/state_estimation',
            self.odom_callback,
            100,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.query_sub = self.create_subscription(
            String, 
            '/object_query', 
            self.handle_object_query, 
            1, 
            # callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        self.caption_pub = self.create_publisher(String, '/queried_captions', 10) # TODO: Server instead of pub?

        self.mapping_timer = self.create_timer(0.1, self.mapping_callback)

        self.caption_pub_timer = self.create_timer(0.1, self.publish_queried_captions)
        self.obj_cloud_pub = self.create_publisher(PointCloud2, '/obj_points', 10)
        self.obj_box_pub = self.create_publisher(MarkerArray, '/obj_boxes', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)


        self.queried_captions = None

        self.do_visualize = True

        if captioner_not_found:
            self.captioner = None
        else:
            self.captioner = Captioner(
                semantic_dict={},
                log_info=self.get_logger().info,
                load_captioner=True
            )
        
        self.cloud_img_fusion = CloudImageFusion(platform=self.platform)

        self.obj_mapper = ObjMapper(tracker=tracker, 
                                    cloud_image_fusion=self.cloud_img_fusion, 
                                    label_template=self.label_template, 
                                    captioner=self.captioner, 
                                    visualize=self.do_visualize)

        if self.ANNOTATE:
            self.box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
            self.label_annotator = sv.LabelAnnotator(
                color=ColorPalette.DEFAULT,
                text_padding=4,
                text_scale=0.3,
                text_position=sv.Position.TOP_LEFT,
                color_lookup=sv.ColorLookup.INDEX,
                smart_position=True,
            )
            self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
            self.ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'annotated_3d_in_loop')
            if os.path.exists(self.ANNOTATE_OUT_DIR):
                os.system(f"rm -r {self.ANNOTATE_OUT_DIR}")
            os.makedirs(self.ANNOTATE_OUT_DIR, exist_ok=True)

            self.VERBOSE_ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'verbose_3d_in_loop')
            if os.path.exists(self.VERBOSE_ANNOTATE_OUT_DIR):
                os.system(f"rm -r {self.VERBOSE_ANNOTATE_OUT_DIR}")
            os.makedirs(self.VERBOSE_ANNOTATE_OUT_DIR, exist_ok=True)


        self.bridge = CvBridge()
        self.get_logger().info('Semantic mapping node has been started.')

    def inference(self, cv_image):
        """
        Perform open-vocabulary semantic inference on the input image.

        cv_image: np.ndarray, shape (H, W, 3), BGR format
        """
        image = cv_image[:, :, ::-1]  # BGR to RGB
        image = image.copy()

        inputs = self.grounding_processor(
            images=image,
            text=self.text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.35,
            text_threshold=0.35,
            target_sizes=[image.shape[:2]],
        )

        class_names = np.array(results[0]["labels"])
        bboxes = results[0]["boxes"].cpu().numpy()  # (n_boxes, 4)
        confidences = results[0]["scores"].cpu().numpy()  # (n_boxes,)
                
        det_result = {
            "bboxes": bboxes,
            "labels": class_names,
            "confidences": confidences,
        }

        return det_result

    def image_callback(self, msg):
        with self.rgb_cbk_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            det_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            if len(self.detection_stamps) == 0 or det_stamp - self.detection_stamps[-1] > self.image_processing_interval:
                self.rgb_stack.append(cv_image)
                # det_result = self.inference(cv_image)
                # self.detections_stack.append(det_result)
                self.detection_stamps.append(det_stamp)
                while len(self.rgb_stack) > 10:
                    self.detection_stamps.pop(0)
                    # self.detections_stack.pop(0)
                    self.rgb_stack.pop(0)
                self.new_detection = True
            else:
                return
            
            # print('processed image: ', det_stamp)

    def cloud_callback(self, msg):
        with self.cloud_cbk_lock:
            points_numpy = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
            self.cloud_stack.append(points_numpy)
            stamp_seconds = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.cloud_stamps.append(stamp_seconds)

            self.global_cloud = np.vstack([self.global_cloud, points_numpy])
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(
                self.global_cloud
            )

            voxel_size = 0.05
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size)

            self.global_cloud = np.asarray(merged_pcd.points)

    def lidar_odom_callback(self, msg):
        with self.lidar_odom_cbk_lock:
            odom = {}
            odom['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            odom['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            odom['linear_velocity'] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            odom['angular_velocity'] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

            self.lidar_odom_stack.append(odom)
            self.lidar_odom_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

            # print(f"lidar odom stamp: {self.lidar_odom_stamps[-1]}")

    def odom_callback(self, msg):
        with self.odom_cbk_lock:
            odom = {}
            odom['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            odom['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            odom['linear_velocity'] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            odom['angular_velocity'] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

            self.odom_stack.append(odom)
            self.odom_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

            # print(f"odom stamp: {self.odom_stamps[-1]}")

    def handle_object_query(self, query_str: String):
        query_list = json.loads(query_str.data)
        self.queried_captions = None # To stop publishing the captions in the other thread (TODO: improve the way concurrency in handled in the entire system)
        self.queried_captions = self.captioner.query_clip_features(query_list)
        self.get_logger().info(f'{self.queried_captions}')

    def mapping_processing(self, image, camera_odom, detections, detection_stamp, neighboring_cloud):
        with self.mapping_processing_lock:
            # ================== Process detection and tracking ==================
            if detections is None:
                detections = self.inference(image)

            det_labels = detections['labels']
            det_bboxes = detections['bboxes']
            det_confidences = detections['confidences']
            
            detections_tracked, _, _ = self.obj_mapper.track_objects(det_bboxes, det_labels,det_confidences, camera_odom)

            # ================== Infer Masks ==================
            # sam2
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mask_predictor.set_image(image)

                if len(detections_tracked['bboxes']) > 0:
                    masks, _, _ = mask_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array(detections_tracked['bboxes']),
                        multimask_output=False,
                    )

                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                    
                    detections_tracked['masks'] = masks
                else: # no information need to add to map
                    # detections_tracked['masks'] = []
                    return

            if self.ANNOTATE:
                image_anno = image.copy()
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
                        f"{class_name} {id} {confidence:.2f}"
                        for class_name, id, confidence in zip(
                            labels, obj_ids, confidences
                        )
                    ]
                    detections = sv.Detections(
                        xyxy=np.array(bboxes),
                        mask=np.array(masks).astype(bool),
                        class_id=class_ids,
                    )
                    image_anno = self.box_annotator.annotate(scene=image_anno, detections=detections)
                    image_anno = self.label_annotator.annotate(scene=image_anno, detections=detections, labels=annotation_labels)
                    image_anno = self.mask_annotator.annotate(scene=image_anno, detections=detections)
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
                    image_verbose = self.box_annotator.annotate(scene=image_verbose, detections=detections)
                    image_verbose = self.label_annotator.annotate(scene=image_verbose, detections=detections, labels=annotation_labels)
                    image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_RGB2BGR)
                    image_verbose = np.vstack((image_verbose, image_anno))

                # draw pcd
                R_b2w = Rotation.from_quat(camera_odom['orientation']).as_matrix()
                t_b2w = np.array(camera_odom['position'])
                R_w2b = R_b2w.T
                t_w2b = -R_w2b @ t_b2w
                cloud_body = neighboring_cloud @ R_w2b.T + t_w2b
                
                self.cloud_img_fusion.generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w, image_src=image_anno)

                cv2.imwrite(os.path.join(self.ANNOTATE_OUT_DIR, f"{detection_stamp}.png"), image_anno)
                cv2.imwrite(os.path.join(self.VERBOSE_ANNOTATE_OUT_DIR, f"{detection_stamp}.png"), image_verbose)

                # ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                # seconds = int(detection_stamp)
                # nanoseconds = int((detection_stamp - seconds) * 1e9)
                # ros_image.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
                # self.annotated_image_pub.publish(ros_image)

                # cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

            # ================== Update the map ==================

            self.obj_mapper.update_map(detections_tracked, detection_stamp, camera_odom, neighboring_cloud)

            # self.publish_map(detection_stamp)

            if self.do_visualize:
                if detection_stamp - self.last_vis_stamp > self.vis_interval:
                    self.last_vis_stamp = detection_stamp
                    self.obj_mapper.rerun_vis(camera_odom, regularized=True, show_bbox=True, debug=False)
                    # self.obj_mapper.rerun_visualizer.visualize_global_pcd(self.global_cloud) 
                    # self.obj_mapper.rerun_visualizer.visualize_local_pcd_with_mesh(np.concatenate(self.cloud_stack, axis=0))
            
            if self.captioner is not None:
                bboxes_2d = []
                obj_ids_global = []
                centroids_3d = []
                class_names = []
                for i, obj_id in enumerate(detections_tracked['ids']):
                    if obj_id is None or obj_id < 0:
                        continue
                    else:
                        obj_ids_global.append(obj_id)
                        bboxes_2d.append(detections_tracked['bboxes'][i])
                        centroids_3d.append(detections_tracked['centroids_3d'][i])
                        class_names.append(detections_tracked['class_names'][i])

                self.captioner.update_object_crops(
                    torch.from_numpy(image).cuda().flip((-1)),
                    bboxes_2d,
                    obj_ids_global,
                    centroids_3d,
                    class_names,
                )

    def mapping_callback(self):
        if self.new_detection:
            self.new_detection = False
            
            with self.rgb_cbk_lock:
                # detections = self.detections_stack[0]
                detections = None
                detection_stamp = self.detection_stamps[0]
                image = self.rgb_stack[0].copy()

            # ================== Time synchronization ==================
            with self.odom_cbk_lock:
                with self.lidar_odom_cbk_lock:
                    det_linear_state_stamp = detection_stamp + self.detection_linear_state_time_bias
                    det_angular_state_stamp = detection_stamp + self.detection_angular_state_time_bias

                    linear_state_stamps = self.lidar_odom_stamps if self.use_lidar_odom else self.odom_stamps
                    angular_state_stamps = self.odom_stamps
                    linear_states = self.lidar_odom_stack if self.use_lidar_odom else self.odom_stack
                    angular_states = self.odom_stack
                    if len(linear_state_stamps) == 0 or len(angular_state_stamps) == 0:
                        print("No odometry found. Waiting for odometry...")
                        return

                    target_left_odom_stamp, target_right_odom_stamp = find_neighbouring_stamps(linear_state_stamps, det_linear_state_stamp)
                    if target_left_odom_stamp > det_linear_state_stamp: # wait for next detection
                        print("Detection older than oldest odom. Waiting for next detection...")
                        return
                    if target_right_odom_stamp < det_linear_state_stamp: # wait for odometry
                        print(f"Odom older than detection. Right odom: {target_right_odom_stamp}, det linear: {det_linear_state_stamp}. Waiting for odometry...")
                        return

                    target_angular_odom_stamp = find_closest_stamp(angular_state_stamps, det_angular_state_stamp)
                    if abs(target_angular_odom_stamp - det_angular_state_stamp) > 0.1:
                        print(f"No close angular state found. Angular odom found: {target_angular_odom_stamp}, det angular: {det_angular_state_stamp}. Waiting for odometry...")
                        return

                    left_linear_odom = linear_states[linear_state_stamps.index(target_left_odom_stamp)]
                    right_linear_odom = linear_states[linear_state_stamps.index(target_right_odom_stamp)]
                    angular_odom = angular_states[angular_state_stamps.index(target_angular_odom_stamp)]

                    linear_left_ratio = (det_linear_state_stamp - target_left_odom_stamp) / (target_right_odom_stamp - target_left_odom_stamp) if target_right_odom_stamp != target_left_odom_stamp else 0.5

                    assert linear_left_ratio <= 1.0 and linear_left_ratio >= 0.0
                    print(f"linear_left_ratio: {linear_left_ratio}, target_left_odom_stamp: {target_left_odom_stamp}, target_right_odom_stamp: {target_right_odom_stamp}, det_linear_state_stamp: {det_linear_state_stamp}")
                    # print(f'left odom stamp index: {linear_state_stamps.index(target_left_odom_stamp)}, right odom stamp index: {linear_state_stamps.index(target_right_odom_stamp)}, angular odom stamp index: {angular_state_stamps.index(target_angular_odom_stamp)}')

                    # interpolate for the camera odometry
                    camera_odom = {}
                    camera_odom['position'] = np.array(right_linear_odom['position']) * linear_left_ratio + np.array(left_linear_odom['position']) * (1 - linear_left_ratio)
                    camera_odom['linear_velocity'] = np.array(right_linear_odom['linear_velocity']) * linear_left_ratio + np.array(left_linear_odom['linear_velocity']) * (1 - linear_left_ratio)
                    # SLERP
                    rotations = Rotation.from_quat([left_linear_odom['orientation'], right_linear_odom['orientation']])
                    slerp = Slerp([0, 1], rotations)
                    camera_odom['orientation'] = slerp(linear_left_ratio).as_quat()
                    camera_odom['angular_velocity'] = angular_odom['angular_velocity']

                    # clean up the odom stacks
                    while linear_state_stamps[0] < target_left_odom_stamp:
                        linear_states.pop(0)
                        linear_state_stamps.pop(0)
                    if self.use_lidar_odom: # two stamp reference point to different containers
                        while angular_state_stamps[0] < target_angular_odom_stamp:
                            angular_states.pop(0)
                            angular_state_stamps.pop(0)

            # ================== Find the cloud collected around rgb timestamp ==================
            with self.cloud_cbk_lock:
                if len(self.cloud_stamps) == 0:
                    return
                while len(self.cloud_stamps) > 0 and self.cloud_stamps[0] < (detection_stamp - 1.0):
                    self.cloud_stack.pop(0)
                    self.cloud_stamps.pop(0)
                    if len(self.cloud_stack) == 0:
                        return

                neighboring_cloud = []
                for i in range(len(self.cloud_stamps)):
                    if self.cloud_stamps[i] >= (detection_stamp - 1.0) and self.cloud_stamps[i] <= (detection_stamp + 0.1):
                        neighboring_cloud.append(self.cloud_stack[i])
                if len(neighboring_cloud) == 0:
                    return
                else:
                    neighboring_cloud = np.concatenate(neighboring_cloud, axis=0)

            # if self.last_camera_odom is not None:
            #     if np.linalg.norm(self.last_camera_odom['position'] - camera_odom['position']) < 0.05:
            #         return
            
            self.last_camera_odom = camera_odom

            threading.Thread(target=self.mapping_processing, args=(image, camera_odom, detections, detection_stamp, neighboring_cloud)).start()

            # self.mapping_processing(image, camera_odom, detections, detection_stamp, neighboring_cloud)

    def publish_map(self, stamp):
        seconds = int(stamp)
        nanoseconds = int((stamp - seconds) * 1e9)

        marker_array_msg = MarkerArray()
        marker_array = []

        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds-1e4).to_msg()
        clear_marker.action = Marker.DELETEALL
        marker_array.append(clear_marker)

        map_vis_msgs = self.obj_mapper.to_ros2_msgs(stamp)
        
        for msg in map_vis_msgs:
            if isinstance(msg, PointCloud2):
                self.obj_cloud_pub.publish(msg)
            elif isinstance(msg, Marker):
                marker_array.append(msg)
            else:
                self.get_logger().error('[In map vis]: Unknown message type.')

        if len(marker_array) > 1:
            marker_array_msg.markers = marker_array
            self.obj_box_pub.publish(marker_array_msg)

    def publish_queried_captions(self):
        if self.queried_captions is None:
            return
        queried_caption_str = json.dumps(self.queried_captions)
        self.caption_pub.publish(String(data=queried_caption_str))

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

    byte_tracker_args = SimpleNamespace(
        **{
            "track_thresh": 0.2, # +0.1 = thresh for adding new tracklet
            "track_buffer": 5, # number of frames to delete a tracklet
            "match_thresh": 0.85, # 
            "mot20": False,
            "min_box_area": 100,
        }
    )
    tracker = BYTETracker(byte_tracker_args)
    
    rclpy.init(args=None)
    node = MappingNode(config, mask_predictor, grounding_processor, grounding_model, tracker)
    
    # executor = MultiThreadedExecutor(num_threads=6)
    # executor.add_node(node)

    try:
        rclpy.spin(node)
        # executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
