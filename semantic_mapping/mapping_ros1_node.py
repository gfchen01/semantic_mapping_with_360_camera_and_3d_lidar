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
from . import point_cloud2 as pc2
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

from .semantic_map import ObjMapper
from .utils import find_closest_stamp, find_neighbouring_stamps
from .cloud_image_fusion import CloudImageFusion
from .tools import ros1_bag_utils

captioner_not_found = False
try:
    captioner_src_path = Path(__file__).resolve().parents[1] / "ai_module" / "src" / "captioner" / "src"
    sys.path.append(str(captioner_src_path))
    from captioning_backend import Captioner
except ModuleNotFoundError:
    captioner_not_found = True
    print(f"Captioner not found. Fall back to no captioning version.")


class MappingNode:
    def __init__(self, config, mask_predictor, grounding_processor, grounding_model, tracker):
        rospy.init_node('semantic_mapping')

        # stacks for time synchronization
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
            self.ANNOTATE_OUT_DIR = os.path.join('debug_mapper', 'annotated_3d_in_loop')
            if os.path.exists(self.ANNOTATE_OUT_DIR):
                os.system(f"rm -r {self.ANNOTATE_OUT_DIR}")
            os.makedirs(self.ANNOTATE_OUT_DIR, exist_ok=True)

            self.VERBOSE_ANNOTATE_OUT_DIR = os.path.join('debug_mapper', 'verbose_3d_in_loop')
            if os.path.exists(self.VERBOSE_ANNOTATE_OUT_DIR):
                os.system(f"rm -r {self.VERBOSE_ANNOTATE_OUT_DIR}")
            os.makedirs(self.VERBOSE_ANNOTATE_OUT_DIR, exist_ok=True)

        # self.obj_cloud_pub = self.create_publisher(PointCloud2, '/obj_points', 10)
        # self.obj_box_pub = self.create_publisher(MarkerArray, '/obj_boxes', 10)
        # self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)

        # image processing interval
        # odom move dist thresh for new processing
        self.odom_move_dist_thresh = 0.05
        self.last_cam_odom = None

        self.bridge = CvBridge()

        # Pubs and Subs

        self.rgb_sub = rospy.Subscriber(
            '/camera/image',
            Image,
            self.image_callback,
            queue_size=1,
        )

        self.cloud_sub = rospy.Subscriber(
            '/registered_scan',
            PointCloud2,
            self.cloud_callback,
            queue_size=20,
        )

        if self.use_lidar_odom:
            self.lidar_odom_sub = rospy.Subscriber(
                '/aft_mapped_to_init_incremental',
                Odometry,
                self.lidar_odom_callback,
                queue_size=20,
            )

        self.odom_sub = rospy.Subscriber(
            '/state_estimation',
            Odometry,
            self.odom_callback,
            queue_size=100,
        )

        self.overall_map_sub = rospy.Subscriber(
            '/overall_map',
            PointCloud2,
            self.overall_map_callback,
            queue_size=20,
        )

        self.query_sub = rospy.Subscriber(
            '/object_query',
            String,
            self.handle_object_query,
            queue_size=1
        )

        self.caption_pub = rospy.Publisher('/queried_captions', String, queue_size=1)

        self.semantic_cloud_pub = rospy.Publisher('/semantic_cloud', PointCloud2, queue_size=1)
        self.global_cloud_pub = rospy.Publisher('/global_cloud', PointCloud2, queue_size=1)
        self.bbox3d_pub = rospy.Publisher('/bbox3d', MarkerArray, queue_size=1)
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=1)

        self.mapping_timer = rospy.Timer(rospy.Duration(1.5), self.mapping_callback)
        self.caption_pub_timer = rospy.Timer(rospy.Duration(0.1), self.publish_queried_captions)

        rospy.loginfo('Semantic mapping node has been started')

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
        # try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            det_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
            if len(self.detection_stamps) == 0 or det_stamp - self.detection_stamps[-1] > self.image_processing_interval:
                self.rgb_stack.append(cv_image)
                # det_result = self.inference(cv_image)
                # self.detections_stack.append(det_result)
                self.detection_stamps.append(det_stamp)
                while len(self.detections_stack) > 5:
                    self.detection_stamps.pop(0)
                    # self.detections_stack.pop(0)
                    self.rgb_stack.pop(0)
                # Publish the processed image
                self.new_detection = True
                # rospy.loginfo('Processed an image.')
            else:
                return
        # except Exception as e:
        #     self.get_logger().error(f'Error processing image: {str(e)}')

    def cloud_callback(self, msg):
        pcd = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_numpy = np.array(pcd)
        self.cloud_stack.append(points_numpy)
        stamp_seconds = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
        self.cloud_stamps.append(stamp_seconds)

    def overall_map_callback(self, msg):
        pcd = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.global_cloud = np.array(pcd)

    def lidar_odom_callback(self, msg):
        odom = {}
        odom['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        odom['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        odom['linear_velocity'] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        odom['angular_velocity'] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

        self.lidar_odom_stack.append(odom)
        self.lidar_odom_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

    def odom_callback(self, msg):
        odom = {}
        odom['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        odom['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        odom['linear_velocity'] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        odom['angular_velocity'] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

        self.odom_stack.append(odom)
        self.odom_stamps.append(msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9)

    def handle_object_query(self, query_str: String):
        query_list = json.loads(query_str.data)
        self.queried_captions = None # To stop publishing the captions in the other thread (TODO: improve the way concurrency in handled in the entire system)
        self.queried_captions = self.captioner.query_clip_features(query_list)
        rospy.loginfo(f'{self.queried_captions}')

    def mapping_callback(self, event):
        if self.new_detection:
            self.new_detection = False
            
            if len(self.detection_stamps) < 2:
                return
            
            detection_stamp = self.detection_stamps[-2]
            

            # ================== Time synchronization ==================
            if len(self.odom_stamps) == 0:
                return
            det_linear_state_stamp = detection_stamp + self.detection_linear_state_time_bias
            det_angular_state_stamp = detection_stamp + self.detection_angular_state_time_bias

            linear_state_stamps = self.lidar_odom_stamps if self.use_lidar_odom else self.odom_stamps
            angular_state_stamps = self.odom_stamps
            linear_states = self.lidar_odom_stack if self.use_lidar_odom else self.odom_stack
            angular_states = self.odom_stack
            if len(linear_state_stamps) == 0 or len(angular_state_stamps) == 0:
                print("Waiting for odometry...")
                return

            target_left_odom_stamp, target_right_odom_stamp = find_neighbouring_stamps(linear_state_stamps, det_linear_state_stamp)
            if target_left_odom_stamp > det_linear_state_stamp: # wait for next detection
                return
            if target_right_odom_stamp < det_linear_state_stamp: # wait for odometry
                return

            target_angular_odom_stamp = find_closest_stamp(angular_state_stamps, det_angular_state_stamp)
            if abs(target_angular_odom_stamp - det_angular_state_stamp) > 0.1:
                return

            # clean up the odom stacks
            if self.use_lidar_odom: # two stamp reference point to different containers
                while angular_state_stamps[0] < target_angular_odom_stamp:
                    angular_states.pop(0)
                    angular_state_stamps.pop(0)

            while linear_state_stamps[0] < min(target_left_odom_stamp, target_angular_odom_stamp):
                linear_states.pop(0)
                linear_state_stamps.pop(0)

            left_linear_odom = self.odom_stack[linear_state_stamps.index(target_left_odom_stamp)]
            right_linear_odom = self.odom_stack[linear_state_stamps.index(target_right_odom_stamp)]
            angular_odom = self.odom_stack[angular_state_stamps.index(target_angular_odom_stamp)]

            linear_left_ratio = (det_linear_state_stamp - target_left_odom_stamp) / (target_right_odom_stamp - target_left_odom_stamp) if target_right_odom_stamp != target_left_odom_stamp else 0.5
            assert linear_left_ratio >= 0 and linear_left_ratio <= 1
            # print(f"linear_left_ratio: {linear_left_ratio}, target_left_odom_stamp: {target_left_odom_stamp}, target_right_odom_stamp: {target_right_odom_stamp}, det_linear_state_stamp: {det_linear_state_stamp}")

            # interpolate for the camera odometry
            camera_odom = {}
            camera_odom['position'] = np.array(right_linear_odom['position']) * linear_left_ratio + np.array(left_linear_odom['position']) * (1 - linear_left_ratio)
            camera_odom['linear_velocity'] = np.array(right_linear_odom['linear_velocity']) * linear_left_ratio + np.array(left_linear_odom['linear_velocity']) * (1 - linear_left_ratio)
            # SLERP
            rotations = Rotation.from_quat([left_linear_odom['orientation'], right_linear_odom['orientation']])
            slerp = Slerp([0, 1], rotations)
            camera_odom['orientation'] = slerp(linear_left_ratio).as_quat()
            camera_odom['angular_velocity'] = angular_odom['angular_velocity']

            # ================== Find the cloud collected around rgb timestamp ==================
            if len(self.cloud_stamps) == 0:
                return
            while len(self.cloud_stamps) > 0 and self.cloud_stamps[0] < (detection_stamp - 0.4):
                self.cloud_stack.pop(0)
                self.cloud_stamps.pop(0)
                if len(self.cloud_stack) == 0:
                    return

            neighboring_cloud = []
            for i in range(len(self.cloud_stamps)):
                if self.cloud_stamps[i] >= (detection_stamp - 0.3) and self.cloud_stamps[i] <= (detection_stamp + 0.3):
                    neighboring_cloud.append(self.cloud_stack[i])
            if len(neighboring_cloud) == 0:
                return
            else:
                neighboring_cloud = np.concatenate(neighboring_cloud, axis=0)

            if self.last_cam_odom is not None:
                if np.linalg.norm(self.last_cam_odom['position'] - camera_odom['position']) < self.odom_move_dist_thresh:
                    return
            
            self.last_cam_odom = camera_odom

            # ================== Process detection and tracking ==================
            time_start = time.time()

            image = self.rgb_stack[-2].copy()

            inference_start = time.time()
            detections = self.inference(image)
            inference_time = time.time() - inference_start

            det_labels = detections['labels']
            det_bboxes = detections['bboxes']
            det_confidences = detections['confidences']
            
            detections_tracked, _, _ = self.obj_mapper.track_objects(det_bboxes, det_labels,det_confidences, camera_odom)

            # ================== Infer Masks ==================
            # sam2
            mask_start = time.time()
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
            mask_time = time.time() - mask_start

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

            mapping_start = time.time()
            self.obj_mapper.update_map(detections_tracked, detection_stamp, camera_odom, neighboring_cloud)
            mapping_time = time.time() - mapping_start

            # ================== Publish the map to ros ==============
            self.publish_map(detection_stamp, global_cloud=self.global_cloud)

            if self.do_visualize:
                if detection_stamp - self.last_vis_stamp > self.vis_interval:
                    self.last_vis_stamp = detection_stamp
                    self.obj_mapper.rerun_vis(camera_odom, regularized=True, show_bbox=True, debug=False)
                    self.obj_mapper.rerun_visualizer.visualize_global_pcd(self.global_cloud) 
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

            print(f"Time taken for mapping cbk: {time.time() - time_start:.4f} seconds, inference: {inference_time:.4f} seconds, mask: {mask_time:.4f} seconds, mapping: {mapping_time:.4f} seconds")
            # print(f"Time taken for mapping cbk: {time.time() - time_start:.4f} seconds, inference: {inference_time:.4f} seconds, mask: {mask_time:.4f} seconds")


    def publish_map(self, stamp, odom=None, global_cloud=None):
        marker_array_msg = MarkerArray()
        marker_array = []

        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = rospy.Time.from_sec(stamp)
        clear_marker.action = Marker.DELETEALL
        marker_array.append(clear_marker)

        map_vis_msgs = self.obj_mapper.to_ros1_msgs(stamp)
        
        for msg in map_vis_msgs:
            if isinstance(msg, PointCloud2):
                self.semantic_cloud_pub.publish(msg)
            elif isinstance(msg, Marker):
                marker_array.append(msg)
            # else:
            #     rospy.logerror('[In map vis]: Unknown message type.')

        if len(marker_array) > 1:
            marker_array_msg.markers = marker_array
            self.bbox3d_pub.publish(marker_array_msg)

        if odom is not None:
            vehicle_tf = ros1_bag_utils.create_tf_msg(odom, stamp, frame_id="map", child_frame_id="sensor")
            self.tf_pub.publish(vehicle_tf)

        if global_cloud is not None:
            point_cloud = ros1_bag_utils.create_point_cloud(global_cloud, stamp, frame_id="map")
            self.global_cloud_pub.publish(point_cloud)

    def publish_queried_captions(self, event):
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
    
    node = MappingNode(config, mask_predictor, grounding_processor, grounding_model, tracker)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down ROS node.")
