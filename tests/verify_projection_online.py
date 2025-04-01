import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import supervision as sv
from supervision.draw.color import ColorPalette

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import cv2

from semantic_mapping.cloud_image_fusion import CloudImageFusion

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

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

import semantic_mapping.utils as utils

class MappingNode(Node):
    def __init__(self):
        super().__init__('projection_verification')

        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/registered_scan',
            self.cloud_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/aft_mapped_to_init_incremental',
            self.odom_callback,
            10
        )

        self.mapping_timer = self.create_timer(0.2, self.mapping_callback)

        self.ANNOTATE = True
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
            self.ANNOTATE_OUT_DIR = os.path.join('debug_proj', 'annotated_3d_in_loop')
            if os.path.exists(self.ANNOTATE_OUT_DIR):
                os.system(f"rm -r {self.ANNOTATE_OUT_DIR}")
            os.makedirs(self.ANNOTATE_OUT_DIR, exist_ok=True)

        self.new_detection = False
        self.new_rgb = False

        # stacks for time synchronization
        self.cloud_stack = []
        self.cloud_stamps = []
        self.odom_stack = []
        self.odom_stamps = []
        self.detections_stack = []
        self.detection_stamps = []
        self.rgb_stack = []

        # time compensation parameters
        self.detection_linear_state_time_bias = 0.0
        self.detection_angular_state_time_bias = 0.0

        # image processing interval
        self.image_processing_interval = 0.5 # seconds

        self.bridge = CvBridge()
        self.get_logger().info('Projection verification node has been started.')

        self.cloud_image_fusion = CloudImageFusion(platform='mecanum')

    def image_callback(self, msg):
        # try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            det_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            if len(self.detection_stamps) == 0 or det_stamp - self.detection_stamps[-1] > self.image_processing_interval:
                self.rgb_stack.append(cv_image)
                self.detection_stamps.append(det_stamp)
                while len(self.rgb_stack) > 5:
                    self.detection_stamps.pop(0)
                    self.rgb_stack.pop(0)
                # Publish the processed image
                self.new_detection = True
                self.get_logger().info('Processed an image.')
            else:
                return
        # except Exception as e:
        #     self.get_logger().error(f'Error processing image: {str(e)}')

    def cloud_callback(self, msg):
        self.cloud_stack.append(point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z")))
        stamp_seconds = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.cloud_stamps.append(stamp_seconds)

    def odom_callback(self, msg):
        odom = {}
        odom['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        odom['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        odom['linear_velocity'] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        odom['angular_velocity'] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

        self.odom_stack.append(odom)
        self.odom_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

    def mapping_callback(self):
        if self.new_detection:
            self.new_detection = False
            
            detection_stamp = self.detection_stamps[0]
            image = self.rgb_stack[0]

            # ================== Time synchronization ==================
            if len(self.odom_stamps) == 0:
                return
            linear_state_stamp = detection_stamp + self.detection_linear_state_time_bias
            angular_state_stamp = detection_stamp + self.detection_angular_state_time_bias

            left_odom_stamp, right_odom_stamp = utils.find_neighbouring_stamps(self.odom_stamps, linear_state_stamp)
            if left_odom_stamp > linear_state_stamp: # wait for next detection
                return
            if right_odom_stamp < linear_state_stamp: # wait for odometry
                return

            angular_odom_stamp = utils.find_closest_stamp(self.odom_stamps, angular_state_stamp)
            if abs(angular_odom_stamp - angular_state_stamp) > 0.1:
                return

            # clean up the lidar odom stacks
            while self.odom_stamps[0] < left_odom_stamp:
                self.odom_stack.pop(0)
                self.odom_stamps.pop(0)

            left_lidar_odom = self.odom_stack[self.odom_stamps.index(left_odom_stamp)]
            right_lidar_odom = self.odom_stack[self.odom_stamps.index(right_odom_stamp)]
            imu_odom = self.odom_stack[self.odom_stamps.index(angular_odom_stamp)]

            linear_left_ratio = (detection_stamp - left_odom_stamp) / (right_odom_stamp - left_odom_stamp) if right_odom_stamp != left_odom_stamp else 0.5

            # interpolate for the camera odometry
            camera_odom = {}
            camera_odom['position'] = np.array(right_lidar_odom['position']) * linear_left_ratio + np.array(left_lidar_odom['position']) * (1 - linear_left_ratio)
            camera_odom['linear_velocity'] = np.array(right_lidar_odom['linear_velocity']) * linear_left_ratio + np.array(left_lidar_odom['linear_velocity']) * (1 - linear_left_ratio)
            # SLERP
            rotations = Rotation.from_quat([left_lidar_odom['orientation'], right_lidar_odom['orientation']])
            slerp = Slerp([0, 1], rotations)
            camera_odom['orientation'] = slerp(linear_left_ratio).as_quat()
            camera_odom['angular_velocity'] = imu_odom['angular_velocity']

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
                if self.cloud_stamps[i] >= (detection_stamp - 2.0) and self.cloud_stamps[i] <= (detection_stamp + 2.0):
                    neighboring_cloud.append(self.cloud_stack[i])
            if len(neighboring_cloud) == 0:
                return
            else:
                neighboring_cloud = np.concatenate(neighboring_cloud, axis=0)

            if self.ANNOTATE:                
                # draw pcd
                R_b2w = Rotation.from_quat(camera_odom['orientation']).as_matrix()
                t_b2w = np.array(camera_odom['position'])
                R_w2b = R_b2w.T
                t_w2b = -R_w2b @ t_b2w
                cloud_body = neighboring_cloud @ R_w2b.T + t_w2b

                dummy_masks = np.ones([1, image.shape[0], image.shape[1]])
                dummy_labels = ['dummy']
                dummy_confidences = np.array([0])

                dummy_cloud = self.cloud_image_fusion.generate_seg_cloud(cloud_body, dummy_masks, dummy_labels, dummy_confidences, R_b2w, t_b2w, image_src=image)
                cv2.imwrite(os.path.join(self.ANNOTATE_OUT_DIR, f"{detection_stamp}.png"), image)

if __name__ == "__main__":
    rclpy.init(args=None)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
