import numpy as np
import open3d as o3d
import os
import cv2
import supervision as sv
from supervision.draw.color import ColorPalette
import time
from tqdm import tqdm
import yaml
from concurrent.futures import ProcessPoolExecutor

# inference infrastructure
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from bytetrack.byte_tracker import BYTETracker
from types import SimpleNamespace

from scipy.spatial.transform import Rotation

from semantic_mapping import scannet_utils
from semantic_mapping.cloud_image_fusion import CloudImageFusion
from semantic_mapping.semantic_map import ObjMapper


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

mask_predictor, grounding_processor, grounding_model = None, None, None

def load_models(
    dino_id="IDEA-Research/grounding-dino-base", sam2_id="facebook/sam2-hiera-large"
):
    mask_predictor = SAM2ImagePredictor.from_pretrained(sam2_id, device=device)
    grounding_processor = AutoProcessor.from_pretrained(dino_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(
        device
    )

    return mask_predictor, grounding_processor, grounding_model

def inference(cv_image, text_prompt):
    image = cv_image[:, :, ::-1]  # BGR to RGB
    image = image.copy()

    inputs = grounding_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
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

if __name__=='__main__':
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

    # ================== #
    # Load data
    # ================== #

    PATH_PREFIX = '/media/all/easystore/dataset/semantic_mapping/scannetv2/scene0000_00_data/'

    depth_dir = PATH_PREFIX + 'depth'
    odom_dir = PATH_PREFIX + 'pose'
    image_dir = PATH_PREFIX + 'color'
    prompt_path = PATH_PREFIX + 'prompt.yaml'
    
    VISUALIZE = True
    ANNOTATE = True

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
        ORIG_ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'orig_3d_in_loop')
        if os.path.exists(VERBOSE_ANNOTATE_OUT_DIR):
            os.system(f"rm -r {VERBOSE_ANNOTATE_OUT_DIR}")
        os.makedirs(VERBOSE_ANNOTATE_OUT_DIR, exist_ok=True)
        if os.path.exists(ORIG_ANNOTATE_OUT_DIR):
            os.system(f"rm -r {ORIG_ANNOTATE_OUT_DIR}")
        os.makedirs(ORIG_ANNOTATE_OUT_DIR, exist_ok=True)

    # ================== #
    # Initialize
    # ================== #

    cnt = 0
    valid_cnt = 1
    cloud_stack = []

    image_length = len(os.listdir(image_dir))
    rgb_intrinsics = scannet_utils.read_intrinsics(PATH_PREFIX + 'intrinsic/intrinsic_color.txt')
    depth_intrinsics = scannet_utils.read_intrinsics(PATH_PREFIX + 'intrinsic/intrinsic_depth.txt')

    rgb_shape = (968, 1296)
    depth_shape = (480, 640)
    
    # ================== #
    # Load prompt
    # ================== #
    text_prompt = []
    with open(prompt_path, 'r') as f:
        label_template = yaml.safe_load(f)
    for value in label_template.values():
        text_prompt += value
    text_prompt = " . ".join(text_prompt) + " ."
    print(f'text prompt: {text_prompt}')

    # ================== #
    # setup byte tracker
    # ================== #

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

    # ================== #
    # setup camera model
    # ================== #

    cloud_image_fusion = CloudImageFusion(platform='scannet')

    # ================== #
    # setup map
    # ================== #
    obj_mapper = ObjMapper(tracker=tracker, cloud_image_fusion=cloud_image_fusion, label_template=label_template, visualize=VISUALIZE)

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

        camera_odom = {
            "position": t_b2w,
            "orientation": Rotation.from_matrix(R_b2w).as_quat(),
        }
        
        # process mask
        image = cv2.imread(image_file)
        det_result = inference(image, text_prompt)
        confidences = det_result['confidences']
        confidences_mask = (confidences >= confidence_thres)
        det_confidences = confidences[confidences_mask]
        det_labels = det_result['labels'][confidences_mask]
        det_bboxes = det_result['bboxes'][confidences_mask]

        detections_tracked, _, _ = obj_mapper.track_objects(det_bboxes, det_labels, det_confidences, camera_odom)

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
                continue

        if ANNOTATE:
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

                cv2.imwrite(os.path.join(ANNOTATE_OUT_DIR, f"{cnt}.png"), image_anno)
                cv2.imwrite(os.path.join(VERBOSE_ANNOTATE_OUT_DIR, f"{cnt}.png"), image_verbose)

        # process depth
        depth_image = scannet_utils.read_depth(depth_file)
        cloud_world = scannet_utils.depth_to_pointcloud(depth_image, depth_intrinsics, SE3)

        obj_mapper.update_map(detections_tracked, stamp, camera_odom, cloud_world)

        if VISUALIZE:
            obj_mapper.rerun_vis(camera_odom, regularized=True, show_bbox=True, debug=False)
