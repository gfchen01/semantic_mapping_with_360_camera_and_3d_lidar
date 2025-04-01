import numpy as np
import scipy.ndimage
from scipy.spatial.transform import Rotation

from line_profiler import profile

import cv2
import scipy

def scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA):
    lidarX = L2C_PARA["x"] #   lidarXStack[imageIDPointer]
    lidarY = L2C_PARA["y"] # idarYStack[imageIDPointer]
    lidarZ = L2C_PARA["z"] # lidarZStack[imageIDPointer]
    lidarRoll = -L2C_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = -L2C_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = -L2C_PARA["yaw"]# lidarYawStack[imageIDPointer]

    imageWidth = CAMERA_PARA["width"]
    imageHeight = CAMERA_PARA["height"]
    cameraOffsetZ = 0   #  additional pixel offset due to image cropping? 
    vertPixelOffset = 0  #  additional vertical pixel offset due to image cropping

    sinLidarRoll = np.sin(lidarRoll)
    cosLidarRoll = np.cos(lidarRoll)
    sinLidarPitch = np.sin(lidarPitch)
    cosLidarPitch = np.cos(lidarPitch)
    sinLidarYaw = np.sin(lidarYaw)
    cosLidarYaw = np.cos(lidarYaw)
    
    lidar_offset = np.array([lidarX, lidarY, lidarZ])
    camera_offset = np.array([0, 0, cameraOffsetZ])
    
    cloud = laserCloud[:, :3] - lidar_offset
    R_z = np.array([[cosLidarYaw, -sinLidarYaw, 0], [sinLidarYaw, cosLidarYaw, 0], [0, 0, 1]])
    R_y = np.array([[cosLidarPitch, 0, sinLidarPitch], [0, 1, 0], [-sinLidarPitch, 0, cosLidarPitch]])
    R_x = np.array([[1, 0, 0], [0, cosLidarRoll, -sinLidarRoll], [0, sinLidarRoll, cosLidarRoll]])
    cloud = cloud @ R_z @ R_y @ R_x
    cloud = cloud - camera_offset
    
    horiDis = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
    horiPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 1], cloud[:, 0]) + imageWidth / 2 + 1).astype(int) - 1
    vertPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 2], horiDis) + imageHeight / 2 + 1 + vertPixelOffset).astype(int)
    PixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    point_pixel_idx = np.array([horiPixelID, vertPixelID, PixelDepth]).T
    
    return point_pixel_idx.astype(int)

def scan2pixels_wheelchair(laserCloud):
    # project scan points to image pixels
    # https://github.com/jizhang-cmu/cmu_vla_challenge_unity/blob/noetic/src/semantic_scan_generation/src/semanticScanGeneration.cpp
    
    # Input: 
    # [#points, 3], x-y-z coordinates of lidar points
    
    # Output: 
    #    point_pixel_idx['horiPixelID'] : horizontal pixel index in the image coordinate
    #    point_pixel_idx['vertPixelID'] : vertical pixel index in the image coordinate

    # L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": 0.0, "pitch": 0, "yaw": -0.0} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    CAMERA_PARA= {"hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"hfov": 360, "vfov": 30}   
    
    return scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA)

def scan2pixels_mecanum_sim(laserCloud):
    CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.1, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_mecanum(laserCloud):
    CAMERA_PARA= {"x": -0.12, "y": -0.075, "z": 0.265, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)
    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_diablo(laserCloud):
    CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.185, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)
    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_scannet(cloud):
    rgb_intrinsics = {
        'fx': 1169.621094,
        'fy': 1167.105103,
        'cx': 646.295044,
        'cy': 489.927032,
    }

    rgb_width = 1296
    rgb_height = 968

    x = cloud[:, 0]
    y = cloud[:, 1]
    x_rgb = x * rgb_intrinsics['fx'] / (cloud[:, 2] + 1e-6) + rgb_intrinsics['cx']
    y_rgb = y * rgb_intrinsics['fy'] / (cloud[:, 2] + 1e-6) + rgb_intrinsics['cy']

    point_pixel_idx = np.array([y_rgb, x_rgb, cloud[:, 2]]).T
    return point_pixel_idx


# @profile
import jax
import jax.numpy as jnp

@jax.jit
def min_depth_per_pixel(coords, depths):
    """
    coords: (N, 2) int array -> each row is (x, y)
    depths: (N,) float array -> depth for each pixel
    
    Returns:
    unique_coords: (M, 2) array of unique pixel coords
    min_depths: (M,) array of minimum depth per unique coord
    """
    # 1) Get unique coordinates + inverse index
    coords = jnp.array(coords)
    depths = jnp.array(depths)

    # unique_coords, inv_idx = jnp.unique(coords, axis=0, return_inverse=True, size=coords.shape[0], fill_value=-1)
    
    # unique_coords = unique_coords[unique_coords[:, 0] != -1]  # Remove fill_value
    # assert len(inv_idx) == len(coords)

    # 2) Prepare output array for minimum depth
    min_depths = jnp.full(len(coords), jnp.inf, dtype=depths.dtype) # discard the last element at last
    
    # # 3) "Scatter" minimum using np.minimum.at
    # #    This will, for each index in inv_idx, do:
    # #       min_depths[inv_idx[i]] = min(min_depths[inv_idx[i]], depths[i])
    # for i in range(len(coords)):
    #     # min_depths[inv_idx[i]] = jnp.min(min_depths[inv_idx[i]], depths[i])
    #     check_depth = min_depths[inv_idx[i]]
    #     if depths[i] < check_depth:
    #         min_depths = jax.ops.index_update(min_depths, inv_idx[i], depths[i])
            
    #     # min_depths.at[inv_idx[i]].set(jnp.min(min_depths[inv_idx[i]], depths[i]))

    def body_fun(i, current_min_depths):
        # Get the current depth and the index corresponding to the coordinate.
        current_depth = depths[i]
        current_value = current_min_depths[i]
        # Use jax.lax.select to pick the minimum without a Python if.
        new_value = jax.lax.select(current_depth < current_value, current_depth, current_value)
        return current_min_depths.at[i].set(new_value)
    
    final_min_depths = jax.lax.fori_loop(0, len(coords), body_fun, min_depths)
    
    # return unique_coords, final_min_depths
    return coords, final_min_depths

class CloudImageFusion:
    def __init__(self, platform):
        self.platform_list = ['wheelchair', 'mecanum', 'mecanum_sim', 'scannet', 'diablo']

        if platform not in self.platform_list:
            raise ValueError(f"Invalid platform: {platform}. Available platforms: {self.platform_list}")
        else:
            self.platform = platform
            self.scan2pixels = eval(f"scan2pixels_{platform}")
        
        if platform == 'wheelchair':
            self.scan2pixels = scan2pixels_wheelchair
        elif platform == 'mecanum':
            self.scan2pixels = scan2pixels_mecanum
        elif platform == 'mecanum_sim':
            self.scan2pixels = scan2pixels_mecanum_sim
        elif platform == 'scannet':
            self.scan2pixels = scan2pixels_scannet
        elif platform == 'diablo':
            self.scan2pixels = scan2pixels_diablo
        else:
            print(f"Invalid platform: {platform}. Available platforms: [wheelchair, mecanum, mecanum_sim, scannet, diablo]")
            raise ValueError
    
    def generate_seg_cloud(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
        # Project the cloud points to image pixels
        point_pixel_idx = self.scan2pixels(cloud)

        if masks is None or len(masks) == 0:
            return None, None
        
        image_shape = masks[0].shape
        
        out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
                            (point_pixel_idx[:, 0] < image_shape[1]) & \
                            (point_pixel_idx[:, 1] >= 0) & \
                            (point_pixel_idx[:, 1] < image_shape[0])

        point_pixel_idx = point_pixel_idx[out_of_bound_filter]
        cloud = cloud[out_of_bound_filter]
        
        horDis = point_pixel_idx[:, 2]
        point_pixel_idx = point_pixel_idx.astype(int)

        all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
        obj_cloud_world_list = []
        for i in range(len(labels)):
            obj_mask = masks[i]
            cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
            all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
            obj_cloud = cloud[cloud_mask]
                    
            # obj_cloud_list.append(obj_cloud)
            
            obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
            obj_cloud_world_list.append(obj_cloud_world)

        if image_src is not None:
            all_obj_cloud = cloud
            all_obj_point_pixel_idx = point_pixel_idx
            horDis = horDis
            # all_obj_cloud = cloud[all_obj_cloud_mask]
            # all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
            # horDis = horDis[all_obj_cloud_mask]
            maxRange = 6.0
            pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
            image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([pixelVal, 255-pixelVal, np.zeros_like(pixelVal)]).T # assume RGB
        
        return obj_cloud_world_list

    # @profile
    def generate_seg_cloud_v2(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
        point_pixel_idx = self.scan2pixels(cloud)

        if masks is None:
            return None, None
        
        image_shape = masks[0].shape
        
        out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
                            (point_pixel_idx[:, 0] < image_shape[1]) & \
                            (point_pixel_idx[:, 1] >= 0) & \
                            (point_pixel_idx[:, 1] < image_shape[0])

        point_pixel_idx = point_pixel_idx[out_of_bound_filter]
        cloud = cloud[out_of_bound_filter]
        
        depths = point_pixel_idx[:, 2]
        point_pixel_idx = point_pixel_idx.astype(int)

        depth_image = np.full(image_shape, np.inf, dtype=np.float32)

        import time
        start_time = time.time()

        # pixel_indices, depths = min_depth_per_pixel(point_pixel_idx[:, :2], horDis)
        # pixel_indices = np.array(pixel_indices, dtype=int)
        # pixel_indices = pixel_indices[pixel_indices[:, 0] >= 0]
        # depths = np.array(depths)

        np.minimum.at(depth_image, (point_pixel_idx[:, 1], point_pixel_idx[:, 0]), depths)
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        inflated_depth_image = scipy.ndimage.grey_dilation(depth_image, footprint=structure, mode='nearest')

        inflated_depth_image = np.minimum(inflated_depth_image, depth_image)

        print(f'pixel conversion: {time.time() - start_time} for {point_pixel_idx.shape[0]} points')
        # for i, pixel_idx in enumerate(pixel_indices):
        #     depth_image[*pixel_idx[[1, 0]].tolist()] = depths[i]
            
        # depth_image[pixel_indices[:, 1], pixel_indices[:, 0]] = depths

        valid_mask = ~np.isinf(inflated_depth_image)  # Mask for valid depth values
        if valid_mask.any():
            min_depth = inflated_depth_image[valid_mask].min()
            max_depth = inflated_depth_image[valid_mask].max()

            print(f"Min depth: {min_depth}, Max depth: {max_depth}")

            # Normalize only valid depth values
            normalized_depth = np.zeros_like(inflated_depth_image, dtype=np.uint8)
            normalized_depth[valid_mask] = 255 * (1 - (inflated_depth_image[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6))
        else:
            normalized_depth = np.zeros_like(inflated_depth_image, dtype=np.uint8)  # If all values are inf, return a blank image
        
        # cv2.imshow("Depth Image", normalized_depth)
        # cv2.waitKey(1)  # Wait for a key press to close the window

        all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
        obj_cloud_world_list = []
        for i in range(len(labels)):
            obj_mask = masks[i]
            cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
            all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
            obj_cloud = cloud[cloud_mask]
                    
            # obj_cloud_list.append(obj_cloud)
            
            obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
            obj_cloud_world_list.append(obj_cloud_world)

        if image_src is not None:
            all_obj_cloud = cloud
            all_obj_point_pixel_idx = point_pixel_idx
            horDis = horDis
            # all_obj_cloud = cloud[all_obj_cloud_mask]
            # all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
            # horDis = horDis[all_obj_cloud_mask]
            maxRange = 6.0
            pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
            image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([pixelVal, 255-pixelVal, np.zeros_like(pixelVal)]).T # assume RGB
        
        return obj_cloud_world_list, normalized_depth
