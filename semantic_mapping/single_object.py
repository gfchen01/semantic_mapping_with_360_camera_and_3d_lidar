import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .utils import normalize_angles_to_pi, R_to_yaw, discretize_angles
import open3d as o3d

from line_profiler import profile

DIMENSION_PRIORS = {
'default': (5.0, 5.0, 2.0),

'table': (5.0, 3.0, 2.0), 
'chair': (1.5, 1.5, 2.0),
'sofa': (3.0, 3.0, 2.0),
'pottedplant': (1.0, 1.0, 1.0),
'fireextinguisher': (0.5, 0.5, 0.5),

# 'door': (1.0, 0.2, 3.0),
# 'case': (1.0, 1.0)
}

def percentile_index_search_binary(sorted_weights, percentile):
    total_weight = np.sum(sorted_weights)
    percentile_weight = total_weight * percentile
    current_weight = 0
    # left = 0
    # right = len(sorted_weights)
    # while left < right:
    #     mid = (left + right) // 2
    #     current_weight += sorted_weights[mid]
    #     if current_weight > percentile_weight:
    #         right = mid
    #     else:
    #         left = mid + 1

    i = 0
    while i < len(sorted_weights) and current_weight < percentile_weight:
        current_weight += sorted_weights[i]
        i += 1

    return i

def get_box_3d(points):
    min_xyz = points[:, :3].min(axis=0)
    max_xyz = points[:, :3].max(axis=0)
    center = (min_xyz + max_xyz) / 2
    extent = (max_xyz - min_xyz)
    q = [0.0, 0.0, 0.0, 1.0] # xyzw
    return center, extent, q

class VoteStatistics:
    def __init__(self, voxels: np.array, voxel_size: int, odom_R, odom_t, num_angle_bin=15):
        num_points = voxels.shape[0]
        
        voxel_to_odom = voxels - odom_t
        voxel_to_odom = voxel_to_odom @ odom_R # transform to body frame. Note: odom_R = odom_R.T.T
        obs_angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        obs_angles = normalize_angles_to_pi(obs_angles)
        obs_angles = discretize_angles(obs_angles, num_angle_bin)
        
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.tree = cKDTree(voxels)
        self.vote = np.ones(num_points)
        self.observation_angles = np.zeros([num_points, num_angle_bin])
        self.observation_angles[np.arange(num_points), obs_angles] = 1
        
        self.regularized_voxel_mask = np.zeros(num_points, dtype=bool)
        
        self.num_angle_bin = num_angle_bin
        
    def update(self, voxels, odom_R, odom_t):
        num_points = voxels.shape[0]
        
        voxel_to_odom = voxels - odom_t
        # voxel_to_odom = voxel_to_odom @ odom_R
        obs_angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        obs_angles = normalize_angles_to_pi(obs_angles)
        obs_angles = discretize_angles(obs_angles, self.num_angle_bin)
        
        distances, indices = self.tree.query(voxels) # indices: the index of the closest point in the original point cloud
        indices_to_merge = indices[distances < self.voxel_size]
        obs_angles_to_merge = obs_angles[distances < self.voxel_size]
        self.vote[indices_to_merge] += 1
        self.observation_angles[indices_to_merge, obs_angles_to_merge] = 1
        
        # # TODO: clear close but not voted points
        # dist_to_odom = np.linalg.norm(self.voxels - odom_t, axis=1)
        # no_update_mask = dist_to_odom > 1.5
        # updated_mask = np.zeros(self.voxels.shape[0], dtype=bool)
        # updated_mask[indices_to_merge] = True
        # valid_mask = no_update_mask | updated_mask
        # self.vote[~valid_mask] -= 2
        
        # process new voxels
        new_voxels = voxels[distances >= self.voxel_size]
        self.vote = np.concatenate([self.vote, np.ones(new_voxels.shape[0])])
        new_observation_angles = np.zeros([new_voxels.shape[0], self.num_angle_bin])
        new_observation_angles[np.arange(new_voxels.shape[0]), obs_angles[distances >= self.voxel_size]] = 1
        self.observation_angles = np.concatenate([self.observation_angles, new_observation_angles], axis=0)
        self.voxels = np.concatenate([self.voxels, new_voxels], axis=0)
        
        self.regularized_voxel_mask = np.concatenate([self.regularized_voxel_mask, np.zeros(new_voxels.shape[0], dtype=bool)])
        
        self.tree = cKDTree(self.voxels)
        
        assert self.observation_angles.shape[0] == self.vote.shape[0] == self.voxels.shape[0] == self.regularized_voxel_mask.shape[0]
    
    def update_through_vote_stat(self, vote_stat):
        voxels = vote_stat.voxels
        votes = vote_stat.vote
        obs_angles = vote_stat.observation_angles
        
        distances, indices = self.tree.query(voxels)
        indices_to_merge = indices[distances < self.voxel_size]
        obs_angles_to_merge = obs_angles[distances < self.voxel_size]
        self.vote[indices_to_merge] += votes[distances < self.voxel_size]
        self.observation_angles[indices_to_merge] = np.logical_or(self.observation_angles[indices_to_merge], obs_angles_to_merge)
        
        new_voxels = voxels[distances >= self.voxel_size]
        new_votes = votes[distances >= self.voxel_size]
        new_observation_angles = obs_angles[distances >= self.voxel_size]
        new_regularized_voxel_mask = vote_stat.regularized_voxel_mask[distances >= self.voxel_size]
        self.voxels = np.concatenate([self.voxels, new_voxels])
        self.vote = np.concatenate([self.vote, new_votes])
        self.observation_angles = np.concatenate([self.observation_angles, new_observation_angles])
        self.tree = cKDTree(self.voxels)

        self.regularized_voxel_mask = np.concatenate([self.regularized_voxel_mask, new_regularized_voxel_mask])
        
        assert self.observation_angles.shape[0] == self.vote.shape[0] == self.voxels.shape[0] == self.regularized_voxel_mask.shape[0]
    
    def update_through_mask(self, mask):
        self.voxels = self.voxels[mask]
        self.vote = self.vote[mask]
        self.observation_angles = self.observation_angles[mask]
        self.tree = cKDTree(self.voxels)
        
        assert self.observation_angles.shape[0] == self.vote.shape[0] == self.voxels.shape[0]
    
    def cal_distance(self, voxels):
        distances, indices = self.tree.query(voxels)
        return distances.mean()
        
    def reproject_filter(self, R_w2b, t_w2b, mask, projection_func):
        voxels = self.voxels
        voxels = voxels @ R_w2b.T + t_w2b
        voxels_on_image = projection_func(voxels).astype(int)
        voxels_mask = mask[voxels_on_image[:, 1], voxels_on_image[:, 0]].astype(bool)
        
        self.vote[~voxels_mask] -= 1

    def reproject_obs_angle(self, R_w2b, t_w2b, mask, projection_func, image_src=None):
        voxels = self.voxels
        voxels = voxels @ R_w2b.T + t_w2b
        voxels_on_image = projection_func(voxels).astype(int)

        if mask.size == 4: # bbox
            xmin, ymin, xmax, ymax = mask
            voxels_mask = (voxels_on_image[:, 0] >= xmin) & (voxels_on_image[:, 0] <= xmax) & (voxels_on_image[:, 1] >= ymin) & (voxels_on_image[:, 1] <= ymax)
        else:
            voxels_mask = mask[voxels_on_image[:, 1], voxels_on_image[:, 0]].astype(bool)

        # print(f"voxels on mask: {np.sum(voxels_mask)}")

        if np.sum(voxels_mask) == 0 and image_src is not None:
            image_src[voxels_on_image[:, 1], voxels_on_image[:, 0]] = [0, 0, 255]
        
        odom_t = -R_w2b.T @ t_w2b
        odom_R = R_w2b.T

        voxel_to_odom = voxels[voxels_mask] - odom_t
        # voxel_to_odom = voxel_to_odom @ odom_R
        obs_angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        obs_angles = normalize_angles_to_pi(obs_angles)
        obs_angles = discretize_angles(obs_angles, self.num_angle_bin)
        self.observation_angles[voxels_mask, obs_angles] = 1
    
    def retrieve_valid_voxel_indices(self, diversity_percentile=0.3, regularized=True):
        if regularized:
            voxels = self.voxels[self.regularized_voxel_mask]
            obs_angles = self.observation_angles[self.regularized_voxel_mask]
            votes = self.vote[self.regularized_voxel_mask]
        else:
            voxels = self.voxels
            obs_angles = self.observation_angles
            votes = self.vote
        
        # Newer version, add votes as weight
        if len(obs_angles) > 0:
            angle_diversity = np.sum(obs_angles, axis=1)
            sorted_indices = np.argsort(angle_diversity) # smaller to larger
            sorted_diversity = angle_diversity[sorted_indices]
            sorted_weights = self.vote[sorted_indices]
            total_weight = np.sum(sorted_weights)

            # search for the index that surpasses the percentile
            percentile_index = percentile_index_search_binary(sorted_diversity, 1 - diversity_percentile)
            voxel_indices = sorted_indices[percentile_index:]
        else:
            voxel_indices = np.empty(0, dtype=int)
        
        return voxel_indices
    
    def retrieve_valid_voxels_by_clustering(self, clustering_labels, diversity_percentile=0.3):
        voxels = self.voxels
        obs_angles = self.observation_angles

        assert len(clustering_labels) == len(voxels)

        added_cluster_labels = set()
        voxel_indices = []
        if len(obs_angles) > 0:
            angle_diversity = np.sum(obs_angles, axis=1)
            sorted_indices = np.argsort(angle_diversity) # smaller to larger
            sorted_diversity = angle_diversity[sorted_indices]
            sorted_weights = self.vote[sorted_indices]
            total_weight = np.sum(sorted_weights)

            current_weight = 0
            # search for the index that surpasses the percentile
            for ind in reversed(sorted_indices):
                if clustering_labels[ind] in added_cluster_labels:
                    continue
                else:
                    added_cluster_labels.add(clustering_labels[ind])
                    selected_cluster_mask = (clustering_labels == clustering_labels[ind])
                    cluster_weight = np.sum(angle_diversity[selected_cluster_mask])
                    current_weight += cluster_weight
                    voxel_indices.append(np.where(selected_cluster_mask != 0)[0])
                    if current_weight > diversity_percentile * total_weight:
                        break
            voxel_indices = np.concatenate(voxel_indices, axis=0)
        else:
            voxel_indices = np.empty(0, dtype=int)

        return voxel_indices

class SingleObject:
    def __init__(self, class_id, obj_id, voxels, voxel_size, odom_R, odom_t, mask, stamp, num_angle_bin=15):
        self.class_id = {class_id: 1}
        self.obj_id = [obj_id]
        self.vote_stat = VoteStatistics(voxels, voxel_size, odom_R, odom_t, num_angle_bin)
        
        self.life = 0
        self.inactive_frame = -1
        
        self.key_frames = [mask]
        self.key_pose = odom_t + [R_to_yaw(odom_R)]
        
        self.latest_stamp = stamp
        self.info_frames_cnt = 1
        self.is_active = True

        self.valid_indices = None
        self.valid_indices_regularized = None

        self.valid_indices_by_clustering = None

        self.clustering_labels = None

        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True

        self.merged_obj_ids = None
    
    def add_key_frame(self, mask, odom_R, odom_t):
        self.key_frames.append(mask)
        self.key_pose.append(odom_t + [R_to_yaw(odom_R)])
    
    def merge(self, voxels, odom_R, odom_t, label, stamp):
        self.vote_stat.update(voxels, odom_R, odom_t)
        self.info_frames_cnt += 1
        self.latest_stamp = stamp

        self.class_id[label] = self.class_id.get(label, 0) + 1
        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True
    
    def merge_object(self, single_obj):
        self.obj_id.extend(single_obj.obj_id)
        self.vote_stat.update_through_vote_stat(single_obj.vote_stat)
        self.life = max(self.life, single_obj.life)
        self.info_frames_cnt += single_obj.info_frames_cnt
        self.latest_stamp = max(self.latest_stamp, single_obj.latest_stamp)

        for key in single_obj.class_id:
            self.class_id[key] = self.class_id.get(key, 0) + single_obj.class_id[key]
        
        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True

        self.merged_obj_ids = single_obj.obj_id

    def reproject_filter(self, R_w2b, t_w2b, mask):
        # convert bounding box to mask
        if mask.size == 4:
            bbox = mask.astype(int)
            mask = np.zeros([640, 1920])
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            
        self.vote_stat.reproject_filter(R_w2b, t_w2b, mask)

    def reproject_obs_angle(self, R_w2b, t_w2b, mask, projection_func, image_src=None):
        # if mask.size == 4:
        #     bbox = mask.astype(int)
        #     mask = np.zeros([640, 1920])
        #     mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        self.vote_stat.reproject_obs_angle(R_w2b, t_w2b, mask, projection_func, image_src)
        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True

    def get_dominant_label(self):
        max_label = max(self.class_id, key=self.class_id.get)
        return max_label
    
    def dbscan_cluster_params(self):
        if self.info_frames_cnt < 3 and self.inactive_frame < 5:
            min_points = 5
        else:
            min_points = 20
        return self.vote_stat.voxel_size * 2.0, min_points
    
    def cal_clusters(self):
        if self.req_clustering:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.vote_stat.voxels)
            eps, min_points = self.dbscan_cluster_params()
            self.clustering_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))        

    def regularize_shape(self, percentile=None):
        """
            Apply a regularization to the object shape
        """
        if self.req_shape_regularization:
            self.cal_clusters()
            clustering_labels = self.clustering_labels
            unique_labels = np.unique(clustering_labels)
            dim_prior = DIMENSION_PRIORS.get(self.get_dominant_label(), DIMENSION_PRIORS['default'])

            # if dim_prior[0] == DIMENSION_PRIORS['default'][0] and dim_prior[1] == DIMENSION_PRIORS['default'][1]:
            #     print(f"Warning: No dimension prior for class {self.get_dominant_label()}, {self.obj_id}")

            cluster_masks = []
            clustering_weight = []
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_masks.append(clustering_labels == label)
                cluster_point_mask = (clustering_labels == label)
                cluster_obs_angles = self.vote_stat.observation_angles[cluster_point_mask]
                cluster_weight = np.sum(cluster_obs_angles)
                clustering_weight.append(cluster_weight)
            
            clustering_weight_index = np.argsort(clustering_weight)
            valid_cluster_mask = np.zeros(self.vote_stat.voxels.shape[0], dtype=bool)

            current_weight = 0
            current_attempt_weight = 0
            total_weight = np.sum(clustering_weight)
            
            for weight_index in reversed(clustering_weight_index):
                mask = cluster_masks[weight_index]

                # remove very small clusters. Useful when the camera odom isn't very accurate
                if clustering_weight[weight_index] < 10:
                    continue

                cluster_mask_attempt = np.logical_or(valid_cluster_mask, mask)
                center, dim, q = get_bbox_3d_oriented(self.vote_stat.voxels[cluster_mask_attempt])

                if center is None:
                    continue

                if dim[0] > dim_prior[0] or dim[1] > dim_prior[1] or dim[2] > dim_prior[2]:
                    continue
                valid_cluster_mask = np.logical_or(valid_cluster_mask, mask)

                if percentile is not None:
                    assert percentile <= 1 and percentile > 0
                    current_weight += clustering_weight[weight_index]
                    if current_weight > percentile * total_weight:
                        break

            
            # self.vote_stat.update_through_voxel_mask(self.vote_stat.voxels, valid_cluster_mask)
            # self.vote_stat.regularized_voxel_mask = np.ones(self.vote_stat.voxels.shape[0], dtype=bool)
            self.vote_stat.regularized_voxel_mask = valid_cluster_mask
            self.req_recompute_indices = True
            self.req_shape_regularization = False
    
    def cal_distance(self, voxels):
        return self.vote_stat.cal_distance(voxels)
    
    def pop(self, mask):
        """
        Pop the info that is not in the mask
        """
        voxels_pop = self.vote_stat.voxels[~mask]
        obs_angles_pop = self.vote_stat.observation_angles[~mask]
        votes_pop = self.vote_stat.vote[~mask]
        self.vote_stat.update_through_mask(mask)

        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True

        return voxels_pop, obs_angles_pop, votes_pop
    
    def add(self, voxels, obs_angles, votes):
        self.vote_stat.voxels = np.concatenate([self.vote_stat.voxels, voxels])
        self.vote_stat.observation_angles = np.concatenate([self.vote_stat.observation_angles, obs_angles])
        self.vote_stat.vote = np.concatenate([self.vote_stat.vote, votes])
        
        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True

    def compute_valid_indices(self, diversity_percentile):
        if self.req_recompute_indices:
            self.valid_indices = self.vote_stat.retrieve_valid_voxel_indices(diversity_percentile=diversity_percentile, regularized=False)
            if self.req_shape_regularization:
                self.regularize_shape(percentile=diversity_percentile)
                self.req_shape_regularization = False
            self.valid_indices_regularized = self.vote_stat.retrieve_valid_voxel_indices(diversity_percentile=diversity_percentile, regularized=True)
            
            if self.req_clustering:
                self.cal_clusters()
                self.req_clustering = False
            self.valid_indices_by_clustering = self.vote_stat.retrieve_valid_voxels_by_clustering(self.clustering_labels, diversity_percentile=diversity_percentile)

            self.req_recompute_indices = False

    def retrieve_valid_voxels(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)

        if self.obj_id[0] < 0: # background points aren't regularized
            regularized = False
        if regularized:
            return self.vote_stat.voxels[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
        else:
            return self.vote_stat.voxels[self.valid_indices]

    def retrieve_valid_voxels_with_weights(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)

        if self.obj_id[0] < 0:
            regularized = False
        if regularized:
            valid_voxels = self.vote_stat.voxels[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
            valid_obs_angles = self.vote_stat.observation_angles[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
            valid_votes = self.vote_stat.vote[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
        else:
            valid_voxels = self.vote_stat.voxels[self.valid_indices]
            valid_obs_angles = self.vote_stat.observation_angles[self.valid_indices]
            valid_votes = self.vote_stat.vote[self.valid_indices]
        return valid_voxels, valid_obs_angles.sum(axis=1) * valid_votes / 5
    
    def retrieve_valid_voxels_clustered(self, diversity_percentile):
        self.compute_valid_indices(diversity_percentile)
        return self.vote_stat.voxels[self.valid_indices_by_clustering]

    @profile
    def infer_centroid(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)

        voxels = self.vote_stat.voxels
        obs_angles = self.vote_stat.observation_angles
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask]
            obs_angles = obs_angles[self.vote_stat.regularized_voxel_mask]
            valid_voxels = voxels[self.valid_indices_regularized]
            valid_obs_angles = obs_angles[self.valid_indices_regularized]
        else:
            valid_voxels = voxels[self.valid_indices]
            valid_obs_angles = obs_angles[self.valid_indices]
        weights = np.sum(valid_obs_angles, axis=1)
        if np.sum(weights) == 0:
            center = None
        else:
            center = np.average(valid_voxels, axis=0, weights=weights)
        return center
    
    def infer_bbox(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)

        voxels = self.vote_stat.voxels
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask]
            valid_voxels = voxels[self.valid_indices_regularized]
        else:
            valid_voxels = voxels[self.valid_indices]
        if len(valid_voxels) == 0:
            return None
        else:
            return get_box_3d(valid_voxels)

    def infer_bbox_oriented(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)

        voxels = self.vote_stat.voxels
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask]
            valid_voxels = voxels[self.valid_indices_regularized]
        else:
            valid_voxels = voxels[self.valid_indices]
        if len(valid_voxels) == 0:
            return None
        else:
            return get_bbox_3d_oriented(valid_voxels)
        
    def get_info_str(self):
        info_str = f"{self.obj_id[0]}, class: {self.get_dominant_label()}, all voxels: {len(self.valid_indices)}, regularized voxels: {len(self.valid_indices_regularized)}"
        return info_str

import math
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
def get_bbox_3d_oriented(points):
    bbox2d, _ = minimum_bounding_rectangle(points[:, :2])
    if bbox2d is not None:
        center2d = np.mean(bbox2d, axis=0)
        edge1 = bbox2d[1] - bbox2d[0]
        edge2 = bbox2d[2] - bbox2d[1]
        edge1_length = np.linalg.norm(edge1)
        edge2_length = np.linalg.norm(edge2)
        longest_edge = edge1 if edge1_length > edge2_length else edge2
        orientation = math.atan2(longest_edge[1], longest_edge[0])
        q = Rotation.from_euler('z', orientation).as_quat()
        extent = np.array([edge1_length, edge2_length, points[:, 2].max() - points[:, 2].min()])
        z_center = points[:, 2].max() - extent[2] / 2
        center = np.array([center2d[0], center2d[1], z_center])
    else:
        center = None
        extent = None
        q = None
    return center, extent, q

def minimum_bounding_rectangle(points):
    try:
        # Compute the convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # Initialize variables
        min_area = float('inf')
        best_rectangle = None
        
        # Rotate calipers
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge = p2 - p1
            
            # Normalize edge vector
            edge_vector = edge / np.linalg.norm(edge)
            perpendicular_vector = np.array([-edge_vector[1], edge_vector[0]])
            
            # Project all points onto the edge and perpendicular vector
            projections_on_edge = points @ edge_vector
            projections_on_perpendicular = points @ perpendicular_vector
            
            # Find bounds
            min_proj_edge = projections_on_edge.min()
            max_proj_edge = projections_on_edge.max()
            min_proj_perp = projections_on_perpendicular.min()
            max_proj_perp = projections_on_perpendicular.max()
            
            # Compute width, height, and area
            width = max_proj_edge - min_proj_edge
            height = max_proj_perp - min_proj_perp
            area = width * height
            
            if area < min_area:
                min_area = area
                best_rectangle = (min_proj_edge, max_proj_edge, min_proj_perp, max_proj_perp, edge_vector, perpendicular_vector)
        
        # Recover rectangle corners
        min_proj_edge, max_proj_edge, min_proj_perp, max_proj_perp, edge_vector, perpendicular_vector = best_rectangle
        corner1 = min_proj_edge * edge_vector + min_proj_perp * perpendicular_vector
        corner2 = max_proj_edge * edge_vector + min_proj_perp * perpendicular_vector
        corner3 = max_proj_edge * edge_vector + max_proj_perp * perpendicular_vector
        corner4 = min_proj_edge * edge_vector + max_proj_perp * perpendicular_vector
        return np.array([corner1, corner2, corner3, corner4]), min_area
    except Exception as e:
        return None, None
    


class AdjacencyGraph:
    """
    Vertex: int
    """
    def __init__(self):
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        # Add a vertex if it doesn't already exist
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
        #     print(f"Vertex '{vertex}' added.")
        # else:
        #     print(f"Vertex '{vertex}' already exists.")

    def add_edge(self, vertex1, vertex2):
        # Add an edge between two vertices
        if vertex1 not in self.adjacency_list:
            self.add_vertex(vertex1)
        if vertex2 not in self.adjacency_list:
            self.add_vertex(vertex2)
        self.adjacency_list[vertex1].append(vertex2) if vertex2 not in self.adjacency_list[vertex1] else None
        self.adjacency_list[vertex2].append(vertex1) if vertex1 not in self.adjacency_list[vertex2] else None

    def is_adjacent(self, vertex1, vertex2):
        if vertex1 not in self.adjacency_list or vertex2 not in self.adjacency_list:
            raise ValueError
        else:
            return (vertex2 in self.adjacency_list[vertex1])
    
    def is_set_adjacent(self, v_set1: list | np.ndarray, v_set2: list | np.ndarray):
        for id1 in v_set1:
            for id2 in v_set2:
                assert id1 != id2
                if self.is_adjacent(id1, id2):
                    return True
        
        return False

    def print_info(self):
        for vertex in self.adjacency_list:
            print(f'{vertex}: {self.adjacency_list[vertex]}')