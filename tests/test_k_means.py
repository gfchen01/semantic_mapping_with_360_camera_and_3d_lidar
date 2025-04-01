import scipy.spatial as spatial
import numpy as np
import rerun as rr
import open3d as o3d

from sklearn.decomposition import PCA

import math
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
def get_bbox_3d_oriented(points):
    bbox2d, _ = minimum_bounding_rectangle(points[:, :2])
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
    return center, extent, q

def minimum_bounding_rectangle(points):
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


if __name__=="__main__":
    cloud_file = "/home/luke/Downloads/door_97.bin"
    cloud = np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 3)
    # cloud_file = "/home/luke/Downloads/350_chair_242.bin"
    cloud = cloud[:, :3]

    size_heuristic = np.array([[1.0, 1.0, 1.0], [1.0, 0.2, 2.0]])

    for k in range(1, 10):
        # center, extent, q = get_bbox_3d_oriented(cloud)
        # point_transform = Rotation.from_quat(q).as_matrix()
        # cloud = (cloud - center) @ point_transform

        point_2d = cloud[:, :2]
        pca = PCA(n_components=2)
        pca.fit(point_2d)
        principle_direction = pca.components_[0]
        orientation = math.atan2(principle_direction[1], principle_direction[0])
        point_transform = Rotation.from_euler('z', orientation).as_matrix()
        centroid = np.mean(cloud[:, :3], axis=0)
        cloud = (cloud - centroid) @ point_transform

        diff = cloud - np.mean(cloud, axis=0)
        far_away_mask = (np.abs(diff) > size_heuristic[1]/2).any(axis=1)
        cloud = cloud[~far_away_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])

        center_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        center_point.translate(np.mean(cloud[:, :3], axis=0))

        direction_vis = o3d.geometry.TriangleMesh.create_arrow(0.1, 0.1, 0.1, 0.2)
        direction_vis.rotate(point_transform, center=(0, 0, 0))

        o3d.visualization.draw_geometries([pcd, center_point, direction_vis])

        if far_away_mask.sum() == 0:
            break
