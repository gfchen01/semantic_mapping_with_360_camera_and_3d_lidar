import numpy as np
from scipy.spatial import cKDTree
import time

import open3d as o3d
from pathlib import Path

def find_nearby_points(A, B, max_distance):
    """
    Finds points in B that have a nearest neighbor in A within max_distance.

    Parameters:
        A (ndarray): (N, d) array of reference points.
        B (ndarray): (M, d) array of query points.
        max_distance (float): Maximum allowable distance to consider a nearest neighbor.

    Returns:
        ndarray: Indices of points in B that satisfy the condition.
    """
    # Build a k-d tree for A
    tree = cKDTree(A)

    # Query the nearest neighbor distance for each point in B
    distances, _ = tree.query(B, k=1)

    # Find indices of B where the nearest neighbor in A is within max_distance
    valid_indices = np.where(distances <= max_distance)[0]

    return valid_indices

def test_find_nearby_points():
    A = np.random.rand(500, 3)
    B = np.random.rand(500, 3)
    max_distance = 0.2

    start = time.time()
    valid_indices = find_nearby_points(A, B, max_distance)
    end = time.time()
    print("Time taken: ", end - start)

    # print(valid_indices)

def draw_two_point_clouds(A, B):
    object_points = o3d.geometry.PointCloud()
    object_points.points = o3d.utility.Vector3dVector(A)
    object_points.paint_uniform_color([1, 0, 0])
    target_points = o3d.geometry.PointCloud()
    target_points.points = o3d.utility.Vector3dVector(B)
    target_points.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_plotly([object_points, target_points], window_name="Object and Target")

def test_point_separation():
    data_dir = Path("output/separation_test/")

    i = 1

    for stamp in data_dir.iterdir():
        if stamp.is_dir():
            for pair in stamp.iterdir():
                if pair.is_dir():
                    A = np.load(pair / "voxel_object.npy")
                    B = np.load(pair / "voxel_target.npy")

                    if A.shape[0] > B.shape[0]:
                        A, B = B, A
                    draw_two_point_clouds(A, B)

                    dist_tests = [0.05, 0.1, 0.2]
                    for dist in dist_tests:
                        v_indices = find_nearby_points(A, B, dist)

                        A_temp = np.concatenate((A, B[v_indices]), axis=0)
                        B_temp = np.delete(B, v_indices, axis=0)
                        
                        draw_two_point_clouds(A_temp, B_temp)
                        
                        print(f"Distance: {dist}, Number of neighbors: {len(v_indices)}")

                    print(f"{i} {pair}: Weight percentatge: {A.shape[0]/B.shape[0]}")
                    
                    i += 1

                    input("Press Enter to continue...")

    # pair_name = Path("output/separation_test/1736999165481445888/chair_77_chair_58")
    # A = np.load(pair_name / "voxel_object.npy")
    # B = np.load(pair_name / "voxel_target.npy")
    # if A.shape[0] > B.shape[0]:
    #     A, B = B, A
    # draw_two_point_clouds(A, B)

    # dist_tests = [0.05, 0.1, 0.2]
    # for dist in dist_tests:
    #     v_indices = find_nearby_points(A, B, dist)

    #     A_temp = np.concatenate((A, B[v_indices]), axis=0)
    #     B_temp = np.delete(B, v_indices, axis=0)
        
    #     draw_two_point_clouds(A_temp, B_temp)
        
    #     print(f"Distance: {dist}, Number of neighbors: {len(v_indices)}")

    # print(f"{i} {pair_name}: Weight percentatge: {A.shape[0]/B.shape[0] if A.shape[0] < B.shape[0] else B.shape[0]/A.shape[0]}")

if __name__ == "__main__":
    # test_find_nearby_points()
    test_point_separation()
