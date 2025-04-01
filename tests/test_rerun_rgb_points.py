import open3d as o3d
import numpy as np
import rerun as rr

def load_and_display_pcd(file_path):
    # Load the point cloud from the PCD file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Convert the Open3D point cloud to a numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 256
    colors = colors.astype(np.int32)
    np.concatenate([colors, np.ones((colors.shape[0], 1)) * 0.5], axis=1)
    
    # Display the point cloud using rerun
    rr.init("PointCloud Display", spawn=True)
    rr.log(
        "world/points",
        rr.Points3D(points, colors=colors, radii=0.005),
    )

if __name__ == "__main__":
    file_path = "/home/all/Downloads/cic_rgb_002.pcd"
    load_and_display_pcd(file_path)