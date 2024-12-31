import os
import open3d as o3d
import numpy as np

def display_folder(path):
    point_clouds = []
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        abs_filename = os.path.join(os.fsdecode(directory), os.fsdecode(file))
        raw_point_cloud_data = np.fromfile(abs_filename, dtype=np.float32).reshape(-1, 3)
        pcd = o3d.t.geometry.PointCloud(raw_point_cloud_data)
        pcd.paint_uniform_color(np.random.rand(3))
        point_clouds.append(pcd.to_legacy())
    o3d.visualization.draw_geometries(point_clouds)

def display_cloud(path):   
    raw_point_cloud_data = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
    pcd = o3d.t.geometry.PointCloud(raw_point_cloud_data)
    o3d.visualization.draw_geometries([pcd.to_legacy()])