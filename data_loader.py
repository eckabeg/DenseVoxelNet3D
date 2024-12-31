from abc import abstractmethod
import open3d as o3d
import numpy as np
import os

class DataLoader:
    @abstractmethod
    def voxelize(self, pcd):
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path):
        raise NotImplementedError

class PlyDataLoader(DataLoader):
    voxel_size = 0.05
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def voxelize(self, data):
        return o3d.geometry.VoxelGrid.create_from_point_cloud(data, voxel_size = self.voxel_size)
    
    def load(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pcd.estimate_normals()
        return pcd
    
class BinaryDataLoader(DataLoader):
    def __init__(self, voxel_size, voxel_amount):
        self.voxel_size = voxel_size
        self.voxel_amount = voxel_amount

    def voxelize(self, data):
        bounding_box = data.get_axis_aligned_bounding_box()
        bounding_box_size = bounding_box.get_extent()
        voxel_size = (np.prod(bounding_box_size) / self.voxel_amount) ** (1/3)
        return o3d.geometry.VoxelGrid.create_from_point_cloud(data, voxel_size = voxel_size)

    def load(self, directoryPath):
        fie_paths = [os.path.join(dp, filename) for dp, dn, filenames in os.walk(directoryPath) for filename in filenames if os.path.splitext(filename)[1] == '.bin']
        all_voxels = []
        for path in fie_paths:
            raw_point_cloud_data = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
            pcd = o3d.t.geometry.PointCloud(raw_point_cloud_data).to_legacy()
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = self.voxel_size)
            voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            if(len(voxels) < self.voxel_amount):
                padding = np.zeros((self.voxel_amount - len(voxels), 3))
                voxels = np.vstack([voxels, padding])
            all_voxels.append(voxels)
        return np.array(all_voxels)
    
    def load_pcd(self, directoryPath):
        fie_paths = [os.path.join(dp, filename) for dp, dn, filenames in os.walk(directoryPath) for filename in filenames if os.path.splitext(filename)[1] == '.bin']
        pcds = []
        for path in fie_paths:
            raw_point_cloud_data = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
            pcd = o3d.t.geometry.PointCloud(raw_point_cloud_data).to_legacy()
            points = np.asarray(pcd.points)
            if(len(points) < self.voxel_amount):
                padding = np.zeros((self.voxel_amount - len(points), 3))
                points = np.vstack([points, padding])
            pcds.append(points)
        return pcds
    
    def load_voxels(self, directoryPath):
        fie_paths = [os.path.join(dp, filename) for dp, dn, filenames in os.walk(directoryPath) for filename in filenames if os.path.splitext(filename)[1] == '.bin']
        all_voxels = []
        for path in fie_paths:
            raw_point_cloud_data = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
            pcd = o3d.t.geometry.PointCloud(raw_point_cloud_data).to_legacy()
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = self.voxel_size)
            all_voxels.append(voxel_grid.get_voxels())
        return all_voxels