from tensorflow_dataloader import TensorFlowDataLoader
import config as CONFIG
import open3d as o3d
import numpy as np


train_data_loader = TensorFlowDataLoader(
    name='train_dataloader',
    file_path=CONFIG.TRAINING_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)


labels, all_voxels = train_data_loader.load_data(CONFIG.TEST_DATA_PATH)
tensor = train_data_loader.create_padded_voxel_tensor(all_voxels[34][7])
print(train_data_loader.ids_to_label[labels[34]])

x, y, z = np.where(tensor == 1)
points = np.vstack((x, y, z)).T

# Convert to Open3D point cloud
pcd = o3d.t.geometry.PointCloud(points)

# Visualize
o3d.visualization.draw_geometries([pcd.to_legacy()])
