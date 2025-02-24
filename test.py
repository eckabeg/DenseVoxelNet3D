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
print(train_data_loader.ids_to_label)
for index, label in enumerate(labels):
    if train_data_loader.ids_to_label[label] == 'close_an_unbrella':
        break


tensors = train_data_loader.normalize_and_create_padded_voxel_tensors(all_voxels[index])
print(train_data_loader.ids_to_label[labels[index]])
print(len(all_voxels[index]))
print(len(tensors))
pcds = []
for result in tensors:
    x, y, z = np.where(result == 1)
    points = np.vstack((x, y, z)).T

    # Convert to Open3D point cloud
    pcd = o3d.t.geometry.PointCloud(points)
    pcds.append(pcd.to_legacy())

# Visualize
o3d.visualization.draw_geometries(pcds)
