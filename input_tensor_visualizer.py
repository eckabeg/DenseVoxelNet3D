from tensorflow_dataloader import TensorFlowDataLoader
import config as CONFIG
import open3d as o3d
import numpy as np

# This script enables us to visualize the final input tensor of our model. The input tensor is created from the TensorflowDataLoader.
# By doing so we are able to visualize and anlyse the input of our model.


actionToVisualize = 'close_an_unbrella'

# Setup the TensorFlowDataLoader
train_data_loader = TensorFlowDataLoader(
    name='train_dataloader',
    file_path=CONFIG.TRAINING_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)
train_data_loader.setup()
labels, all_voxels = train_data_loader.load_data(CONFIG.TEST_DATA_PATH)
print(train_data_loader.ids_to_label)

# This loop enables us to find the first action with the specifed label
for index, label in enumerate(labels):
    if train_data_loader.ids_to_label[label] == actionToVisualize:
        break

# Create the raw input tensor for the selected action
tensors = train_data_loader.normalize_and_create_padded_voxel_tensors(all_voxels[index])

# Convert the input tensor into an shape that can be visualized
pcds = []
for result in tensors:
    x, y, z = np.where(result == 1)
    points = np.vstack((x, y, z)).T

    pcd = o3d.t.geometry.PointCloud(points)
    pcds.append(pcd.to_legacy())

# visualize the input tensor
o3d.visualization.draw_geometries(pcds)
