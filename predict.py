import config as CONFIG
from tensorflow import keras
import tensorflow as tf
from tensorflow_dataloader import TensorFlowDataLoader
import numpy as np

model = keras.models.load_model("models/checkpoints/direct_regression.model.keras")

data_loader = TensorFlowDataLoader(
    name='Loader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

labels, all_voxels = data_loader.load_random_data(CONFIG.TEST_DATA_PATH, 1)
labels, all_voxels = data_loader.convert_voxels_to_dense_tensor(all_voxels, labels)
print(np.array(all_voxels).shape)

for i, voxels in enumerate(all_voxels):
    probabilities = model(voxels)
    print(labels[i], " Top 5 actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
        print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")