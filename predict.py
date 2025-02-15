import config as CONFIG
from tensorflow import keras
import tensorflow as tf
from tensorflow_dataloader import TensorFlowDataLoader
import numpy as np

model = keras.models.load_model("models/direct_regression.keras")

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
    voxels = np.reshape(voxels, (64, 64, 64, 2))
    voxels = np.expand_dims(voxels, axis=0)
    print(voxels.shape)
    probabilities = model(voxels).numpy().flatten()  # Convert to NumPy and flatten

    # Get top 5 class indices
    top_5_indices = np.argsort(probabilities)[::-1][:5]  # Sort in descending order

    # Print results
    print("-------------------")
    print(f"Ground Truth: {data_loader.ids_to_label[labels[0]]}")
    print("Top 5 Predictions:")
    for idx in top_5_indices:
        class_name = data_loader.ids_to_label[idx]
        confidence = probabilities[idx] * 100
        print(f"  {class_name}: {confidence:.2f}%")