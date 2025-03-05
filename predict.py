import config as CONFIG
from tensorflow import keras
import tensorflow as tf
from tensorflow_dataloader import TensorFlowDataLoader
import numpy as np

model = keras.models.load_model("models/checkpoints/direct_regression_newnet-v3-small.model.keras")

valid_data_loader = TensorFlowDataLoader(
    name='valid_dataloader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

valid_data_loader.setup()

vali_dataset = (
    tf.data.TFRecordDataset(valid_data_loader.TFRecord_file_paths)
    .map(valid_data_loader.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .flat_map(lambda x: x)
    .shuffle(15000)
    .batch(CONFIG.BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

index = 0
for batch_voxels, batch_labels in vali_dataset:
    for voxels, label in zip(batch_voxels, batch_labels):
        voxels = np.expand_dims(voxels, axis=0)
        probabilities = model(voxels).numpy().flatten()  # Convert to NumPy and flatten

        # Get the top 5 class indices for this sample
        top_5_indices = np.argsort(probabilities)[::-1][:5]
        
        # Print results
        print("-------------------")
        print(f"Ground Truth: {valid_data_loader.ids_to_label[label.numpy()]}")
        print("Top 5 Predictions:")
        for idx in top_5_indices:
            class_name = valid_data_loader.ids_to_label[idx]
            confidence = probabilities[idx] * 100
            print(f"  {class_name}: {confidence:.2f}%")
        index += 1
    if(index > 3):
        break
