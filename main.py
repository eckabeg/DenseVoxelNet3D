import numpy as np
from data_loader import BinaryDataLoader, PickleDataLoader, PlyDataLoader
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import config as CONFIG
from model_builder import ModelBuilder
from tensorflow_dataloader import TensorFlowDataLoader


def create_dense_voxel_tensor(voxels, voxel_size, bounding_box):
    # Create an empty 3D grid with the bounding box dimensions
    grid_shape = (
        int(np.ceil(bounding_box[0] / voxel_size)),
        int(np.ceil(bounding_box[1] / voxel_size)),
        int(np.ceil(bounding_box[2] / voxel_size)),
    )
    dense_grid = np.zeros(grid_shape, dtype=np.float32)

    for voxel in voxels:
        x, y, z = voxel.grid_index
        dense_grid[x, y, z] = 1  # Mark the voxel as occupied
    
    return dense_grid

def pad_or_trim_voxel_grid(voxel_grid, target_shape):
    padded_grid = np.zeros(target_shape, dtype=np.float32)

    # Find the slicing limits to center the voxel grid
    min_shape = np.minimum(voxel_grid.shape, target_shape)
    slices = tuple(slice(0, s) for s in min_shape)
    padded_slices = tuple(slice(0, s) for s in target_shape)

    padded_grid[padded_slices] = voxel_grid[slices]

    return padded_grid

def create_padded_voxel_tensor(voxels, bounding_box, target_shape):
    dense_voxel_grid = create_dense_voxel_tensor(voxels, CONFIG.VOXEL_SIZE, bounding_box)
    padded_voxel_grid = pad_or_trim_voxel_grid(dense_voxel_grid, target_shape)
    return tf.convert_to_tensor(padded_voxel_grid, dtype=tf.float32)

#display_folder("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C22_-__run_with_box_stageii")
#display_cloud("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C20_-__run_to_jump_to_walk_stageii/28.bin")

#dataLoader = BinaryDataLoader(0.05, 300)
#voxelized_data = dataLoader.load("C:/Users/lagro/source/repos/Uni/LidarSensorProject/data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C2_-_Run_to_stand_stageii")

def prepare_voxel_tensor(voxel_grids):
    tensor = tf.convert_to_tensor(voxel_grids, dtype=tf.float32)
    return tensor

def convert_voxels_to_dense_tensor(all_action_voxels, labels):
    input_tensor = []
    all_labels = []
    for actionIndex, action in enumerate(all_action_voxels):
        frame_grouping = []
        for i in range(0, len(action)):
            padded_voxel_tensor = create_padded_voxel_tensor(action[i], bounding_box, target_shape)
            all_labels.append(labels[actionIndex])
            if(CONFIG.FRAME_GROUPING <= 1):
                input_tensor.append(padded_voxel_tensor)
                continue

            frame_grouping.append(padded_voxel_tensor)
            for y in range(1, CONFIG.FRAME_GROUPING):
                frame_grouping.append(create_padded_voxel_tensor(action[i], bounding_box, target_shape))
            input_tensor.append(frame_grouping)
            frame_grouping = []
    
    return all_labels, input_tensor

print('Start create train_data_loader')
train_data_loader = TensorFlowDataLoader(
    file_paths=[CONFIG.TRAINING_DATA_PATH],
    bounding_box=(10, 10, 10),
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)
print('End create train_data_loader')

print('Start create train_dataset')
train_dataset = train_data_loader.get_tf_dataset()
print('End create train_dataset')

def preprocess_dataset(voxel, label):
    # Add a channel dimension to the voxel grid
    voxel = tf.expand_dims(voxel, axis=-1)  # Shape becomes (..., 1)
    return voxel, label

# Apply preprocessing
print('Start preprocess train_dataset')
#train_dataset = train_dataset.map(preprocess_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_prepared_dataset = train_dataset.shuffle(100).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print('End preprocess train_dataset')

def get_classes_from_files(file_paths, data_loader):
    unique_classes = set()
    for file_path in file_paths:
        labels, _ = data_loader.load_data(file_path)
        unique_classes.update(labels)
    return sorted(unique_classes), len(unique_classes)

# Example usage
print('Start unique_classes')
unique_classes, num_classes = get_classes_from_files([CONFIG.TRAINING_DATA_PATH], train_data_loader)
print('End unique_classes')
#input_tensor = tf.transpose(input_tensor, perm=(0, 2, 3, 4, 1))

model = ModelBuilder.AlexNet((num_classes), CONFIG.INPUT_SHAPE, CONFIG.FRAME_GROUPING)
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_prepared_dataset, epochs=CONFIG.EPOCHS)
model.save("models/direct_regression.keras")


test_data_loader = TensorFlowDataLoader(
    file_paths=[CONFIG.TEST_DATA_PATH],
    bounding_box=(10, 10, 10),
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

test_dataset = test_data_loader.get_tf_dataset()

# Apply preprocessing
test_prepared_dataset = test_dataset.shuffle(100).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_model = keras.models.load_model("models/direct_regression.keras")
predicted_labels = test_model(test_prepared_dataset, training=False)

# Assuming test_result is the output from your model
predicted_labels = tf.argmax(predicted_labels, axis=-1).numpy()

def extract_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label.numpy())  # Convert from Tensor to NumPy
    return labels

true_labels = extract_labels(test_prepared_dataset)

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print predicted labels
print("Predicted Labels:", predicted_labels)