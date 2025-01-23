import numpy as np
from data_loader import BinaryDataLoader, PlyDataLoader
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import config as CONFIG
from model_builder import ModelBuilder


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

def convert_voxels_to_dense_tensor(all_action_voxels):
    input_tensor = []
    for action in all_action_voxels:
        frame_grouping = []
        for i in range(0, len(action)):
            padded_voxel_tensor = create_padded_voxel_tensor(action[i], bounding_box, target_shape)
            if(CONFIG.FRAME_GROUPING <= 1):
                input_tensor.append(padded_voxel_tensor)
                continue

            frame_grouping.append(padded_voxel_tensor)
            for y in range(1, CONFIG.FRAME_GROUPING):
                frame_grouping.append(create_padded_voxel_tensor(action[i], bounding_box, target_shape))
            input_tensor.append(frame_grouping)
            frame_grouping = []
    
    return input_tensor

# Voxelize the point cloud
dataLoader = BinaryDataLoader(CONFIG.VOXEL_SIZE, 100)
all_action_voxels = [
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C2_-_Run_to_stand_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C3_-_Run_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C4_-_Run_to_walk1_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C5_-_walk_to_run_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C6_-_stand_to_run_backwards_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C7_-__run_backwards_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C8_-__run_backwards_to_stand_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C9_-__run_backwards_turn_run_forward_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C10_-__run_backwards_stop_run_forward_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C11_-__run_turn_left_(90)_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C13_-__run_turn_left_(135)_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C14_-__run_turn_right__(90)_stageii"),
]


# Define a bounding box (e.g., 10x10x10 units) and target shape
bounding_box = (10, 10, 10)
target_shape = CONFIG.INPUT_SHAPE

# Convert voxels to dense tensor and pad
input_tensor = convert_voxels_to_dense_tensor(all_action_voxels)
input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
#input_tensor = tf.transpose(input_tensor, perm=(0, 2, 3, 4, 1))
labels = np.concatenate([
    np.repeat(0, 18), np.repeat(1, 35),

    np.repeat(0, 16),

    np.repeat(0, 15), np.repeat(2, 20),

    np.repeat(2, 41), np.repeat(0, 13),

    np.repeat(1, 45), np.repeat(3, 29),

    np.repeat(3, 30),

    np.repeat(3, 32), np.repeat(1, 12),

    np.repeat(3, 6), np.repeat(4, 4), np.repeat(0, 12),

    np.repeat(3, 13), np.repeat(1, 37),

    np.repeat(0, 23),

    np.repeat(0, 28),

    np.repeat(0, 42)
])
labels = tf.convert_to_tensor(labels, dtype=tf.int32)
classes = ('running', 'standing', 'walking', 'running backwards', "turning")


"""
model = models.Sequential()
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(64, 64, 64, 1)))  # Input: 3D grid
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.Dropout(0.3))

# Flatten and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(classes), activation='softmax'))  # Output layer for classification
"""
model = ModelBuilder.AlexNet(len(classes), CONFIG.INPUT_SHAPE, CONFIG.FRAME_GROUPING)
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(input_tensor, labels, epochs=CONFIG.EPOCHS)
model.save("models/direct_regression.keras")


predict_action_voxels = [
    # already seen data
    #dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C2_-_Run_to_stand_stageii"),
    #dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C3_-_Run_stageii"),
    #dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C4_-_Run_to_walk1_stageii"),
    #dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C5_-_walk_to_run_stageii"),

    # new data
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C15_-__run_turn_right__(45)_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C16_-__run_turn_right__(135)_stageii"),
    dataLoader.load_voxels("data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C19_-__run_to_hop_to_walk_stageii"),
]
predict_tensor = convert_voxels_to_dense_tensor(predict_action_voxels)
predict_tensor = tf.convert_to_tensor(predict_tensor, dtype=tf.float32)
predict_tensor = np.expand_dims(predict_tensor, axis=-1)
test_model = keras.models.load_model("models/direct_regression.keras")
predicted_labels = test_model(predict_tensor, training=False)

# Assuming test_result is the output from your model
predicted_labels = tf.argmax(predicted_labels, axis=-1).numpy()
# If you have true labels for the test data
true_labels = np.concatenate([
    #np.repeat(0, 18), np.repeat(1, 35),

    #np.repeat(0, 16),

    #np.repeat(0, 15), np.repeat(2, 20),

    #np.repeat(2, 41), np.repeat(0, 13),


    np.repeat(0, 21),

    np.repeat(0, 23),

    np.repeat(0, 28), np.repeat(2, 14),
])

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print predicted labels
print("Predicted Labels:", predicted_labels)



def PointCloudClassifier():
    dataLoader = BinaryDataLoader(0.05, 600)
    pcds = dataLoader.load_pcd("C:/Users/lagro/source/repos/Uni/LidarSensorProject/data/LIPD/PC_Data/train/ACCAD/Female1Running_c3d/C2_-_Run_to_stand_stageii")
    input_tensor = tf.convert_to_tensor(pcds, dtype=tf.float32)
    labels = tf.convert_to_tensor(np.concatenate([np.repeat(0, 18), np.repeat(1, 35)]), dtype=tf.int32)

    model = models.Sequential()
    model.add(layers.Conv1D(64, 1, activation='relu', input_shape=(500, 3)))
    model.add(layers.Conv1D(128, 1, activation='relu'))
    model.add(layers.Conv1D(1024, 1, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(input_tensor, labels, epochs=10)