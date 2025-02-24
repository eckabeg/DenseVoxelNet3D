import tensorflow as tf
import numpy as np
import open3d as o3d
import pickle
import os
import config as CONFIG
import random

class TensorFlowDataLoader:
    def __init__(self, name, file_path, bounding_box, target_shape, voxel_size, frame_grouping=1):
        self.name = name
        self.file_path = file_path
        self.bounding_box = bounding_box
        self.target_shape = target_shape
        self.voxel_size = voxel_size
        self.frame_grouping = frame_grouping
        self.all_labels = []
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.TFRecord_file_paths = []


    def voxelize(self, pcd):
        pcd = o3d.t.geometry.PointCloud(pcd).to_legacy()
        voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        return voxels.get_voxels()

    def load_data(self, file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        #data = data[:1]  # Limit to the first sequence
        labels = [seq["action"] for seq in data]
        self.labels_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.ids_to_label = {idx: label for label, idx in self.labels_to_id.items()}

        labels = [self.labels_to_id[label] for label in labels]
        self.all_labels = labels
        point_cloud_sequences = [seq["human_pc"] for seq in data]

        return labels, point_cloud_sequences
    
    def load_random_data(self, file_path, action_amount):
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        labels = [seq["action"] for seq in data]
        self.labels_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.ids_to_label = {idx: label for label, idx in self.labels_to_id.items()}
        data = random.choices(data, k=action_amount)
        labels = [seq["action"] for seq in data]
        labels = [self.labels_to_id[label] for label in labels]
        self.all_labels = labels
        point_cloud_sequences = [seq["human_pc"] for seq in data]

        all_voxels = []
        for point_cloud_sequence in point_cloud_sequences:
            sequence_voxels = []
            for raw_point_cloud in point_cloud_sequence:
                voxels = self.voxelize(raw_point_cloud)
                sequence_voxels.append(voxels)
            all_voxels.append(sequence_voxels)

        return labels, all_voxels

    def create_dense_voxel_tensor(self, voxels, voxel_size, bounding_box):
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

    def pad_or_trim_voxel_grid(self, voxel_grid, target_shape):
        padded_grid = np.zeros(target_shape, dtype=np.float32)

        # Find the slicing limits to center the voxel grid
        min_shape = np.minimum(voxel_grid.shape, target_shape)
        slices = tuple(slice(0, s) for s in min_shape)
        padded_slices = tuple(slice(0, s) for s in target_shape)

        padded_grid[padded_slices] = voxel_grid[slices]

        return padded_grid
    
    def normalize_and_create_padded_voxel_tensors(self, action):
        num_frames = len(action)
        action_input_tensor = np.zeros((num_frames, *self.target_shape), dtype=np.uint8)

        # Compute global min and max across all frames
        all_points = np.vstack(action)  # Stack all frames together
        global_min = np.min(all_points, axis=0)
        global_max = np.max(all_points, axis=0)

        # Compute global scale factor
        scale_factors = np.array(self.target_shape) / (global_max - global_min)
        scale = np.min(scale_factors)  # Uniform scaling

        for frame_idx, frame in enumerate(action):
            # Translate all frames using the same global min
            translated = frame - global_min

            # Apply global scale
            normalized = translated * scale

            # Round to integer voxel indices
            voxel_indices = np.floor(normalized).astype(int)
            voxel_indices = np.clip(voxel_indices, 0, np.array(self.target_shape) - 1)

            # Store in 4D voxel grid (x, y, z, t)
            for v in voxel_indices:
                action_input_tensor[frame_idx, v[0], v[1], v[2]] = 1  # Preserve all frames
        return tf.convert_to_tensor(action_input_tensor, dtype=tf.float32)

    def create_padded_voxel_tensor(self, voxels):
        dense_voxel_grid = self.create_dense_voxel_tensor(voxels, self.voxel_size, self.bounding_box)
        padded_voxel_grid = self.pad_or_trim_voxel_grid(dense_voxel_grid, self.target_shape)
        return tf.convert_to_tensor(padded_voxel_grid, dtype=tf.float32)
    
    def convert_voxels_to_dense_tensor(self, all_action_voxels, labels):
        input_tensor = []
        all_labels = []
        for actionIndex, action in enumerate(all_action_voxels):
            all_labels.append(labels[actionIndex])
            action_input_tensor = self.normalize_and_create_padded_voxel_tensors(action)
            input_tensor.append(action_input_tensor)
        
        return all_labels, input_tensor
    
    def setup(self):
        os.makedirs(os.path.dirname(CONFIG.INPUT_TENSOR_PATH), exist_ok=True)
        labels, all_voxels = self.load_data(self.file_path)
        chunk_size = max(1, len(all_voxels) // CONFIG.INPUT_TENSOR_CHUNK_SIZE)
        for index, start in enumerate(range(0, len(all_voxels), chunk_size)):
            file_path = f'{CONFIG.INPUT_TENSOR_PATH}{index}_{chunk_size}_{self.name}_{self.target_shape}_{self.bounding_box}_{self.voxel_size}.tfrecord'
            self.TFRecord_file_paths.append(file_path)
            if os.path.isfile(file_path):
                continue
            end = start + chunk_size
            chunk_voxels = all_voxels[start:end]
            chunk_labels = labels[start:end]
            
            print("start create input tensor")
            chunk_input_labels, chunk_input_voxels = self.convert_voxels_to_dense_tensor(chunk_voxels, chunk_labels)
            print("end create input tensor")
            
            self.save_to_tfrecord(file_path, chunk_input_labels, chunk_input_voxels)

    def save_to_tfrecord(self, file_path, labels, voxels):
        #Saves voxel and label data into a TFRecords file.
        with tf.io.TFRecordWriter(file_path) as writer:
            for label, voxel in zip(labels, voxels):
                voxel = tf.ensure_shape(voxel, [len(voxel), *self.target_shape])
                example = self.serialize_example(label, voxel)
                writer.write(example.SerializeToString())

    def serialize_example(self, label, voxel):
        #Converts a single (label, voxel) pair into a tf.train.Example.
        feature = {
            "voxel": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(voxel).numpy()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "actionCount": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(voxel)])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def parse_tfrecord(self, example_proto):
        feature_description = {
            "voxel": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "actionCount": tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        label = parsed_features["label"]
        actionCount = tf.cast(parsed_features["actionCount"], tf.int32)

        voxels = tf.io.parse_tensor(parsed_features["voxel"], out_type=tf.float32)
        voxels = tf.reshape(voxels, tf.concat([[actionCount], CONFIG.INPUT_SHAPE], axis=0))
        voxels = tf.ensure_shape(voxels, [None, *self.target_shape])
        grouped_voxels, grouped_labels = self.group_voxels(voxels, label)
        return tf.data.Dataset.from_tensor_slices((grouped_voxels, grouped_labels))
    
    def group_voxels(self, voxels, label):
        grouping = CONFIG.FRAME_GROUPING  # Expected to be 2 based on model input
        num_frames = tf.shape(voxels)[0]

        # Adjust range to prevent out-of-bounds slicing
        num_groups = num_frames - grouping + 1
        indices = tf.range(num_groups)

        def slice_voxels(i):
            return tf.reshape(voxels[i : i + grouping], (64, 64, 64, grouping))  # Shape: (grouping, 64, 64, 64)

        grouped_voxels = tf.map_fn(slice_voxels, indices, dtype=tf.float32)  # Shape: (num_groups, grouping, 64, 64, 64)

        # Adjust labels to match the number of grouped sequences
        grouped_labels = tf.broadcast_to(label, [tf.shape(grouped_voxels)[0]])

        return grouped_voxels, grouped_labels


    def generator(self):
        for index in range(0, CONFIG.INPUT_TENSOR_CHUNK_SIZE):
            file_path = f'{CONFIG.INPUT_TENSOR_PATH}{index}_{self.name}_{self.frame_grouping}_{self.target_shape}_{self.bounding_box}_{self.voxel_size}.p'
            with open(file_path, "rb") as file:
                (chunk_input_labels, chunk_input_voxels) = pickle.load(file)
            
            for index, voxels in enumerate(chunk_input_voxels):
                yield voxels, chunk_input_labels[index]

    def get_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(self.frame_grouping, *self.target_shape) if self.frame_grouping > 1 else self.target_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
