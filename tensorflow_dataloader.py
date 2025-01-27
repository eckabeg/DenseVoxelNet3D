import tensorflow as tf
import numpy as np
import open3d as o3d
import pickle
import os
import config as CONFIG

class TensorFlowDataLoader:
    def __init__(self, name, file_path, bounding_box, target_shape, voxel_size, frame_grouping=1):
        self.name = name
        self.file_path = file_path
        self.bounding_box = bounding_box
        self.target_shape = target_shape
        self.voxel_size = voxel_size
        self.frame_grouping = frame_grouping

    def voxelize(self, pcd):
        pcd = o3d.t.geometry.PointCloud(pcd).to_legacy()
        voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        return voxels.get_voxels()

    def load_data(self, file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        #data = data[:25]  # Limit to the first 25 sequences
        labels = [seq["action"] for seq in data]
        labels_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        #reverse_labels = {idx: label for label, idx in labels_to_id.items()}

        labels = [labels_to_id[label] for label in labels]
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

    def create_padded_voxel_tensor(self, voxels):
        dense_voxel_grid = self.create_dense_voxel_tensor(voxels, self.voxel_size, self.bounding_box)
        padded_voxel_grid = self.pad_or_trim_voxel_grid(dense_voxel_grid, self.target_shape)
        return tf.convert_to_tensor(padded_voxel_grid, dtype=tf.float32)
    
    def convert_voxels_to_dense_tensor(self, all_action_voxels, labels):
        input_tensor = []
        all_labels = []
        for actionIndex, action in enumerate(all_action_voxels):
            frame_grouping = []
            for i in range(0, len(action)):
                padded_voxel_tensor = self.create_padded_voxel_tensor(action[i])
                all_labels.append(labels[actionIndex])
                if(self.frame_grouping <= 1):
                    input_tensor.append(padded_voxel_tensor)
                    continue

                frame_grouping.append(padded_voxel_tensor)
                for y in range(1, self.frame_grouping):
                    frame_grouping.append(self.create_padded_voxel_tensor(action[i]))
                input_tensor.append(frame_grouping)
                frame_grouping = []
        
        return all_labels, input_tensor
    
    def setup(self):
        os.makedirs(os.path.dirname(CONFIG.INPUT_TENSOR_PATH), exist_ok=True)
        labels, all_voxels = self.load_data(self.file_path)
        chunk_size = max(1, len(all_voxels) // CONFIG.INPUT_TENSOR_CHUNK_SIZE)
        for index, start in enumerate(range(0, len(all_voxels), chunk_size)):
            file_path = f'{CONFIG.INPUT_TENSOR_PATH}{index}_{self.name}_{self.frame_grouping}_{self.target_shape}_{self.bounding_box}_{self.voxel_size}.p'
            if not os.path.isfile(file_path):
                continue

            end = start + chunk_size
            chunk_voxels = all_voxels[start:end]
            chunk_labels = labels[start:end]
            
            print("start create input tensor")
            chunk_input_labels, chunk_input_voxels = self.convert_voxels_to_dense_tensor(chunk_voxels, chunk_labels)
            print("end create input tensor")
            with open(file_path, 'wb') as path:
                pickle.dump((chunk_input_labels, chunk_input_voxels), path)


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
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        dataset = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    '''
    def create_dense_voxel_tensor(self, voxels):
        grid_shape = (
            int(np.ceil(self.bounding_box[0] / self.voxel_size)),
            int(np.ceil(self.bounding_box[1] / self.voxel_size)),
            int(np.ceil(self.bounding_box[2] / self.voxel_size)),
        )
        dense_grid = np.zeros(grid_shape, dtype=np.float32)

        for voxel in voxels:
            x, y, z = voxel.grid_index
            dense_grid[x, y, z] = 1  # Mark the voxel as occupied

        return dense_grid

    def pad_or_trim_voxel_grid(self, voxel_grid):
        padded_grid = np.zeros(self.target_shape, dtype=np.float32)

        min_shape = np.minimum(voxel_grid.shape, self.target_shape)
        slices = tuple(slice(0, s) for s in min_shape)
        padded_slices = tuple(slice(0, s) for s in self.target_shape)

        padded_grid[padded_slices] = voxel_grid[slices]

        return padded_grid

    def create_padded_voxel_tensor(self, voxels):
        dense_voxel_grid = self.create_dense_voxel_tensor(voxels)
        return self.pad_or_trim_voxel_grid(dense_voxel_grid)
    
    def create_padded_voxel_tensor(voxels, bounding_box, target_shape):
        dense_voxel_grid = create_dense_voxel_tensor(voxels, CONFIG.VOXEL_SIZE, bounding_box)
        padded_voxel_grid = pad_or_trim_voxel_grid(dense_voxel_grid, target_shape)
        return tf.convert_to_tensor(padded_voxel_grid, dtype=tf.float32)


    def prepare_voxel_tensor(self, voxel_grids):
        return tf.convert_to_tensor(voxel_grids, dtype=tf.float32)

                    for action_index, action_voxels in enumerate(all_voxels):
                for i in range(len(action_voxels)):
                    padded_voxel_tensor = self.create_padded_voxel_tensor(action_voxels[i])

                    if self.frame_grouping > 1:
                        frame_group = [padded_voxel_tensor] * self.frame_grouping
                        frame_group_tensor = tf.stack(frame_group, axis=0)
                        yield frame_group_tensor, labels[action_index]
                    else:
                        yield padded_voxel_tensor, labels[action_index]
'''