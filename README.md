# DenseVoxelNet3D

## Introduction
LiDAR-based point cloud analysis for Human Activity Recognition (HAR) is increasingly relevant in assistive technologies, particularly for supporting elderly and care-dependent individuals through applications such as fall detection. In this work, we present DenseVoxelNet3D, a compact and efficient 3D convolutional neural network designed for voxelized LiDAR input. The architecture combines a voxelization-based preprocessing pipeline with densely con-
nected layers to improve feature propagation and mitigate vanishing gradient issues. By integrating global average pooling and dropoutbased regularization, the model achieves strong generalization with minimal overfitting. On the [HmPEAR dataset](http://www.lidarhumanmotion.net/hmpear/), DenseVoxelNet3D outperforms conventional architectures such as AlexNet and ResNet in terms of classification accuracy and computational efficiency.

## Classes

### Config-Class
The script starts by defining the paths for the training and test datasets, as well as setting various configuration parameters, such as voxel size, bounding box dimensions, frame grouping, and input shape.

The next step involves initializing the data loaders for training and validation datasets using these configurations. The training data loader is configured with the specified parameters, including the file path, target shape, voxel size, frame grouping, and a shuffle buffer size of 75,000. The validation data loader is similarly configured, with a shuffle buffer size of 10,000. The test data loader is set to be the same as the validation data loader.

An input tensor path and chunk size are defined, along with a list of labels that represent different actions.

The script then sets several hyperparameters, including the number of epochs (100), learning rate (0.001), and batch size (32). Paths for saving model checkpoints and the final trained model are specified.

The optimizer is set to Adam with the defined learning rate, and the loss function is set to sparse categorical cross-entropy. A list of callbacks is defined, including a model checkpoint callback to save the best model based on validation loss, a learning rate reduction callback that reduces the learning rate when the validation loss plateaus, and a metrics logger callback to log training metrics to Weights and Biases (wandb).

Finally, the script provides options for three different 3D CNN model architectures: MotionNet, AlexNet, and ResNet. Each model can be configured with different activation functions, dropout rates, and regularization settings. The chosen model is set to MotionNet in this case, with ReLU activation, specified dropout rates, and L2 regularization.

### Main-Class
First, the training data loader is initialized and set up using predefined parameters from the CONFIG object. Once the data loader is set up, the labels-to-ID mappings are printed, and the training dataset is retrieved.

Similarly, the validation data loader is set up using configurations from the CONFIG object, and the validation dataset is retrieved.

Next, the model training process begins:

- The model is initialized using the configuration specified in the CONFIG object.
- The model's architecture is displayed to provide an overview of its structure.
- The model is compiled with the specified optimizer, loss function, and accuracy metric.
- The model is trained using the training dataset and validated with the validation dataset. The training runs for a specified number of epochs, and various callbacks are used to monitor the training progress.
- After training, the model is saved to a specified path for future use.

Finally, the performance of the trained model is evaluated on a test dataset. This involves setting up the test data loader with configurations from the CONFIG object and retrieving the test dataset. The trained model is loaded from the saved path, and its performance is evaluated on the test dataset, yielding test loss and accuracy metrics.

### Model Builder
The ModelBuilder class is designed for constructing 3D Convolutional Neural Network (CNN) models, with a particular focus on incorporating various attention mechanisms to enhance model performance. The class includes multiple methods, each implementing a specific architectural component or mechanism.

Channel Attention Mechanism: The channel attention mechanism enhances feature maps by recalibrating channel-wise feature responses. It works by computing global context through global average pooling and learning channel weights using dense layers. The learned weights are then applied to the original features, effectively focusing on the most informative channels.

Spatial Attention Mechanism: The spatial attention mechanism emphasizes important spatial regions within the feature maps. It utilizes a convolutional layer to learn spatial weights and applies these weights to the original features, highlighting key spatial areas that contribute to the model's performance.

Convolutional Block Attention Module (CBAM): The CBAM integrates both channel and spatial attention mechanisms. It first applies channel attention to enhance the most informative channels and then applies spatial attention to highlight crucial spatial regions. This combination helps in capturing both channel-wise and spatial dependencies, improving the overall representation.

Squeeze-and-Excitation (SE) Block: The SE block is another attention mechanism that recalibrates channel-wise feature responses. It computes global context using global average pooling and learns channel weights through dense layers. These weights are then applied to the original features to focus on the most important channels.

MotionNet Architecture: The MotionNet architecture is a 3D CNN model specifically designed for activity recognition. It consists of several convolutional layers with batch normalization and activation functions, followed by pooling layers. The network also includes global average pooling, dropout layers to prevent overfitting, and fully connected layers with specified activation functions. The final layer uses the softmax activation to output class probabilities.

AlexNet Architecture: The AlexNet architecture is another 3D CNN model for activity recognition. It includes multiple convolutional layers with batch normalization, activation functions, and pooling layers. The architecture also features dropout layers and fully connected layers, similar to the original AlexNet but adapted for 3D input.

ResNet Architecture: The ResNet architecture incorporates residual connections to build a 3D CNN model for activity recognition. It starts with an initial convolution and pooling layer, followed by several residual blocks. Each residual block contains convolutional layers with batch normalization and activation functions, and it may include skip connections to improve gradient flow. The final layers include global average pooling, dropout, and fully connected layers, with the output layer using softmax activation.

### Predict
The script begins by loading a pre-trained model from the specified file path using keras.models.load_model.

Next, a validation data loader is initialized with configurations that include the file path, bounding box dimensions, target shape, voxel size, and frame grouping parameters. The validation data loader is set up to prepare it for loading data.

The validation dataset is created from TensorFlow records. The TFRecordDataset is mapped using the parse_tfrecord method from the validation data loader, and further operations such as shuffling, batching, and prefetching are applied to optimize the data pipeline.

The script then enters a loop to iterate through the batches of voxels and labels from the validation dataset. For each voxel and label in the batch:

- The voxels are expanded along a new axis to match the input shape expected by the model.
- The model predicts probabilities for each class by passing the voxels through the model and converting the output to a NumPy array.
- The top 5 class indices with the highest probabilities are identified.

For each voxel sample, the ground truth label is printed along with the top 5 predicted classes and their corresponding confidence levels.

The loop is set to break after processing a few samples (in this case, after 4 iterations) to limit the output.

### Residual Connections
The ResidualConnectionsBuilder class contains a method called residual_block that constructs a residual block for a 3D Convolutional Neural Network (CNN). Residual blocks are essential components of ResNet architectures, which allow for deeper networks by facilitating the flow of gradients.

The residual_block method performs the following steps:

- Save Input for Skip Connection: The input tensor x is saved as shortcut for use later in the skip connection.
- First Convolution: A 3D convolution is applied to the input tensor x, using the specified number of filters, kernel_size, and stride. The result is then batch normalized and activated using the ReLU function.
- Second Convolution: A second 3D convolution is applied to the output of the first convolution, with the same number of filters and kernel_size, but with a stride of 1. This is also followed by batch normalization.
- Adjust Shortcut Dimensions if Needed: If the stride is not equal to 1 or the number of channels in the shortcut does not match the number of filters, the shortcut tensor is adjusted using a 1x1 3D convolution. This ensures that the dimensions of the shortcut tensor match those of the main path.
- Add Skip Connection: The adjusted shortcut tensor is added to the output of the second convolution using element-wise addition. This forms the skip connection, which helps preserve gradient flow and allows for deeper networks.
- ReLU Activation: Finally, a ReLU activation is applied to the result of the addition, producing the output tensor of the residual block.

The residual_block method effectively builds a building block for residual networks, enabling the creation of deeper and more effective 3D CNNs by leveraging skip connections.

### Visualizer
The provided script includes two functions designed to visualize point clouds from files.

The first function, display\_folder, processes all files in a specified directory and visualizes the point clouds contained within those files in a single window. It begins by encoding the directory path and iterating through each file within the directory. For each file, it reads the raw point cloud data, reshapes it into a format suitable for visualization, and adds a randomly assigned color. The point clouds are collected into a list, which is then displayed using Open3D's visualization tools.

The second function, display\_cloud, focuses on reading and displaying a single point cloud file. It reads the raw point cloud data from the specified file path, reshapes it, and visualizes the resulting point cloud using Open3D.

### Input Tensor Visualizer
The TensorFlowDataLoader class is designed to handle the loading, processing, and preparation of point cloud data for use in a TensorFlow-based 3D CNN model. Below is a detailed explanation of the class and its methods:

