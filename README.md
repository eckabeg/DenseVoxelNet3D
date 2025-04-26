# DenseVoxelNet3D



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

    The model is initialized using the configuration specified in the CONFIG object.
    The model's architecture is displayed to provide an overview of its structure.
    The model is compiled with the specified optimizer, loss function, and accuracy metric.
    The model is trained using the training dataset and validated with the validation dataset. The training runs for a specified number of epochs, and various callbacks are used to monitor the training progress.
    After training, the model is saved to a specified path for future use.

Finally, the performance of the trained model is evaluated on a test dataset. This involves setting up the test data loader with configurations from the CONFIG object and retrieving the test dataset. The trained model is loaded from the saved path, and its performance is evaluated on the test dataset, yielding test loss and accuracy metrics.
