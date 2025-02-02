import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config as CONFIG
from model_builder import ModelBuilder
from tensorflow_dataloader import TensorFlowDataLoader

# Setup the train data loader and get the train_dataset
train_data_loader = TensorFlowDataLoader(
    name='train_dataloader',
    file_path=CONFIG.TRAINING_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

train_data_loader_setup_start = time.time()
print('Startng the setup of the train_data_loader.')
train_data_loader.setup()
train_data_loader_setup_end = time.time()
print('Finished setup of the train_data_loader after: ', train_data_loader_setup_end - train_data_loader_setup_start)

train_dataset = train_data_loader.get_tf_dataset()
train_prepared_dataset = train_dataset.shuffle(100).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_classes = len(train_data_loader.labels_to_id)
model = ModelBuilder.AlexNet((num_classes), CONFIG.INPUT_SHAPE, CONFIG.FRAME_GROUPING)
model.summary()
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CONFIG.CHECKPOINT_PATH,
    monitor='loss',
    mode='auto',
    save_best_only=True,
    save_freq='epoch')
optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_prepared_dataset, epochs=CONFIG.EPOCHS, callbacks=[model_checkpoint_callback])
model.save("models/direct_regression.keras")

# ----------------------------------
# Test the trained model
test_data_loader = TensorFlowDataLoader(
    name='test_dataloader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=(10, 10, 10),
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

test_data_loader_setup_start = time.time()
print('Startng the setup of the test_data_loader.')
test_data_loader.setup()
test_data_loader_setup_end = time.time()
print('Finished setup of the test_data_loader after: ', test_data_loader_setup_end - test_data_loader_setup_start)

test_dataset = test_data_loader.get_tf_dataset()

# Apply preprocessing
test_prepared_dataset = test_dataset.shuffle(100).batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_model = keras.models.load_model("models/direct_regression.keras")
predicted_labels = test_model(test_prepared_dataset, training=False)

predicted_labels = tf.argmax(predicted_labels, axis=-1).numpy()
true_labels = test_data_loader.all_labels

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print predicted labels
print("Predicted Labels:", predicted_labels)