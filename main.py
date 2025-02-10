import wandb
from wandb.integration.keras import WandbMetricsLogger
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config as CONFIG
from model_builder import ModelBuilder
from tensorflow_dataloader import TensorFlowDataLoader

wandb.init(
    project = "activity-regocgnition",
    config={
        "learning_rate": CONFIG.LEARNING_RATE,
        "architecture": "CNN-ResNet18",
        "dataset": "HmPEAR",
        "epochs": CONFIG.EPOCHS,
        "voxel_size": CONFIG.VOXEL_SIZE,
        "bounding_box": CONFIG.BOUNDING_BOX,
        "input_shape": CONFIG.INPUT_SHAPE,
        "frame_grouping": CONFIG.FRAME_GROUPING,
        "batch_size": CONFIG.BATCH_SIZE
    }
)

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

print(train_data_loader.labels_to_id)

#train_dataset = train_data_loader.get_tf_dataset()

train_dataset = (
    tf.data.TFRecordDataset(train_data_loader.TFRecord_file_paths)
    .map(train_data_loader.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .shuffle(1000)
    .batch(CONFIG.BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

valid_data_loader = TensorFlowDataLoader(
    name='valid_dataloader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

valid_data_loader_setup_start = time.time()
print('Startng the setup of the valid_data_loader.')
valid_data_loader.setup()
valid_data_loader_setup_end = time.time()
print('Finished setup of the valid_data_loader after: ', valid_data_loader_setup_end - valid_data_loader_setup_start)

vali_dataset = (
    tf.data.TFRecordDataset(valid_data_loader.TFRecord_file_paths)
    .map(valid_data_loader.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .shuffle(1000)
    .batch(CONFIG.BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CONFIG.CHECKPOINT_PATH,
    monitor='loss',
    mode='auto',
    save_best_only=True,
    save_freq='epoch'
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

num_classes = len(train_data_loader.labels_to_id)
model = ModelBuilder.ResNet((num_classes), CONFIG.INPUT_SHAPE, CONFIG.FRAME_GROUPING)
model.summary()
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=CONFIG.EPOCHS, validation_data=vali_dataset, callbacks=[model_checkpoint_callback, lr_scheduler, WandbMetricsLogger(log_freq=5)])
model.save("models/direct_regression.keras")

# ----------------------------------
# Test the trained model
test_data_loader = TensorFlowDataLoader(
    name='test_dataloader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
)

test_data_loader_setup_start = time.time()
print('Startng the setup of the test_data_loader.')
test_data_loader.setup()
test_data_loader_setup_end = time.time()
print('Finished setup of the test_data_loader after: ', test_data_loader_setup_end - test_data_loader_setup_start)

test_dataset = (
    tf.data.TFRecordDataset(test_data_loader.TFRecord_file_paths)
    .map(test_data_loader.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .shuffle(1000)
    .batch(CONFIG.BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_model = keras.models.load_model("models/direct_regression.keras")
score = test_model.evaluate(test_dataset)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
