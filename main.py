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
        "architecture": "CNN-OwnNet",
        "dataset": "HmPEAR",
        "epochs": CONFIG.EPOCHS,
        "voxel_size": CONFIG.VOXEL_SIZE,
        "bounding_box": CONFIG.BOUNDING_BOX,
        "input_shape": CONFIG.INPUT_SHAPE,
        "frame_grouping": CONFIG.FRAME_GROUPING,
        "batch_size": CONFIG.BATCH_SIZE,
        "dropout": CONFIG.DROPOUT,
    }
)

# Setup the train data loader and retrieve the train_dataset
train_data_loader = TensorFlowDataLoader(
    name='train_dataloader',
    file_path=CONFIG.TRAINING_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
    shuffle_buffer=75000,
)
train_data_loader.setup()
print(train_data_loader.labels_to_id)
train_dataset = train_data_loader.get_dataset()

# Setup the validation data loader and retrieve the validation_dataset
valid_data_loader = TensorFlowDataLoader(
    name='valid_dataloader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
    shuffle_buffer=10000,
)
valid_data_loader.setup()
vali_dataset = valid_data_loader.get_dataset()


optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CONFIG.CHECKPOINT_PATH,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_freq='epoch'
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5
)

num_classes = len(train_data_loader.labels_to_id)
model = ModelBuilder.OwnNet((num_classes), CONFIG.INPUT_SHAPE, CONFIG.FRAME_GROUPING)
model.summary()
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=CONFIG.EPOCHS, validation_data=vali_dataset, callbacks=[model_checkpoint_callback, lr_scheduler, WandbMetricsLogger(log_freq=5)])
model.save("models/direct_regression.keras")

# ----------------------------------
# Test the trained model
test_data_loader = TensorFlowDataLoader(
    name='valid_dataloader',
    file_path=CONFIG.TEST_DATA_PATH,
    bounding_box=CONFIG.BOUNDING_BOX,
    target_shape=CONFIG.INPUT_SHAPE,
    voxel_size=CONFIG.VOXEL_SIZE,
    frame_grouping=CONFIG.FRAME_GROUPING,
    shuffle_buffer=10000,
)
test_data_loader.setup()
test_dataset = test_data_loader.get_dataset()

test_model = keras.models.load_model("models/direct_regression.keras")
score = test_model.evaluate(test_dataset)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
