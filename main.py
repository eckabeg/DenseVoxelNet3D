import wandb
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config as CONFIG
from tensorflow_dataloader import TensorFlowDataLoader


wandb.init(
    project = "activity-regocgnition",
    config={
        "learning_rate": CONFIG.LEARNING_RATE,
        "architecture": "CNN-MotionNet",
        "dataset": "HmPEAR",
        "epochs": CONFIG.EPOCHS,
        "voxel_size": CONFIG.VOXEL_SIZE,
        "input_shape": CONFIG.INPUT_SHAPE,
        "frame_grouping": CONFIG.FRAME_GROUPING,
        "batch_size": CONFIG.BATCH_SIZE,
        "dropout": CONFIG.DROPOUT,
    }
)

# Setup the train data loader and retrieve the train_dataset
train_data_loader = CONFIG.TRAIN_DATA_LOADER
train_data_loader.setup()
print(train_data_loader.labels_to_id)
train_dataset = train_data_loader.get_dataset()

# Setup the validation data loader and retrieve the validation_dataset
valid_data_loader = CONFIG.VALID_DATA_LOADER
valid_data_loader.setup()
vali_dataset = valid_data_loader.get_dataset()

# Train the model
model = CONFIG.MODEL
model.summary()
model.compile(optimizer=CONFIG.OPTIMIZER, loss=CONFIG.LOSS_FUNCTION, metrics=['accuracy'])
history = model.fit(train_dataset, epochs=CONFIG.EPOCHS, validation_data=vali_dataset, callbacks=CONFIG.MODEL_CALLBACKS)
model.save(CONFIG.MODEL_PATH)

# ----------------------------------
# Test the trained model
test_data_loader = CONFIG.TEST_DATA_LOADER
test_data_loader.setup()
test_dataset = test_data_loader.get_dataset()

test_model = keras.models.load_model(CONFIG.MODEL_PATH)
score = test_model.evaluate(test_dataset)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')