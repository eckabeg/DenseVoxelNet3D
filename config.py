from keras import regularizers, callbacks
import tensorflow as tf
from model_builder import ModelBuilder
from tensorflow_dataloader import TensorFlowDataLoader
from wandb.integration.keras import WandbMetricsLogger

# Path to training data
TRAINING_DATA_PATH  = 'data/HmPEAR/label/train_act.pkl'
# Path to test data
TEST_DATA_PATH      = 'data/HmPEAR/label/test_act.pkl'

# Set voxel size for 3D grid
VOXEL_SIZE          = 0.05
# Bounding box in which all actions get fitted in 
BOUNDING_BOX        = (32, 32, 32)
# Number of frames to group for processing
FRAME_GROUPING      = 5
# Define input shape using bounding box and frame grouping
INPUT_SHAPE         = BOUNDING_BOX + (FRAME_GROUPING,)

TRAIN_DATA_LOADER = TensorFlowDataLoader(
    name='train_dataloader',       # Name of the data loader
    file_path=TRAINING_DATA_PATH,  # Path to the training data
    target_shape=BOUNDING_BOX,     # Shape of the bounding box
    voxel_size=VOXEL_SIZE,         # Voxel size for 3D grid
    frame_grouping=FRAME_GROUPING, # Number of frames to group
    shuffle_buffer=75000,          # Buffer size for shuffling data
)

VALID_DATA_LOADER = TensorFlowDataLoader(
    name='valid_dataloader',       # Name of the validation data loader
    file_path=TEST_DATA_PATH,      # Path to the test data
    target_shape=INPUT_SHAPE,      # Define input shape using bounding box and frame grouping
    voxel_size=VOXEL_SIZE,         # Voxel size for 3D grid
    frame_grouping=FRAME_GROUPING, # Number of frames to group
    shuffle_buffer=10000,          # Buffer size for shuffling validation data
)
# Data loader used for testing
TEST_DATA_LOADER    = VALID_DATA_LOADER

# Input tensor path
INPUT_TENSOR_PATH   = 'data/HmPEAR/input_tensor/'
# Chunk size
INPUT_TENSOR_CHUNK_SIZE          = 4
#ALL_LABELS         = ['bend_over', 'carry_sth', 'cast_a_ball', 'catch_a_ball', 'clap_hands', 'close_an_unbrella', 'drink_sth', 'hand_waving', 'jump_forward', 'jump_up', 'kick_left_leg', 'kick_right_leg', 'kicking_sth', 'look_at_the_phone', 'make_phone_calls', 'open_an_unbrella', 'pick_up', 'put_on_backpack', 'put_on_coat', 'put_on_earphone', 'put_on_hat', 'put_sth_into_bag', 'running', 'shake_head', 'sit_down', 'sitting', 'squat_down', 'stand_up', 'standing', 'stretch_oneself', 'swing_a_racket', 'take_off_backpack', 'take_off_coat', 'take_off_earphone', 'take_off_hat', 'take_photo', 'take_sth_from_bag', 'throw', 'turn_around', 'walking', 'watching_back', 'wave_left_hand', 'wave_right_hand']
# Labels used to train the model
LABELS              = [
                        'bend_over', 'carry_sth', 'drink_sth', 'hand_waving', 'jump_forward',
                        'jump_up', 'pick_up', 'running', 'sit_down', 'sitting',
                        'squat_down', 'stand_up', 'standing', 'stretch_oneself', 'turn_around',
                        'walking', 'wave_left_hand', 'wave_right_hand'
                    ]

# Amount of epochs
EPOCHS              = 100
# Learning rate
LEARNING_RATE       = 0.001

# Size of batch
BATCH_SIZE          = 32

# Path to save checkpoints
CHECKPOINT_PATH     = 'models/checkpoints/direct_regression.model.keras'
# Path to save model
MODEL_PATH          = 'models/direct_regression.keras'

# Define Adam optimizer with specified learning rate
OPTIMIZER           = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# Define loss function with sparse categorial crossentropy
LOSS_FUNCTION       = tf.keras.losses.SparseCategoricalCrossentropy()

MODEL_CALLBACKS = [
    callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,    # Save model checkpoints to the specified path
        monitor='val_loss',          # Monitor validation loss
        mode='auto',                 # Automatically choose the direction to monitor
        save_best_only=True,         # Save only the best model
        save_freq='epoch'            # Save checkpoints at the end of each epoch
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5  # Reduce learning rate when validation loss plateaus
    ),
    WandbMetricsLogger(log_freq=5)  # Log metrics to Weights & Biases every 5 epochs


# MotionNet
ACTIVATION_FUNCTION = 'relu'
DROPOUT             = [0.4, 0.4, 0.3]
CONV_REGULARIZER    = regularizers.l2(0.005)
MODEL               = ModelBuilder.ModtionNet(len(LABELS), INPUT_SHAPE, CONV_REGULARIZER, DROPOUT, ACTIVATION_FUNCTION)

# AlexNet
# ACTIVATION_FUNCTION = 'relu'
# DROPOUT             = [0.4, 0.4, 0.3]
# CONV_REGULARIZER    = regularizers.l2(0.005)
# MODEL               = ModelBuilder.AlexNet(len(LABELS), INPUT_SHAPE, CONV_REGULARIZER, DROPOUT, ACTIVATION_FUNCTION)

# ResNet
# ACTIVATION_FUNCTION = 'relu'
# DROPOUT             = [0.4, 0.3]
# CONV_REGULARIZER    = regularizers.l2(0.005)
# MODEL               = ModelBuilder.ResNet(len(LABELS), INPUT_SHAPE, CONV_REGULARIZER, DROPOUT, ACTIVATION_FUNCTION)