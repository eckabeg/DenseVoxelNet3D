from keras import regularizers, callbacks
import tensorflow as tf
from model_builder import ModelBuilder
from tensorflow_dataloader import TensorFlowDataLoader
from wandb.integration.keras import WandbMetricsLogger

TRAINING_DATA_PATH  = 'data/HmPEAR/label/train_act.pkl'
TEST_DATA_PATH      = 'data/HmPEAR/label/test_act.pkl'

VOXEL_SIZE          = 0.05
BOUNDING_BOX        = (32, 32, 32)
FRAME_GROUPING      = 5
INPUT_SHAPE         = BOUNDING_BOX + (FRAME_GROUPING,)

TRAIN_DATA_LOADER   = TensorFlowDataLoader(
    name='train_dataloader',
    file_path=TRAINING_DATA_PATH,
    target_shape=BOUNDING_BOX,
    voxel_size=VOXEL_SIZE,
    frame_grouping=FRAME_GROUPING,
    shuffle_buffer=75000,
)
VALID_DATA_LOADER   = TensorFlowDataLoader(
    name='valid_dataloader',
    file_path=TEST_DATA_PATH,
    target_shape=INPUT_SHAPE,
    voxel_size=VOXEL_SIZE,
    frame_grouping=FRAME_GROUPING,
    shuffle_buffer=10000,
) 
TEST_DATA_LOADER    = VALID_DATA_LOADER


INPUT_TENSOR_PATH   = 'data/HmPEAR/input_tensor/'
INPUT_TENSOR_CHUNK_SIZE          = 4
#ALL_LABELS         = ['bend_over', 'carry_sth', 'cast_a_ball', 'catch_a_ball', 'clap_hands', 'close_an_unbrella', 'drink_sth', 'hand_waving', 'jump_forward', 'jump_up', 'kick_left_leg', 'kick_right_leg', 'kicking_sth', 'look_at_the_phone', 'make_phone_calls', 'open_an_unbrella', 'pick_up', 'put_on_backpack', 'put_on_coat', 'put_on_earphone', 'put_on_hat', 'put_sth_into_bag', 'running', 'shake_head', 'sit_down', 'sitting', 'squat_down', 'stand_up', 'standing', 'stretch_oneself', 'swing_a_racket', 'take_off_backpack', 'take_off_coat', 'take_off_earphone', 'take_off_hat', 'take_photo', 'take_sth_from_bag', 'throw', 'turn_around', 'walking', 'watching_back', 'wave_left_hand', 'wave_right_hand']
LABELS              = [
                        'bend_over', 'carry_sth', 'drink_sth', 'hand_waving', 'jump_forward',
                        'jump_up', 'pick_up', 'running', 'sit_down', 'sitting',
                        'squat_down', 'stand_up', 'standing', 'stretch_oneself', 'turn_around',
                        'walking', 'wave_left_hand', 'wave_right_hand'
                    ]


EPOCHS              = 100
LEARNING_RATE       = 0.001

BATCH_SIZE          = 32

CHECKPOINT_PATH     = 'models/checkpoints/direct_regression.model.keras'
MODEL_PATH          = 'models/direct_regression.keras'


OPTIMIZER           = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNCTION       = tf.keras.losses.SparseCategoricalCrossentropy()
MODEL_CALLBACKS     = [
    callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        save_freq='epoch'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    ),
    WandbMetricsLogger(log_freq=5)
]

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