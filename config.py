EPOCHS              = 20


VOXEL_SIZE          = 0.05


DROPOUT             = 0.5

# Defines how many frames will be grouod together by the data loader
BOUNDING_BOX        = (10, 10, 10)
INPUT_SHAPE         = (64, 64, 64)
FRAME_GROUPING      = 5



LEARNING_RATE       = 0.0001
CONV_REGULARIZERS   = 0.0001
BATCH_SIZE          = 32

TRAINING_DATA_PATH  = 'data/HmPEAR/label/train_act.pkl'
TEST_DATA_PATH      = 'data/HmPEAR/label/test_act.pkl'

CHECKPOINT_PATH     = "models/checkpoints/direct_regression.model.keras"

INPUT_TENSOR_PATH   = "data/HmPEAR/input_tensor/"
INPUT_TENSOR_CHUNK_SIZE          = 10


#ALL_LABELS         = ['bend_over', 'carry_sth', 'cast_a_ball', 'catch_a_ball', 'clap_hands', 'close_an_unbrella', 'drink_sth', 'hand_waving', 'jump_forward', 'jump_up', 'kick_left_leg', 'kick_right_leg', 'kicking_sth', 'look_at_the_phone', 'make_phone_calls', 'open_an_unbrella', 'pick_up', 'put_on_backpack', 'put_on_coat', 'put_on_earphone', 'put_on_hat', 'put_sth_into_bag', 'running', 'shake_head', 'sit_down', 'sitting', 'squat_down', 'stand_up', 'standing', 'stretch_oneself', 'swing_a_racket', 'take_off_backpack', 'take_off_coat', 'take_off_earphone', 'take_off_hat', 'take_photo', 'take_sth_from_bag', 'throw', 'turn_around', 'walking', 'watching_back', 'wave_left_hand', 'wave_right_hand']
LABELS              = [
                        'bend_over', 'carry_sth', 'drink_sth', 'hand_waving', 'jump_forward',
                        'jump_up', 'kicking_sth', 'look_at_the_phone', 'pick_up', 'running',
                        'sit_down', 'sitting', 'squat_down', 'stand_up', 'standing',
                        'stretch_oneself', 'take_sth_from_bag', 'turn_around', 'walking', 'wave_left_hand',
                        'wave_right_hand'
                    ]
