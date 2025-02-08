EPOCHS              = 500


VOXEL_SIZE          = 0.05


DROPOUT             = 0.5

# Defines how many frames will be grouod together by the data loader
BOUNDING_BOX        = (10, 10, 10)
INPUT_SHAPE         = (64, 64, 64)
FRAME_GROUPING      = 2



LEARNING_RATE       = 0.00001
BATCH_SIZE          = 4

TRAINING_DATA_PATH  = 'data/HmPEAR/label/train_act.pkl'
TEST_DATA_PATH      = 'data/HmPEAR/label/test_act.pkl'

CHECKPOINT_PATH     = "models/checkpoints/direct_regression.model.keras"

INPUT_TENSOR_PATH   = "data/HmPEAR/input_tensor/temp"
INPUT_TENSOR_CHUNK_SIZE          = 25
