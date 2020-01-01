from configs.keypoints_config import *

TPU_MODE = False
RUN_NAME = ""
# Training Mode
INCLUDE_MASK = True
SAVE_CHECKPOINTS = True
SAVE_TENSORBOARD = True
ASK_FOR_CHECKPOINTS = True


# this determines the size images will be resized to, and the size of the labels created
IMAGE_WIDTH = 368
IMAGE_HEIGHT = 368
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)  # for convenience
LABEL_HEIGHT = 46  # this stems from the model label output size, cannot be configured!!! must be derived from model.
LABEL_WIDTH = 46  # same

# model settings
PAF_NUM_FILTERS = len(JOINTS_DEF) * 2  # dont change
HEATMAP_NUM_FILTERS = len(KEYPOINTS_DEF)  # dont change
BATCH_NORMALIZATION_ON = False
DROPOUT_RATE = 0  # set to 0 to disable

# this is the gaussian spot sie that will be drawn on the training labels
KPT_HEATMAP_GAUSSIAN_SIGMA_SQ = 0.008  # used for the size of the gaussian spot for each keypoint
# JOINT_WIDTH=0.02  #used for the width of the vector field generated for each joint as a PAF, the unit is relative to image size ie 0..1
# for lower resolution, a value too low (~0.005) here will make the vectors too sparse
PAF_GAUSSIAN_SIGMA_SQ = 0.0015  # similar to joint width, but works on gaussian width, tradeoff between model certainty and number of persons that can be discriminated in a frame

# dataset settings
SHUFFLE = True
SHUFFLE_BUFFER = 1000
PREFETCH = 10  # size of prefetch size, 0 to disable
CACHE = False  # depends on available memory size, around 20gb required for both cache and graph

# Dataset reference values
DATASET_SIZE = 56000  # exact size not critical
DATASET_VAL_SIZE = 2500

IMAGES_PER_TFRECORD = 1000

BATCH_SIZE = 1  # for use when on cpu for development, if on GPU, can safely increase

# Augmentation settings
IMAGE_AUG = True
CONTRAST_RANGE = 0.5
BRIGHTNESS_RANGE = 0.2
HUE_RANGE = 0.1
SATURATION_RANGE = 0.2
MIRROR_AUG = True
