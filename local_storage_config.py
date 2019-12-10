from os import sep

# Definitions for COCO 2017 dataset
DATASET_PATH = "." + sep + "dataset"
IMAGES_PATH = DATASET_PATH + sep + "images"
TRAIN_ANNS = DATASET_PATH + sep + "annotations" + sep + "person_keypoints_train2017.json"
VALID_ANNS = DATASET_PATH + sep + "annotations" + sep + "person_keypoints_val2017.json"

# will be used as output files
ROOT_TFRECORDS_PATH = "." + sep + "dataset" + sep + "TFrecords" + sep
TRAIN_TFRECORDS = ROOT_TFRECORDS_PATH + "training"
VALID_TFRECORDS = ROOT_TFRECORDS_PATH + "validation"

RESULTS_ROOT = "./tmp"
TENSORBOARD_PATH = RESULTS_ROOT + sep + "tensorboard"
CHECKPOINTS_PATH = RESULTS_ROOT + sep + "checkpoints"
MODELS_PATH = RESULTS_ROOT + sep + "models"
