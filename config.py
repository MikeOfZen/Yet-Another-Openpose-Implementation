from os import sep
import numpy as np


#Training Mode
TPU_MODE=False


#Definitions for COCO 2017 dataset
DATASET_PATH="." + sep + "dataset"
IMAGES_PATH= DATASET_PATH + sep + "images"
TRAIN_ANNOTATIONS_PATH= DATASET_PATH + sep + "annotations" + sep + "person_keypoints_train2017.json"
VALIDATION_ANNOTATIONS_PATH= DATASET_PATH + sep + "annotations" + sep + "person_keypoints_val2017.json"

#will be used as output files
TRANSFORMED_ANNOTATIONS_PATH="." + sep + "dataset" + sep + "transformed" + sep + ""
TRANSFORMED_TRAIN_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"person_keypoints_train2017"
TRANSFORMED_VALIDATION_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"person_keypoints_val2017"

#Dataset reference values
DATASET_SIZE=56000 #exact size not critical
DATASET_VAL_SIZE=2500

#this determines the size images will be resized to, and the size of the labels vreated
IMAGE_WIDTH=368
IMAGE_HEIGHT=368
IMAGE_SIZE=(IMAGE_HEIGHT,IMAGE_WIDTH)
LABEL_HEIGHT=46 #this stems from the model label output size
LABEL_WIDTH=46 #same

#model settings
PAF_NUM_FILTERS=38
HEATMAP_NUM_FILTERS=17
BATCH_NORMALIZATION_ON=False

#taken directly from the annotations JSON file
DATASET_KPTS=['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                'right_knee', 'left_ankle', 'right_ankle']
#shifted by -1 to match keypoints idx
DATASET_JOINTS=[[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
                [ 1, 2], [ 0, 1], [ 0, 2], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6]]

#this is the gaussian spot sie that will be drawn on the training labels
GAUSSIAN_SPOT_SIGMA_SQ=0.02 #used for the size of the gaussian spot for each keypoint
JOINT_WIDTH=0.02  #used for the width of the vector field generated for each joint as a PAF, the unit is relative to image size ie 0..1
                    #for lower resolution, a value too low (~0.005) here will make the vectors too sparse


#dataset settings
SHUFFLE=True
PREFETCH=10  #size of prefetch size, 0 to disable
CACHE=False #depends on available memory size, around 20gb required for both cache and graph

BATCH_SIZE=2  #for use when on cpu for development, if on GPU, can safely increase
if TPU_MODE:
    BATCH_SIZE=128  #for the size of the dataset this is optimizied for tpuv2-8 node. lower this if getting OOM or tpu crashes

STEPS_PER_EPOCH=int(DATASET_SIZE/BATCH_SIZE)

#Training settings
TRAINING_EPOCHS=100

#adam_learning_rate=0.001  #for reference
BASE_LEARNING_RATE=0.001
LEARNING_RATE_SCHEDUELE=np.zeros(1000)
LEARNING_RATE_SCHEDUELE[:3]=0.2
LEARNING_RATE_SCHEDUELE[3:20]=3
LEARNING_RATE_SCHEDUELE[20:40]=2
LEARNING_RATE_SCHEDUELE[40:60]=1
LEARNING_RATE_SCHEDUELE[60:]=0.1
LEARNING_RATE_SCHEDUELE*=BASE_LEARNING_RATE



TENSORBOARD_PATH="." + sep + "tmp" + sep + "tensorboard" + sep + "" #this will get overriden by tpu_config is used
CHECKPOINTS_PATH="." + sep + "tmp" + sep + "checkpoints" + sep + "" #this will get overriden by tpu_config is used
MODELS_PATH="." + sep + "tmp" + sep + "models" + sep + ""



if TPU_MODE:
    from tpu_training.config_tpu import *