from os import sep
import numpy as np
from keypoints_config import *

TPU_MODE=False
#Training Mode
INCLUDE_MASK=True
ASK_FOR_CHECKPOINTS=True
RUN_NAME=""

#Definitions for COCO 2017 dataset
DATASET_PATH="." + sep + "dataset"
IMAGES_PATH= DATASET_PATH + sep + "images"
TRAIN_ANNS= DATASET_PATH + sep + "annotations" + sep + "person_keypoints_train2017.json"
VALID_ANNS= DATASET_PATH + sep + "annotations" + sep + "person_keypoints_val2017.json"

#will be used as output files
ROOT_TFRECORDS_PATH= "." + sep + "dataset" + sep + "TFrecords" + sep
TRAIN_TFRECORDS= ROOT_TFRECORDS_PATH + "training"
VALID_TFRECORDS= ROOT_TFRECORDS_PATH + "validation"

#Dataset reference values
DATASET_SIZE=56000 #exact size not critical
DATASET_VAL_SIZE=2500

IMAGES_PER_TFRECORD=1000

#this determines the size images will be resized to, and the size of the labels vreated
IMAGE_WIDTH=368
IMAGE_HEIGHT=368
IMAGE_SIZE=(IMAGE_HEIGHT,IMAGE_WIDTH) #for convinience
LABEL_HEIGHT=46 #this stems from the model label output size, cannot be configured!!! must be derived from model.
LABEL_WIDTH=46 #same

#model settings
PAF_NUM_FILTERS= len(JOINTS_DEF) * 2
HEATMAP_NUM_FILTERS=len(KEYPOINTS_DEF)
BATCH_NORMALIZATION_ON=False



#this is the gaussian spot sie that will be drawn on the training labels
KPT_HEATMAP_GAUSSIAN_SIGMA_SQ=0.02 #used for the size of the gaussian spot for each keypoint
#JOINT_WIDTH=0.02  #used for the width of the vector field generated for each joint as a PAF, the unit is relative to image size ie 0..1
                    #for lower resolution, a value too low (~0.005) here will make the vectors too sparse
PAF_GAUSSIAN_SIGMA_SQ=0.0015 #similiar to joint width, but works on gaussian width,tradeoff between model certainty and number of persons that can be discriminated in a frame

#dataset settings
SHUFFLE=True
PREFETCH=10  #size of prefetch size, 0 to disable
CACHE=True  #depends on available memory size, around 20gb required for both cache and graph

BATCH_SIZE=2  #for use when on cpu for development, if on GPU, can safely increase
STEPS_PER_EPOCH=int(DATASET_SIZE/BATCH_SIZE)


#Training settings
TRAINING_EPOCHS=100

#adam_learning_rate=0.001  #for reference
BASE_LEARNING_RATE=0.001
LEARNING_RATE_SCHEDUELE=np.zeros(1000)
LEARNING_RATE_SCHEDUELE[:3]=1
LEARNING_RATE_SCHEDUELE[3:20]=1
LEARNING_RATE_SCHEDUELE[20:40]=1
LEARNING_RATE_SCHEDUELE[40:100]=0.5
LEARNING_RATE_SCHEDUELE[100:]=0.3
LEARNING_RATE_SCHEDUELE*=BASE_LEARNING_RATE


RESULTS_ROOT="./tmp"
TENSORBOARD_PATH=RESULTS_ROOT + sep +"tensorboard" #this will get overriden by tpu_config is used
CHECKPOINTS_PATH=RESULTS_ROOT+ sep +"checkpoints" #this will get overriden by tpu_config is used
MODELS_PATH=RESULTS_ROOT + sep + "models"

