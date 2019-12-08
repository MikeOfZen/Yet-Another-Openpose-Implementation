from os import sep
import numpy as np
from keypoints_config import *

#Training Mode
TPU_MODE=True
INCLUDE_MASK=True 
ASK_FOR_CHECKPOINTS=True

#Definitions for COCO 2017 dataset
DATASET_PATH="." + sep + "dataset"
IMAGES_PATH= DATASET_PATH + sep + "images"
TRAIN_ANNOTATIONS_PATH= DATASET_PATH + sep + "annotations" + sep + "person_keypoints_train2017.json"
VALIDATION_ANNOTATIONS_PATH= DATASET_PATH + sep + "annotations" + sep + "person_keypoints_val2017.json"

#will be used as output files
TRANSFORMED_ANNOTATIONS_PATH="." + sep + "dataset" + sep + "TFrecords" + sep + ""
TRANSFORMED_TRAIN_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"training"
TRANSFORMED_VALIDATION_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"validation"

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
PAF_NUM_FILTERS= len(JOINTS_DEF) * 2
HEATMAP_NUM_FILTERS=len(KEYPOINTS_DEF)
BATCH_NORMALIZATION_ON=False



#this is the gaussian spot sie that will be drawn on the training labels
KPT_HEATMAP_GAUSSIAN_SIGMA_SQ=0.02 #used for the size of the gaussian spot for each keypoint
#JOINT_WIDTH=0.02  #used for the width of the vector field generated for each joint as a PAF, the unit is relative to image size ie 0..1
                    #for lower resolution, a value too low (~0.005) here will make the vectors too sparse
PAF_GAUSSIAN_SIGMA_SQ=0.003 #similiar to joint width, but works on gaussian width,tradeoff between model certainty and number of persons that can be discriminated in a frame

#dataset settings
SHUFFLE=True  
PREFETCH=10  #size of prefetch size, 0 to disable
CACHE=True  #depends on available memory size, around 20gb required for both cache and graph 

BATCH_SIZE=2  #for use when on cpu for development, if on GPU, can safely increase
if TPU_MODE:
    BATCH_SIZE=128  #for the size of the dataset this is optimizied for tpuv2-8 node. lower this if getting OOM or tpu crashes


STEPS_PER_EPOCH=int(DATASET_SIZE/BATCH_SIZE)

#Training settings
TRAINING_EPOCHS=100

#adam_learning_rate=0.001  #for reference
BASE_LEARNING_RATE=0.001
LEARNING_RATE_SCHEDUELE=np.zeros(1000)
LEARNING_RATE_SCHEDUELE[:3]=0.5
LEARNING_RATE_SCHEDUELE[3:20]=0.5
LEARNING_RATE_SCHEDUELE[20:40]=0.5
LEARNING_RATE_SCHEDUELE[40:100]=0.5
LEARNING_RATE_SCHEDUELE[100:]=0.3
LEARNING_RATE_SCHEDUELE*=BASE_LEARNING_RATE


RESULTS_ROOT="/tmp"
TENSORBOARD_PATH=RESULTS_ROOT + sep +"tensorboard" #this will get overriden by tpu_config is used
CHECKPOINTS_PATH=RESULTS_ROOT+ sep +"checkpoints" #this will get overriden by tpu_config is used
MODELS_PATH=RESULTS_ROOT + sep + "models"



if TPU_MODE:
    from tpu_training.config_tpu import *
