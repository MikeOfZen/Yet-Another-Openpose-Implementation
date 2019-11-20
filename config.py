import os
s=os.sep

#Definitions for COCO 2017 dataset

DATASET_PATH=f".{s}dataset"
IMAGES_PATH=DATASET_PATH+f"{s}trainval2017"
TRAIN_ANNOTATIONS_PATH=DATASET_PATH+f"{s}annotations{s}person_keypoints_train2017.json"
VALIDATION_ANNOTATIONS_PATH=DATASET_PATH+f"{s}annotations{s}person_keypoints_val2017.json"


#will be used as output files
TRANSFORMED_ANNOTATIONS_PATH=f".{s}dataset{s}transformed{s}"
TRANSFORMED_TRAIN_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"person_keypoints_train2017"
TRANSFORMED_VALIDATION_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"person_keypoints_val2017"

#this determines the size images will be resized to, and the size of the labels vreated
IMAGE_WIDTH=368
IMAGE_HEIGHT=368
IMAGE_SIZE=(IMAGE_HEIGHT,IMAGE_WIDTH)

PAF_OUTPUT_FILTERS=38
HEATMAP_FILTERS=17

#taken directly from the annotations JSON file
DATASET_KPTS=['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                'right_knee', 'left_ankle', 'right_ankle']
#shifted by -1 to match keypoints idx
DATASET_JOINTS=[[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
                [ 1, 2], [ 0, 1], [ 0, 2], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6]]



#this is the gaussian spot sie that will be drawn on the training labels
GAUSSIAN_SPOT_SIGMA_SQ=0.02 #used for the size of the gaussian spot for each keypoint
JOINT_WIDTH=0.005  #used for the width of the vector field generated for each joint as a PAF, the unit is relative to image size ie 0..1
