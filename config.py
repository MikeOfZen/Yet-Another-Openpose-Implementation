
DATASET_PATH=r"./dataset"
IMAGES_PATH=DATASET_PATH+"/train2017"
TRAIN_ANNOTATIONS_PATH=DATASET_PATH+"/annotations/person_keypoints_train2017.json"
VALIDATION_ANNOTATIONS_PATH=DATASET_PATH+"/annotations/person_keypoints_val2017.json"


#will be used as output files
TRANSFORMED_ANNOTATIONS_PATH=r"./dataset/transformed/"
TRANSFORMED_TRAIN_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"person_keypoints_train2017"
TRANSFORMED_VALIDATION_ANNOTATIONS_PATH=TRANSFORMED_ANNOTATIONS_PATH+"person_keypoints_val2017"

#this determines the size images will be resized to, and the size of the labels vreated
IMAGE_WIDTH=300
IMAGE_HEIGHT=300
IMAGE_SIZE=(IMAGE_HEIGHT,IMAGE_WIDTH)

#this is the gaussian spot sie that will be drawn on the training labels
GAUSSIAN_SPOT_SIGMA_SQ=0.02 #used for the size of the gaussian spot for each keypoint
JOINT_WIDTH=0.005  #used for the width of the vector field generated for each joint as a PAF, the unit is relative to image size ie 0..1
