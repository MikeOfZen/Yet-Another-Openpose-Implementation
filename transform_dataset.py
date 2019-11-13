import json
import os
import objectpath
import numpy as np

import config as c

def script(keypoint_annotations_file):
    """This script transforms the COCO 201 keypoint train,val files
    into a format with all keypoints and joints for an image, in a more convinent format,
    where the first axes is the bodypart or joint, the second is the object, and the third are the
    components (x,y,a) for keypoint and (x1,y1,x2,y2,a) for joint.
    The script saves it into matching Json files
    """









if __name__ == "__main__":
    script(c.TRAIN_ANNOTATIONS_PATH)
    script(c.TEST_ANNOTATIONS_PATH)