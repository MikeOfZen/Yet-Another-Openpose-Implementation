import config as c

import os

def id_to_filename(id):
    return os.path.join(c.IMAGES_PATH,f"{id:012}.jpg")

#taken directly from the annotations JSON file
coco_keypoints=['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                'right_knee', 'left_ankle', 'right_ankle']
#shifted by -1 to match keypoints idx
coco_joints=[[15,13], [13,11], [16,14], [14,12], [11,12], [ 5,11], [ 6,12], [ 5, 6], [ 5, 7], [ 6, 8], [ 7, 9], [ 8,10],
            [ 1, 2], [ 0, 1], [ 0, 2], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6]]

