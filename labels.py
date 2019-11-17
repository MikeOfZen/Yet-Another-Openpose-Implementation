import json
import tensorflow as tf
import pickle

class COCO_dataset():
    def __init__(self,transformed_annotation_file):
        with open(transformed_annotation_file, 'rb') as f:
            self.combined_dict = pickle.load(f)

    def get_dataset(self):
        ids = []
        sizes = []
        keypoints = []
        joints = []
        for idd, l in self.combined_dict.items():
            ids.append(idd)
            sizes.append(l[0])
            keypoints.append(l[1])
            joints.append(l[2])

        rt_keypoints = tf.ragged.constant(keypoints)
        rt_joints=tf.ragged.constant(joints)
        return tf.data.Dataset.from_tensor_slices((ids, sizes, rt_keypoints, rt_joints))

