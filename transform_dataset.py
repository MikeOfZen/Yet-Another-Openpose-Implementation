import json
import collections
import os
import objectpath
import numpy as np
import pickle

import config as c
import coco_helper as ch

def transform_dataset(keypoint_annotations_file,transformed_annotations_file,verbose=True):
    """This script transforms the COCO 201 keypoint train,val files
    into a format with all keypoints and joints for an image, in a more convinent format,
    where the first axes is the bodypart or joint, the second is the object, and the third are the
    components (x,y,a) for keypoint and (x1,y1,x2,y2,a) for joint.
    The script saves it into matching pickle files

    """

    if verbose:
        print(f"Reading {keypoint_annotations_file}")

    with open(keypoint_annotations_file, 'r') as f:
        json_file = json.load(f)
    json_tree = objectpath.Tree(json_file)
    del json_file

    if verbose:
        print(f"Getting all images keypoints")

    #collect id and keypoints, since the annotatios file contains two types of annotatios
    # which look the same but serve different purpose, filter by "num_keypoints is not 0"
    #and also collect all keypoints for a single image
    img_keypts = collections.defaultdict(list)
    for x in json_tree.execute("$.annotations[@.num_keypoints is not 0]"):
        img_keypts[x['image_id']].append(x['keypoints'])

    if verbose:
        print(f"Found {len(img_keypts)} images")

    #for every image get it's size
    image_sizes = {}
    for x in json_tree.execute("$.images"):
        image_sizes[x['id']] = (x['width'], x["height"])

    #take the list form, numpyify and form to (x,17,3) tensor where x is the number of persons in image
    def transform_keypts(keypts_list):
        keypts_np=np.array(keypts_list, dtype=np.float32)
        keypts_np=keypts_np.reshape((-1,17,3)) #form the list into a correctly shaped tensor
        return keypts_np

    #create a joints tensor from keypoints tensor, according to COCO joints
    def create_all_joints(all_keypts):
        def create_joints(keypts):
            joints = []
            for kp1_idx, kp2_idx in ch.coco_joints:
                kp1 = keypts[kp1_idx]
                kp2 = keypts[kp2_idx]
                if kp1[2] == 0 or kp2[2] == 0:
                    #if either of the keypoints is missing, the joint is zero
                    new_joint = (0, 0, 0, 0, 0)
                    joints.append(new_joint)
                    continue
                #create new joint from both keypoint coords, with the visibility being the minimum of either keypoint
                new_joint = (*kp1[0:2], *kp2[0:2], min(kp1[2], kp2[2]))
                joints.append(new_joint)
            return np.array(joints, dtype=np.float32)
        all_joints = [create_joints(x) for x in all_keypts]

        #numpify result transpose joints
        return np.array(all_joints, dtype=np.float32).transpose((1, 0, 2))

    if verbose:
        print(f"Creating combined dict")

    #create a combined dict of size,transposed keypoints and transposed joints
    combined=collections.OrderedDict()
    for i,img_id in enumerate(sorted(img_keypts)):
        size=image_sizes[img_id]
        keypoints_list=img_keypts[img_id]
        keypoints=transform_keypts(keypoints_list)
        tr_keypoints= keypoints.transpose((1, 0, 2))  # transpose keypoints

        tr_joint=create_all_joints(keypoints)

        combined[img_id]=[size,tr_keypoints,tr_joint]
        if verbose and not i%100:
            print(".",end="", flush=True)


    file_path=transformed_annotations_file
    #write out to json file the transformed dataset

    if verbose:
        print(f"\nWriting out JSON file to {file_path}")

    with open(file_path, 'w') as fp:
        pickle.dump(combined, fp)

if __name__ == "__main__":
    transform_dataset(c.TRAIN_ANNOTATIONS_PATH,c.TRANSFORMED_TRAIN_ANNOTATIONS_PATH)
    transform_dataset(c.VALIDATION_ANNOTATIONS_PATH,c.TRANSFORMED_VALIDATION_ANNOTATIONS_PATH)