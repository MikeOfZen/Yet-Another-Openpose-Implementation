import json
import collections
import os
import objectpath
import numpy as np
import pickle
import tensorflow as tf

import config as c
import coco_helper as ch


def int64_feature(value):
    if type(value) != list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_example(idd, image_raw, size, kpts, joints):
    kpts=tf.constant(kpts)
    joints = tf.constant(joints)
    kpts=tf.io.serialize_tensor(kpts).numpy()
    joints = tf.io.serialize_tensor(joints).numpy()

    # kpts = kpts.tobytes() #numpy serialization
    # joints = joints.tobytes()  #numpy serialization
    image_raw=image_raw.numpy()

    feature = {
        'id': int64_feature(idd),
        'image_raw': bytes_feature(image_raw),
        'size': int64_feature(size),
        'kpts': bytes_feature(kpts),
        'joints': bytes_feature(joints)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# def encode_example(idd, image_raw, size, kpts, joints):
#     features = {
#         'id': idd,
#         'image_raw': image_raw,
#         'size': size,
#         'kpts': kpts,
#         'joints': joints
#     }
#     return pickle.dumps(features) #serialized


def COCOjson_to_pickle(keypoint_annotations_file, transformed_annotations_file, normalize_size=True,records_per_file=1000):
    """This script transforms the COCO 201 keypoint train,val files
    into a format with all keypoints and joints for an image, in a more convinent format,
    where the first axes is the bodypart or joint, the second is the object, and the third are the
    components (x,y,a) for keypoint and (x1,y1,x2,y2,a) for joint.
    The script saves it into matching pickle files.
    Meant to run once.
    :param normalize_size determines whether the pixel coords should be normalized by size to 0..1 range
    """

    print(f"Reading {keypoint_annotations_file}")

    with open(keypoint_annotations_file, 'r') as f:
        json_file = json.load(f)
    json_tree = objectpath.Tree(json_file)
    del json_file

    print(f"Getting all images keypoints")

    #collect id and keypoints, since the annotatios file contains two types of annotatios
    # which look the same but serve different purpose, filter by "num_keypoints is not 0"
    #and also collect all keypoints for a single image
    img_keypts = collections.defaultdict(list)
    for x in json_tree.execute("$.annotations[@.num_keypoints is not 0]"):
        img_keypts[x['image_id']].append(x['keypoints'])

    print(f"Found {len(img_keypts)} images")

    #for every image get it's size
    image_sizes = {}
    for x in json_tree.execute("$.images"):
        image_sizes[x['id']] = [x['width'], x["height"]]

    #take the list form, numpyify and form to (x,17,3) tensor where x is the number of persons in image
    def transform_keypts(keypts_list,size):
        keypts_np=np.array(keypts_list, dtype=np.float32)
        keypts_np=keypts_np.reshape((-1,17,3)) #form the list into a correctly shaped tensor
        #normalizing now saves this computation later for every tensor
        #the pixel idx get normalized to 0..1 range so pixel at (100,300) on a (400,600) sized image becomes (0.25,0.5)
        if normalize_size:
            keypts_np[:,:,0:2]=keypts_np[:,:,0:2]/size
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


    print(f"Writing combined dataset")

    #create a combined dict of size,transposed keypoints and transposed joints
    combined=collections.OrderedDict()
    for i,img_id in enumerate(sorted(img_keypts)):
        if not i % records_per_file:
            try:
                writer.flush()
                writer.close()
                print("|", end="")
            except NameError:
                pass
            filename = transformed_annotations_file+f"-{int(i / records_per_file)+1:03}.tfrecords"
            writer = tf.io.TFRecordWriter(filename)
            print(f"\nWriting file:{filename}",flush=True)
        if not i % 100: print(".", end="",flush=True)

        size=image_sizes[img_id]
        keypoints_list=img_keypts[img_id]
        keypoints=transform_keypts(keypoints_list,size)
        tr_keypoints= keypoints.transpose((1, 0, 2))  # transpose keypoints

        tr_joint=create_all_joints(keypoints)
        try:
            image_raw=tf.io.read_file(ch.id_to_filename(img_id))
            # with open(ch.id_to_filename(img_id),'rb') as f:
            #     image_raw = f.read()
        except:
            print(f"Couldnt read file {ch.id_to_filename(img_id)}")
            i-=1
            continue

        example = encode_example(img_id, image_raw, size, tr_keypoints, tr_joint)
        writer.write(example)
    writer.flush()
    writer.close()



if __name__ == "__main__":
    COCOjson_to_pickle(c.TRAIN_ANNOTATIONS_PATH, c.TRANSFORMED_TRAIN_ANNOTATIONS_PATH)
    COCOjson_to_pickle(c.VALIDATION_ANNOTATIONS_PATH, c.TRANSFORMED_VALIDATION_ANNOTATIONS_PATH)