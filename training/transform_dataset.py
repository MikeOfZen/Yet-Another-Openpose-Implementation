#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import cv2
from os import environ

from pycocotools.coco import COCO

if "DEBUG" in environ:  # useful for debugging imgs with sciview
    import matplotlib

    matplotlib.use('module://backend_interagg')
    # matplotlib.pyplot.imshow(total_mask);matplotlib.pyplot.show()


def int64_feature(value):
    if type(value) != list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_example(idd, image_raw, size, kpts, joints, mask):
    kpts = tf.constant(kpts)
    joints = tf.constant(joints)
    mask = tf.constant(mask)
    kpts = tf.io.serialize_tensor(kpts).numpy()
    joints = tf.io.serialize_tensor(joints).numpy()
    mask = tf.io.serialize_tensor(mask).numpy()

    image_raw = image_raw.numpy()

    feature = {
            'id'       : int64_feature(idd),
            'image_raw': bytes_feature(image_raw),
            'size'     : int64_feature(size),
            'kpts'     : bytes_feature(kpts),
            # 'person_kpts_bbox': bytes_feature(kpts_bb),
            'joints'   : bytes_feature(joints),
            'mask'     : bytes_feature(mask),
            # 'mask_bb': bytes_feature(mask_bb),
            }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def middle_kpt(kpt1, kpt2):
    """Makes a middle keypoint from 2, if one of them is 0, also returns 0"""
    if kpt1[2] == 0 or kpt2[2] == 0:
        return 0, 0, 0
    else:
        return [
                (kpt1[0] + kpt2[0]) / 2,
                (kpt1[1] + kpt2[1]) / 2,
                min(kpt1[2], kpt2[2])
                ]


def reshape_kpts(keypoints: list, config) -> np.ndarray:
    """reshapes keypoints list into numpy array
    :param keypoints list of coco keypoints of  ...kpt x,kpt y,kpt visibility...
    :param config the effective config
    :returns np.ndarray of shape (DS_NUM_KEYPOINTS,3)"""
    keypts_np = np.array(keypoints, dtype=np.float32)
    keypts_np = keypts_np.reshape((config.DS_NUM_KEYPOINTS, 3))
    return keypts_np


def map_new_kpts(keypoints: np.ndarray, config) -> list:
    """Map from dataset keypoints to own definition of keypoints, defined in KEYPOINTS_DEF.
     for example dataset has no neck keypoint,this map it by averaging left and right shoulders
     otherwise, it rearranges kpts in a more sensible order"""
    new_keypts = []
    for kpt_name, kpt_def in config.KEYPOINTS_DEF.items():
        ds_idxs = kpt_def["ds_idxs"]
        assert type(ds_idxs) is int or (type(ds_idxs) is tuple and len(ds_idxs) == 2)

        if type(ds_idxs) is tuple:
            first_kpt = keypoints[ds_idxs[0]]
            second_kpt = keypoints[ds_idxs[1]]
            new_kpt = np.array(middle_kpt(first_kpt, second_kpt), dtype=np.float32)
        else:
            new_kpt = keypoints[ds_idxs]
        new_keypts.append(new_kpt)
    return new_keypts


def transform_keypts(keypoints, size: np.ndarray):
    """take the list form, numpyifies and forms to (number of persons,DS_NUM_KEYPOINTS,3) tensor,
    also switches coords to match the rest of the system ie Y,X instead of X,Y"""

    # keypts_np=np.array(keypts, dtype=np.float32)
    # keypts_np=keypts_np.reshape((-1,DS_NUM_KEYPOINTS,3)) #form the list into a correctly shaped tensor

    # critical, the incoming coords are in X,Y order, but everything else is in Y,X order!
    X = np.array(keypoints[..., 0], dtype=np.float32)
    Y = np.array(keypoints[..., 1], dtype=np.float32)
    keypoints[..., 0] = Y
    keypoints[..., 1] = X

    # normalizing now saves this computation later for every tensor
    # the pixel idx get normalized to 0..1 range so pixel at (100,300) on a (400,600) sized image becomes (0.25,0.5)
    keypoints[:, :, 0:2] = keypoints[:, :, 0:2] / size
    return keypoints


def create_all_joints(all_keypts, config):
    """create a joints tensor from keypoints tensor, according to COCO joints
    :param config: effective config dict, must include JOINTS_DEF
    :param all_keypts - tensor of shape (number of persons,number of kpts(DS_NUM_KEYPOINTS),3)
    :return tensor of shape (number of persons,number of joints(19),5)"""

    def create_joints(keypts):
        joints = []
        for joint_name, joint_def in config.JOINTS_DEF.items():
            kp1_name, kp2_name = joint_def["kpts"]
            kp1_idx = config.KEYPOINTS_DEF[kp1_name]["idx"]
            kp2_idx = config.KEYPOINTS_DEF[kp2_name]["idx"]
            kp1 = keypts[kp1_idx]
            kp2 = keypts[kp2_idx]
            if kp1[2] == 0 or kp2[2] == 0:
                # if either of the keypoints is missing, the joint is zero
                new_joint = (0, 0, 0, 0, 0)
                joints.append(new_joint)
                continue
            # create new joint from both keypoint coords, with the visibility being the minimum of either keypoint
            new_joint = (*kp1[0:2], *kp2[0:2], min(kp1[2], kp2[2]))
            joints.append(new_joint)
        return np.array(joints, dtype=np.float32)

    all_joints = [create_joints(x) for x in all_keypts]  # for each person

    # numpify result transpose joints
    return np.array(all_joints, dtype=np.float32).transpose((1, 0, 2))


class FileSharder:
    def __init__(self, file_writer, base_filename_format: str, records_per_file: int, verbose: bool = True):
        """Provides a convenient interface to write TFrecord files with auto sharding
        :param base_filename_format the full path to a single file, must include single {} for .format()
        :param file_writer, is the class to use as a writer, must have .write()"""
        assert base_filename_format.format(0) != base_filename_format

        self._file_writer = file_writer
        self._base_filename_format = base_filename_format
        self._records_per_file = records_per_file
        self._example_counter = 0
        self._file_counter = 1
        self._verbose = verbose
        self._start_file()

    def __enter__(self):
        return self

    def _start_file(self):
        self._filename = self._base_filename_format.format(self._file_counter)
        if self._verbose: print("\nWriting file:" + self._filename, flush=True)
        self._writer = self._file_writer(self._filename)

    def _finish_file(self):
        self._writer.flush()
        self._writer.close()

    def _advance_file(self):
        self._finish_file()
        self._file_counter += 1
        self._example_counter = 0
        self._start_file()

    def write(self, item):
        """write a single item, sharded files will be created as needed"""
        self._writer.write(item)
        if self._verbose and not self._example_counter % 100: print(".", end="", flush=True)
        self._example_counter += 1
        if not self._example_counter % self._records_per_file:
            self._advance_file()

    def __exit__(self, *args):
        self._finish_file()


def coco_to_TFrecords(keypoint_annotations_file, transformed_annotations_file, config):
    """This script transforms the COCO 2017 keypoint train,val files
    into a format with all keypoints and joints for an image, in a more convenient format,
    where the first axes is the body part or joint, the second is the object, and the third are the
    components (x,y,a) for keypoint and (x1,y1,x2,y2,a) for joint.
    The script saves it into matching pickle files.
    Meant to run once.
    normalizes size the pixel coords to be normalized by size to 0..1 range
    """

    print("\nReading " + keypoint_annotations_file)

    coco = COCO(keypoint_annotations_file)

    category = 1
    imgIds = coco.getImgIds(catIds=[category])
    imgIds.sort()
    print("Found %d images" % len(imgIds))

    files_path = transformed_annotations_file + "-{:03}.tfrecords"
    with FileSharder(tf.io.TFRecordWriter, files_path, config.IMAGES_PER_TFRECORD) as writer:
        for img_id in imgIds:
            img_info = coco.loadImgs(img_id)[0]

            size = [img_info['height'], img_info['width']]

            annIds = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(annIds)

            persons_kpts = []
            for annotation in anns:
                if annotation['num_keypoints'] > 0:
                    kpts = annotation['keypoints']

                    # map to new kpts
                    kpts = reshape_kpts(kpts, config)
                    kpts = map_new_kpts(kpts, config)

                    persons_kpts.append(kpts)

            if not persons_kpts:
                continue  # this means that the image has no people with keypoints annotations

            persons_kpts = np.array(persons_kpts, dtype=np.float32)  # convert from list to array

            keypoints = transform_keypts(persons_kpts, np.array(size, dtype=np.int))
            tr_joint = create_all_joints(keypoints, config)
            tr_keypoints = keypoints.transpose((1, 0, 2))  # transpose keypoints for later stages

            total_mask = np.zeros(size, dtype=np.float32)
            for annotation in anns:
                if annotation['num_keypoints'] == 0:  # only mask those without keypoints
                    single_mask = coco.annToMask(annotation)
                    total_mask = np.max([total_mask, single_mask], axis=0)

            total_mask = cv2.resize(total_mask, (config.LABEL_HEIGHT, config.LABEL_WIDTH))
            total_mask = (total_mask > 0.01).astype(np.int16)

            kernel = np.ones((5, 5), np.uint8)
            total_mask = cv2.dilate(total_mask, kernel)  # get more area after downsample
            total_mask = total_mask.astype(np.bool)
            total_mask = np.invert(total_mask)  # invert for loss multiplication later
            total_mask = total_mask.astype(np.float32)

            try:
                img_path = config.IMAGES_PATH + "/" + img_info['file_name']
                image_raw = tf.io.read_file(img_path)
            except:
                print("Couldn't read file %s" % img_path)
                continue

            example = encode_example(img_id, image_raw, size, tr_keypoints, tr_joint, total_mask)
            writer.write(example)


if __name__ == "__main__":
    from configs import default_config as cfg, local_storage_config as storage_cfg

    cfg.__dict__.update(storage_cfg.__dict__)

    coco_to_TFrecords(cfg.TRAIN_ANNS, cfg.TRAIN_TFRECORDS, cfg)
    coco_to_TFrecords(cfg.VALID_ANNS, cfg.VALID_TFRECORDS, cfg)
