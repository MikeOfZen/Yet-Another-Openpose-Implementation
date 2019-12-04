import numpy as np
import tensorflow as tf
import cv2
from os import sep,environ

from pycocotools.coco import COCO
from config import IMAGES_PATH,DATASET_JOINTS ,TRAIN_ANNOTATIONS_PATH, \
    TRANSFORMED_TRAIN_ANNOTATIONS_PATH,VALIDATION_ANNOTATIONS_PATH, TRANSFORMED_VALIDATION_ANNOTATIONS_PATH,LABEL_HEIGHT,LABEL_WIDTH

if environ["DEBUG"]: #useful for debugging imgs with scview
    import matplotlib
    matplotlib.use('module://backend_interagg')
    # matplotlib.pyplot.imshow(total_mask);matplotlib.pyplot.show()

NORMALIZE_SIZE=True

def int64_feature(value):
    if type(value) != list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_example(idd, image_raw, size, kpts, joints,mask):
    kpts=tf.constant(kpts)
    joints = tf.constant(joints)
    mask = tf.constant(mask)
    kpts=tf.io.serialize_tensor(kpts).numpy()
    joints = tf.io.serialize_tensor(joints).numpy()
    mask = tf.io.serialize_tensor(mask).numpy()

    image_raw=image_raw.numpy()

    feature = {
        'id': int64_feature(idd),
        'image_raw': bytes_feature(image_raw),
        'size': int64_feature(size),
        'kpts': bytes_feature(kpts),
        #'person_kpts_bbox': bytes_feature(kpts_bb),
        'joints': bytes_feature(joints),
        'mask': bytes_feature(mask),
        #'mask_bb': bytes_feature(mask_bb),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



def transform_keypts(keypts:list, size:np.ndarray):
    """take the list form, numpyify and form to (x,17,3) tensor where x is the number of persons in image"""
    keypts_np=np.array(keypts, dtype=np.float32)
    keypts_np=keypts_np.reshape((-1,17,3)) #form the list into a correctly shaped tensor
    #normalizing now saves this computation later for every tensor
    #the pixel idx get normalized to 0..1 range so pixel at (100,300) on a (400,600) sized image becomes (0.25,0.5)
    if NORMALIZE_SIZE:
        keypts_np[:,:,0:2]=keypts_np[:,:,0:2]/size
    return keypts_np


def create_all_joints(all_keypts):
    """create a joints tensor from keypoints tensor, according to COCO joints"""
    def create_joints(keypts):
        joints = []
        for kp1_idx, kp2_idx in DATASET_JOINTS:
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

class FileSharder():
    def __init__(self,file_writer,base_filename_format:str,records_per_file:int,verbose:bool=True):
        """Provides a convinient interface to write TFrecord files with auto sharding
        :param base_filename_format the full path to a single file, must include single {} for .format()
        :param FileWriter, is the class to use as a writer, must have .write()"""
        assert base_filename_format.format(0) != base_filename_format

        self._filw_writer=file_writer
        self._base_filename_format=base_filename_format
        self._records_per_file=records_per_file
        self._example_counter=0
        self._file_counter=1
        self._verbose=verbose
        self._start_file()

    def __enter__(self):
        return self

    def _start_file(self):
        self._filename = self._base_filename_format.format(self._file_counter)
        if self._verbose:print("\nWriting file:"+self._filename,flush=True)
        self._writer = self._filw_writer(self._filename)

    def _finish_file(self):
        self._writer.flush()
        self._writer.close()

    def _advance_file(self):
        self._finish_file()
        self._file_counter+=1
        self._example_counter=0
        self._start_file()

    def write(self, item):
        """write a single item, sharded files will be created as needed"""
        self._writer.write(item)
        if self._verbose and not self._example_counter % 100: print(".", end="", flush=True)
        self._example_counter+=1
        if not self._example_counter % self._records_per_file:
            self._advance_file()

    def __exit__(self, *args):
        self._finish_file()

def coco_to_TFrecords(keypoint_annotations_file, transformed_annotations_file, normalize_size=True, records_per_file=1000):
    """This script transforms the COCO 201 keypoint train,val files
    into a format with all keypoints and joints for an image, in a more convinent format,
    where the first axes is the bodypart or joint, the second is the object, and the third are the
    components (x,y,a) for keypoint and (x1,y1,x2,y2,a) for joint.
    The script saves it into matching pickle files.
    Meant to run once.
    :param normalize_size determines whether the pixel coords should be normalized by size to 0..1 range
    """

    print("\nReading "+keypoint_annotations_file)

    coco = COCO(keypoint_annotations_file)

    category=1
    imgIds = coco.getImgIds(catIds=[category])
    #imgIds.sort()
    print("Found %d images" % len(imgIds))

    files_path=transformed_annotations_file+"-{:03}.tfrecords"
    with FileSharder(tf.io.TFRecordWriter,files_path,2000) as writer:
        for img_id in imgIds:
            img_info = coco.loadImgs(img_id)[0]

            size=[int(img_info['height']),int(img_info['width'])]

            annIds = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(annIds)

            persons_kpts=[]
            for annotation in anns:
                if annotation['num_keypoints']>0:
                    kpts=annotation['keypoints']
                    persons_kpts+=kpts

            if not persons_kpts:
                continue #this means that the image has no people with keypoints annotations

            keypoints=transform_keypts(persons_kpts,size)
            tr_joint=create_all_joints(keypoints)
            tr_keypoints= keypoints.transpose((1, 0, 2))  # transpose keypoints for later stages

            total_mask=np.zeros(size,dtype=np.float32)
            for annotation in anns:
                if annotation['num_keypoints']==0: #only mask those without keypoints
                    single_mask=coco.annToMask(annotation)
                    total_mask=np.max([total_mask,single_mask],axis=0)


            total_mask = cv2.resize(total_mask,(LABEL_HEIGHT,LABEL_WIDTH))
            total_mask =(total_mask >0.01).astype(np.int16)

            kernel = np.ones((5, 5), np.uint8)
            total_mask=cv2.dilate(total_mask,kernel)#get more area after downsample
            total_mask=total_mask.astype(np.bool)
            total_mask=np.invert(total_mask) #invert for loss multiplcation later
            total_mask = total_mask.astype(np.float32)

            try:
                img_path=IMAGES_PATH +sep+ img_info['file_name']
                image_raw = tf.io.read_file(img_path)
            except:
                print("Couldnt read file %s" % img_path)
                continue

            example = encode_example(img_id, image_raw, size, tr_keypoints, tr_joint,total_mask)
            writer.write(example)

if __name__ == "__main__":
    coco_to_TFrecords(TRAIN_ANNOTATIONS_PATH, TRANSFORMED_TRAIN_ANNOTATIONS_PATH)
    coco_to_TFrecords(VALIDATION_ANNOTATIONS_PATH, TRANSFORMED_VALIDATION_ANNOTATIONS_PATH)