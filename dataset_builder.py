import tensorflow as tf
import dataset_functions
from config import *

if TPU_MODE:
    from google.cloud import storage
    from tpu_training.config_tpu import *
    print("Retrieving TFrecords in TPU_mode")

    gs_prefix = "gs://"
    def get_tfrecord_filenames():
        train_prefix = TRANSFORMED_TRAIN_ANNOTATIONS_PATH.split(os.sep)[-1]
        val_prefix = TRANSFORMED_VALIDATION_ANNOTATIONS_PATH.split(os.sep)[-1]

        storage_client = storage.Client()
        # must have apropriate authenitication to work
        train_blobs = storage_client.list_blobs(GCS_TFRECORDS_BUCKETNAME, prefix=train_prefix)
        val_blobs = storage_client.list_blobs(GCS_TFRECORDS_BUCKETNAME, prefix=val_prefix)

        if not train_blobs:
            raise ValueError("Couldn't find training TFrecord files at:"+GCS_TFRECORDS_BUCKETNAME+"/"+train_prefix)
        if not val_blobs:
            raise ValueError("Couldn't find validation TFrecord files at:"+GCS_TFRECORDS_BUCKETNAME+"/"+train_prefix)

        tfrecord_files_train = [gs_prefix + GCS_TFRECORDS_BUCKETNAME + '/' + blob.name for blob in train_blobs]
        tfrecord_files_val = [gs_prefix + GCS_TFRECORDS_BUCKETNAME + '/' + blob.name for blob in val_blobs]

        return  tfrecord_files_train,tfrecord_files_val
else:
    import glob
    def get_tfrecord_filenames():
        print("Retrieving TFrecords in local mode")
        tfrecord_files_train = glob.glob(TRANSFORMED_VALIDATION_ANNOTATIONS_PATH + "-*.tfrecords")
        tfrecord_files_train.sort()
        tfrecord_files_val = glob.glob(TRANSFORMED_VALIDATION_ANNOTATIONS_PATH+ "-*.tfrecords")
        tfrecord_files_val.sort()

        if not tfrecord_files_val:
            raise ValueError("Couldn't find training TFrecord files at:"+TRANSFORMED_VALIDATION_ANNOTATIONS_PATH)
        if not tfrecord_files_train:
            raise ValueError("Couldn't find validation TFrecord files at:"+TRANSFORMED_VALIDATION_ANNOTATIONS_PATH)
        return tfrecord_files_train,tfrecord_files_val

TF_parser = dataset_functions.TFrecordParser()
def build_validation_ds(tfrecord_filenames:list)->tf.data.Dataset:
    """Generate validation dataset from TFrecord file locations
    :param tfrecord_files should be list of correct TFrecord filename, either local or remote (gcs, with gs:// prefix)"""
    # TFrecord files to raw format
    ds = tf.data.TFRecordDataset(tfrecord_filenames)  # numf reads can be put here, but I don't think I/O is the bottleneck
    # raw format to imgs,tensors(coords kpts)
    ds = ds.map(TF_parser.read_tfrecord)

    if CACHE: ds = ds.cache()

    # imgs,tensors to label_tensors (46,46,17/38)
    ds = ds.map(dataset_functions.make_label_tensors)
    # imgs,label_tensors arrange for model outputs
    ds = ds.map(dataset_functions.place_training_labels)
    ds = ds.batch(BATCH_SIZE)
    return ds

def build_training_ds(tfrecord_filenames:list)->tf.data.Dataset:
    """Generate training dataset from TFrecord file locations
    :param tfrecord_files should be list of correct TFrecord filename, either local or remote (gcs, with gs:// prefix)"""
    # TFrecord files to raw format
    ds = tf.data.TFRecordDataset(tfrecord_filenames)  # numf reads can be put here, but I don't think I/O is the bottleneck

    # raw format to imgs,tensors(coords kpts)
    ds = ds.map(TF_parser.read_tfrecord)

    # cache  ,caching is here before decompressing jpgs and label tensors (should be ~9GB) , (full dataset should be ~90, cache later if RAM aviable)
    if CACHE: ds = ds.cache()
    if SHUFFLE: ds = ds.shuffle(100)

    # Augmentation should be here, to operate on smaller tensors

    # imgs,tensors to label_tensors (46,46,17/38)
    ds = ds.map(dataset_functions.make_label_tensors)
    # imgs,label_tensors arrange for model outputs
    ds = ds.map(dataset_functions.place_training_labels)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.repeat()
    if PREFETCH: ds = ds.prefetch(PREFETCH)
    return ds
