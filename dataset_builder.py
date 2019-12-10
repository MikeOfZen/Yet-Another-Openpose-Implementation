import tensorflow as tf

import dataset_functions

def get_tfrecord_filenames(path:str,config):
    if config.TPU_MODE:
        from google.cloud import storage
        def _get_tfrecord_filenames(path:str):
            assert path.startswith("gs://")
            print("Retrieving TFrecords from:",path)

            bucket_name=path.lstrip("gs://").split("/")[0]
            prefix = "/".join((path.split("/")[3:]))

            storage_client = storage.Client()
            blobs = storage_client.list_blobs(bucket_name, prefix=prefix) # must have apropriate authenitication to work
            if not blobs:
                raise ValueError("Couldn't find training TFrecord files at:"+bucket_name+"/"+prefix)
            tfrecord_files = ["gs://" + bucket_name + '/' + blob.name for blob in blobs]
            return  tfrecord_files
    else:
        import glob
        def _get_tfrecord_filenames(path:str):
            print("Retrieving TFrecords from:",path)
            tfrecord_files = glob.glob(path)
            tfrecord_files.sort()
            if not tfrecord_files:
                raise ValueError("Couldn't find TFrecord files at:"+path)
            return tfrecord_files
    return _get_tfrecord_filenames(path)

#gcs path gs://datasets_bucket_a/training-002.tfrecords
#glob path TRANSFORMED_TRAIN_ANNOTATIONS_PATH + "-*.tfrecords"

def build_training_ds(tfrecord_filenames:list,labels_placement_function,config)->tf.data.Dataset:
    """Generate training dataset from TFrecord file locations
    :param tfrecord_files should be list of correct TFrecord filename, either local or remote (gcs, with gs:// prefix)"""
    # TFrecord files to raw format
    dataset_transformer = dataset_functions.DatasetTransformer()
    ds = tf.data.TFRecordDataset(tfrecord_filenames)  # numf reads can be put here, but I don't think I/O is the bottleneck

    # raw format to imgs,tensors(coords kpts)
    ds = ds.map(dataset_transformer.read_tfrecord)

    # cache  ,caching is here before decompressing jpgs and label tensors (should be ~9GB) , (full dataset should be ~90, cache later if RAM aviable)
    if config.CACHE: ds = ds.cache()
    if config.SHUFFLE: ds = ds.shuffle(100)

    # Augmentation should be here, to operate on smaller tensors

    # imgs,tensors to label_tensors (46,46,17/38)
    ds = ds.map(dataset_transformer.make_label_tensors)
    # imgs,label_tensors arrange for model outputs
    ds = ds.map(labels_placement_function)

    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.repeat()
    if config.PREFETCH: ds = ds.prefetch(config.PREFETCH)
    return ds

def build_validation_ds(tfrecord_filenames:list,labels_placement_function,config)->tf.data.Dataset:
    """Generate validation dataset from TFrecord file locations
    :param tfrecord_files should be list of correct TFrecord filename, either local or remote (gcs, with gs:// prefix)"""
    # TFrecord files to raw format
    dataset_transformer = dataset_functions.DatasetTransformer()

    ds = tf.data.TFRecordDataset(tfrecord_filenames)  # numf reads can be put here, but I don't think I/O is the bottleneck
    # raw format to imgs,tensors(coords kpts)
    ds = ds.map(dataset_transformer.read_tfrecord)

    if config.CACHE: ds = ds.cache()

    # imgs,tensors to label_tensors (46,46,17/38)
    ds = ds.map(dataset_transformer.make_label_tensors)
    # imgs,label_tensors arrange for model outputs
    ds = ds.map(labels_placement_function)
    ds = ds.batch(config.BATCH_SIZE)
    return ds
