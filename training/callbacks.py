from training import utils
from os import makedirs

import tensorflow as tf


def make_checkpoint_callback(config, sig, freq):
    checkpoints_path = config.CHECKPOINTS_PATH + "/" + config.RUN_NAME + sig + "/-E{epoch:04d}.ckpt"
    if not config.TPU_MODE:
        makedirs(config.CHECKPOINTS_PATH + "/" + config.RUN_NAME + utils.now(), exist_ok=True)

    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path
                                              , save_weights_only=True
                                              , save_freq=freq
                                              , verbose=1)


def make_tensorboard_callback(config, sig, hist_freq=0):
    tensorboard_path = config.TENSORBOARD_PATH + "/" + config.RUN_NAME + sig
    if not config.TPU_MODE:
        makedirs(tensorboard_path, exist_ok=False)
    return tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path
                                          # , update_freq=config.TENSORBOARD_FREQ
                                          , histogram_freq=hist_freq
                                          )


class PrintLR(tf.keras.callbacks.Callback):
    """Callback for printing the LR at the beginning of each epoch"""

    def on_epoch_begin(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch, self.model.optimizer.lr.numpy()))


def make_LRscheduler_callback(learning_rate_schedule):
    def get_lr(epoch):
        return learning_rate_schedule[epoch]

    return tf.keras.callbacks.LearningRateScheduler(get_lr)
