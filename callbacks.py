import datetime
from os import sep,makedirs

import tensorflow as tf

from config import LEARNING_RATE_SCHEDUELE,TENSORBOARD_PATH,CHECKPOINTS_PATH,TPU_MODE,RUN_NAME

now=datetime.datetime.now().strftime("%d%a%m%y-%H%M")

checkpoints_path = CHECKPOINTS_PATH+RUN_NAME+now+sep+"Checkpoint-E{epoch:04d}.ckpt"
if not TPU_MODE:
    makedirs(CHECKPOINTS_PATH+sep+now, exist_ok=False)
checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,
                                             save_weights_only=True,
                                             verbose=1)

tensorboard_path=TENSORBOARD_PATH+RUN_NAME+now
if not TPU_MODE:
    makedirs(tensorboard_path, exist_ok=False)
tensorboard_callback=tf.keras.callbacks.TensorBoard(
    log_dir=tensorboard_path
    ,update_freq=30
    ,histogram_freq=1
)


class PrintLR(tf.keras.callbacks.Callback):
    """Callback for printing the LR at the beginging of each epoch"""
    # def __init__(self,model:tf.keras.Model):
    #     """:param model is the trained model"""
    #     self.model=model
    def on_epoch_begin(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch,self.model.optimizer.lr.numpy()))

print_lr_callback=PrintLR()

def get_lr(epoch):
    return LEARNING_RATE_SCHEDUELE[epoch]

learning_rate_scheduler_callback=tf.keras.callbacks.LearningRateScheduler(get_lr)
