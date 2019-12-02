import datetime
from os import sep,mkdir

import tensorflow as tf

from config import LEARNING_RATE_SCHEDUELE,TENSORBOARD_PATH,CHECKPOINTS_PATH

now=datetime.datetime.now().strftime("%d%a%m%y-%H%M")

checkpoints_path = CHECKPOINTS_PATH+sep+now+sep+"Checkpoint-E{epoch:04d}.ckpt"
mkdir(CHECKPOINTS_PATH+sep+now)
checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,
                                             save_weights_only=True,
                                             verbose=1)

tensorboard_path=TENSORBOARD_PATH+sep+now
mkdir(tensorboard_path)
tensorboard_callback=tf.keras.callbacks.TensorBoard(
    log_dir=tensorboard_path
    #,update_freq=5000 #to update sooner than every epoch
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
