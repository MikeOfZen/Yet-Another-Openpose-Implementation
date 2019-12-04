import tensorflow as tf

class AnalogRecall(tf.keras.metrics.Metric):
    """This metric returns the overlap of the true gaussian 'islands' and the predicted ones"""

    def __init__(self, name='analog_recall', thershold=0.01, **kwargs):
        super(AnalogRecall, self).__init__(name=name, **kwargs)
        self.mean = self.add_weight(name='mean', initializer='zeros')
        self.thershold = thershold

    def update_state(self, y_true, y_pred, **kwargs):
        true_island_sum = tf.reduce_sum(tf.where(y_true > self.thershold, y_true, 0))  # get sum of the true island
        true_island_size = tf.cast(tf.math.count_nonzero(y_true > self.thershold), dtype=tf.float32)  # get size of the island
        mean_island_true = true_island_sum / true_island_size  # average island value

        err = y_true - y_pred  # get all error
        recall_err = tf.where(err > 0, err, 0)  # get only recall error, the parts where prediction is missing
        recall_err_sum = tf.reduce_sum(recall_err)
        err_island_size = tf.cast(tf.math.count_nonzero(y_true > self.thershold), tf.float32)  # get size of the islands above thershold

        mean_island_recall_err = recall_err_sum / err_island_size  # mean of the error

        value = 1 - mean_island_recall_err / mean_island_true  # the 1- converts it to recall accuracy onstead of err
        self.mean.assign_add(value)

    def result(self):
        return self.mean

