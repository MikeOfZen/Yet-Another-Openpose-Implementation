import tensorflow as tf


class AnalogRecall(tf.keras.metrics.Metric):
    """This metric returns the overlap of the true gaussian 'islands' and the predicted ones"""

    def __init__(self, name='analog_recall', threshold=0.01, **kwargs):
        super(AnalogRecall, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, **kwargs):
        a_true = abs(y_true)
        a_pred = abs(y_pred, )
        boundary = a_true > self.threshold
        bounded_true = tf.where(boundary, a_true, 0)
        bounded_pred = tf.where(boundary, a_pred, 0)

        err = bounded_true - bounded_pred
        recall_err = tf.where(err > 0, err, 0)

        value = 1.0 - tf.reduce_sum(recall_err) / tf.reduce_sum(bounded_true)

        self.sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return self.sum / self.count


class MeanAbsolute(tf.keras.metrics.Metric):
    """This metric returns the sum of the absolute of the predictions"""

    def __init__(self, name='MeanAbsolute', **kwargs):
        super(MeanAbsolute, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        value = tf.reduce_mean(abs(y_pred))
        self.sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return self.sum / self.count


class MeanAbsoluteRatio(tf.keras.metrics.Metric):
    """This metric returns the ratio of the mean absolute of the prediction vs truth"""

    def __init__(self, name='MeanAbsoluteRatio', **kwargs):
        super(MeanAbsoluteRatio, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        value = tf.reduce_mean(abs(y_pred)) / tf.reduce_mean(abs(y_true))
        self.sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return self.sum / self.count


class MaskedMeanSquaredError(tf.keras.losses.MeanSquaredError):
    def __call__(self, true, pred):
        mask = true[..., -1:]  # split the concatenated input
        true = true[..., :-1]
        # empty_mask = pred[..., -1:]  # coming from the model, required to silence keras about shape mismatch
        pred = pred[..., :-1]

        return super(MaskedMeanSquaredError, self).__call__(true, pred, mask)


class MaskedMeanAbsoluteError(tf.keras.losses.MeanAbsoluteError):
    def __call__(self, true, pred):
        mask = true[..., -1:]  # split the concatenated input
        true = true[..., :-1]
        # empty_mask = pred[..., -1:]  # coming from the model, required to silence keras about shape mismatch
        pred = pred[..., :-1]

        return super(MaskedMeanAbsoluteError, self).__call__(true, pred, mask)
