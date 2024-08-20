import tensorflow as tf


class MeanConfidence(tf.keras.metrics.Metric):
    def __init__(self, name='mean_confidence', alpha=0.75, **kwargs):
        super(MeanConfidence, self).__init__(name=name, **kwargs)
        self.mean_conf = self.add_variable(
            shape=(),
            initializer='zeros',
            name='mean_conf'
        )
        self.alpha = alpha

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_conf.assign(self.alpha * self.mean_conf + (1-self.alpha) * tf.reduce_mean(tf.sigmoid(y_pred[:, -1])))

    def result(self):
        return self.mean_conf


class SparseCategoricalAccuracyByKey(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, key, *args, name='sparse_categorical_accuracy_{key}', **kwargs):
        name = name.format(key=key) if '{key}' in name else name
        super(SparseCategoricalAccuracyByKey, self).__init__(*args, name=name, **kwargs)
        self.key = key

    def update_state(self, y_true, y_pred, **kwargs):
        return super(SparseCategoricalAccuracyByKey, self).update_state(y_true[self.key], y_pred, **kwargs)


class MeanAbsoluteErrorByKeyMetric(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, key, *args, name='mean_absolute_err_{key}', **kwargs):
        name = name.format(key=key) if '{key}' in name else name
        super(MeanAbsoluteErrorByKeyMetric, self).__init__(*args, name=name, **kwargs)
        self.key = key

    def update_state(self, y_true, y_pred, **kwargs):
        return super(MeanAbsoluteErrorByKeyMetric, self).update_state(y_true[self.key], y_pred, **kwargs)


