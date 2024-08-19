from utils.tf_utils import serialize
import tensorflow as tf
from tensorflow.keras.losses import Loss

A_PULL = None
A_PUSH = None

serialize(tf.keras.regularizers.L2)


@serialize
class KoLeoRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, lambda_=1, norm=False):
        self.lambda_ = lambda_
        self.norm = norm

    def __call__(self, x):
        if self.norm:
            x = tf.math.l2_normalize(x, axis=1)
        dist = tf.norm(x[None] - x[:, None], axis=2)
        dist = tf.where(tf.eye(tf.shape(dist)[0]) == 0, dist, tf.fill(tf.shape(dist), tf.reduce_max(dist)))
        min_dist = tf.reduce_min(dist, axis=1)
        out = -self.lambda_ * tf.reduce_mean(tf.math.log(min_dist))
        return out


class KoLeoLoss(Loss):
    def __init__(self, *args, lambda_=1, **kwargs):
        super(KoLeoLoss, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def call(self, y_true, y_pred):
        out = 0.
        for i in range(tf.shape(y_pred)[-1]):
            embd = y_pred[..., i]
            dist = tf.norm(embd[None] - embd[:, None], axis=-1)
            dist = tf.where(tf.eye(tf.shape(dist)[0]) == 0, dist, tf.fill(tf.shape(dist), tf.reduce_max(dist)))
            min_dist = tf.reduce_min(dist, axis=1)
            out += -self.lambda_ * tf.reduce_mean(tf.math.log(min_dist))
        return out


class NullLoss(Loss):
    def __init__(self, *args, name='NullLoss', **kwargs):
        super(NullLoss, self).__init__(*args, name=name, **kwargs)

    def call(self, y_true, y_pred):
        return 0.


class GeneralLossByKey(tf.keras.losses.Loss):
    def __init__(self, key, *args, name='general_loss_{key}', **kwargs):
        name = name.format(key=key) if '{key}' in name else name
        super(GeneralLossByKey, self).__init__(*args, name=name, **kwargs)
        self.key = key

    def loss_func(self, y_true, y_pred):
        raise NotImplementedError()

    def call(self, y_true, y_pred):
        raise self.loss_func(y_true[self.key], y_pred)

