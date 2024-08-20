from utils.model.losses import GeneralLossByKey
import tensorflow as tf


class CrossPathwayTemporalContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, *args, a=None, contrast_t=1, start_t=0, temperature=10, **kwargs):
        super(CrossPathwayTemporalContrastiveLoss, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.a = a
        self.contrast_t = contrast_t
        self.start_t = start_t

    def call(self, y_true, y_pred):
        # y_pred shape (B, P, DIM, T)
        n = tf.shape(y_pred)[1]
        loss = 0.

        for i in range(n):
            for j in range(i+1, n):
                if self.a is not None and not self.a[i][j] and not self.a[j][i]:
                    continue
                anchor_i = y_pred[:, i, :, self.start_t + self.contrast_t:]      # (B, DIM, T-cont_t-start_t)
                anchor_j = y_pred[:, j, :, self.start_t + self.contrast_t:]    # (B, DIM, T-cont_t-start_t)
                pos_temped_sqr_dist = (tf.linalg.norm(anchor_i - anchor_j, axis=-2) ** 2) / self.temperature  # (B, T-cont_t-start_t)

                if self.a is None or (self.a is not None and self.a[i][j]):
                    negative_j = y_pred[:, j, :, self.start_t:-self.contrast_t]  # (B, DIM, T-cont_t-start_t)
                    neg_temped_sqr_dist_i = (tf.linalg.norm(anchor_i - negative_j, axis=-2) ** 2) / self.temperature
                    zi = tf.exp(-pos_temped_sqr_dist) + tf.exp(-neg_temped_sqr_dist_i)
                    minus_log_likelihood_i = pos_temped_sqr_dist + tf.math.log(zi)   # (B, T-cont_t-start_t)
                    loss = loss + tf.reduce_mean(minus_log_likelihood_i)

                if self.a is None or (self.a is not None and self.a[j][i]):
                    negative_i = y_pred[:, i, :, self.start_t:-self.contrast_t]  # (B, DIM, T-cont_t-start_t)
                    neg_temped_sqr_dist_j = (tf.linalg.norm(anchor_j - negative_i, axis=-2) ** 2) / self.temperature
                    zj = tf.exp(-pos_temped_sqr_dist) + tf.exp(-neg_temped_sqr_dist_j)
                    minus_log_likelihood_j = pos_temped_sqr_dist + tf.math.log(zj) # (B, T-cont_t-start_t)
                    loss = loss + tf.reduce_mean(minus_log_likelihood_j)

        return loss / (tf.cast(n ** 2, dtype=loss.dtype) if self.a is None else tf.reduce_sum(tf.cast(self.a, dtype=loss.dtype)))


class SparseCategoricalCrossEntropyByKey(GeneralLossByKey):
    def __init__(self, *args, from_logits=True, name='sparse_categorical_ce_{key}', **kwargs):
        super(SparseCategoricalCrossEntropyByKey, self).__init__(*args, name=name, **kwargs)
        self.from_logits = from_logits

    def loss_func(self, y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)


class MeanAbsoluteErrorByKeyLoss(GeneralLossByKey):
    def __init__(self, *args, name='absolute_error_{key}', **kwargs):
        super(MeanAbsoluteErrorByKeyLoss, self).__init__(*args, name=name, **kwargs)

    def loss_func(self, y_true, y_pred):
        return tf.keras.losses.mean_absolute_loss(y_true, y_pred)