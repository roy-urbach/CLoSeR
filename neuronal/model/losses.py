from utils.model.losses import GeneralLossByKey
import tensorflow as tf


class CrossPathwayTemporalContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, *args, a=None, contrast_t=1, start_t=0, temperature=10, cosine=False, stable=True,
                 eps=1e-3, triplet=None, **kwargs):
        super(CrossPathwayTemporalContrastiveLoss, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.a = a
        self.contrast_t = contrast_t
        self.start_t = start_t
        self.cosine = cosine
        self.eps = eps
        self.stable = stable
        self.triplet = triplet

    def minus_log_likelihood_from_log_sim(self, pos, neg):
        if self.stable:
            max_ = tf.maximum(pos, neg)
            neg = neg - max_
            pos = pos - max_
        z = tf.maximum(tf.exp(pos), self.eps) + tf.maximum(tf.exp(neg), self.eps)
        minus_log_likelihood = -pos + tf.math.log(z)  # (B, T-cont_t-start_t)
        return minus_log_likelihood

    def loss_func(self, pos, neg):
        if self.triplet is not None:
            loss = tf.maximum(pos - neg + self.triplet, 0)
        else:
            loss = self.minus_log_likelihood_from_log_sim(pos, neg)
        return loss

    def val_func(self, anchor, other):
        if self.triplet is not None:
            val = tf.linalg.norm(anchor - other, axis=-1)
        elif self.cosine:
            val = tf.einsum('...i,...i->...', anchor, other) / self.temperature
        else:
            val = -(tf.linalg.norm(a - b, axis=-1) ** 2)
        return val

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)
        n = tf.shape(y_pred)[-1]
        loss = 0.
        if self.cosine:
            y_pred = y_pred / tf.maximum(tf.linalg.norm(tf.stop_gradient(y_pred), axis=-2, keepdims=True), self.eps)

        for i in range(n):
            for j in range(i+1, n):
                if self.a is not None and not self.a[i][j] and not self.a[j][i]:
                    continue
                anchor_i = y_pred[:, self.start_t + self.contrast_t:, :, i]      # (B, T-cont_t-start_t, DIM)
                anchor_j = y_pred[:, self.start_t + self.contrast_t:, :, j]    # (B, T-cont_t-start_t, DIM)
                pos_val = self.val_func(anchor_i, anchor_j)  # (B, T-cont_t-start_t)

                if self.a is None or (self.a is not None and self.a[i][j]):
                    negative_j = y_pred[:, self.start_t:-self.contrast_t, :, j]  # (B, T-cont_t-start_t, DIM)
                    neg_val_i = self.val_func(anchor_i, negative_j)   # (B, T-cont_t-start_t)
                    loss += tf.reduce_mean(self.loss_func(pos_val, neg_val_i))

                if self.a is None or (self.a is not None and self.a[j][i]):
                    negative_i = y_pred[:, self.start_t:-self.contrast_t, :, i]  # (B, T-cont_t-start_t, DIM)
                    neg_val_j = self.log_similarity(anchor_j, negative_i)   # (B, T-cont_t-start_t)
                    loss += tf.reduce_mean(self.loss_func(pos_val, neg_val_j))

        return loss / (tf.cast(n * (n - 1), dtype=loss.dtype) if self.a is None else tf.reduce_sum(tf.cast(self.a, dtype=loss.dtype)))


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