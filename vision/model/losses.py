from utils.tf_utils import serialize
import tensorflow as tf
from tensorflow.keras.losses import Loss

serialize(tf.keras.regularizers.L2)


@serialize
class ContrastiveLoss(Loss):
    def __init__(self, *args, pos=.5, neg=.5, m=10, **kwargs):
        super(ContrastiveLoss, self).__init__(*args, **kwargs)
        self.pos = pos
        self.neg = neg
        self.m = m

    def call(self, y_true, y_pred):
        b = tf.shape(y_pred)[0]
        n = tf.shape(y_pred)[-1]

        samples_mask = tf.eye(b, dtype=tf.bool)[..., None, None]  # (loss over diagonal)
        models_mask = (~tf.eye(n, dtype=tf.bool))[None, None]  # (loss over every pair)

        positive_mask = samples_mask & models_mask
        negative_mask = (~samples_mask) & models_mask

        pos_n = tf.cast(tf.reduce_sum(tf.cast(positive_mask, tf.int32)), y_pred.dtype)
        neg_n = tf.cast(tf.reduce_sum(tf.cast(negative_mask, tf.int32)), y_pred.dtype)

        embedding = y_pred
        dist = tf.reduce_sum(tf.pow(embedding[:, None, ..., None, :] - embedding[None, :, ..., None], 2), axis=2)
        pos = tf.reduce_sum(tf.where(positive_mask, dist / pos_n, 0.))
        if self.neg:
            neg = tf.reduce_sum(tf.where(negative_mask, self.calculate_negative(dist) / neg_n, 0.))
        else:
            neg = 0.
        return self.pos * pos + self.neg * neg

    def calculate_negative(self, dist):
        neg = tf.pow(tf.maximum(0., self.m - tf.math.sqrt(dist)), 2)
        return neg


@serialize
class ContrastiveLossExpDecay(ContrastiveLoss):
    def __init__(self, *args, tau=1, **kwargs):
        super(ContrastiveLossExpDecay, self).__init__(*args, **kwargs)
        self.tau = tau

    def calculate_negative(self, dist):
        exp_decay = tf.exp((self.m - dist) / self.tau)
        squared = super(ContrastiveLossExpDecay, self).calculate_negative(dist)
        return tf.where(squared > 0, squared, exp_decay)


@serialize
class ContrastiveSoftmaxLoss(Loss):
    def __init__(self, *args, temperature=10, eps=0, stable=True, **kwargs):
        super(ContrastiveSoftmaxLoss, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.eps = eps
        self.stable = stable

    def calculate_logits(self, embedding):
        dist = tf.reduce_sum(tf.pow(embedding[:, None, ..., None, :] - embedding[None, :, ..., None], 2), axis=2)
        if self.eps:
            dist = tf.minimum(dist, self.eps)
        similarity = -dist / self.temperature
        if self.stable:
            similarity = similarity - tf.reduce_max(similarity, axis=0, keepdims=True)
        logits = tf.exp(similarity)
        return logits

    def calculate_probs(self, embedding, similarity=None):
        similarity = self.calculate_logits(embedding) if similarity is None else similarity
        softmaxed = similarity / tf.reduce_sum(similarity, axis=0, keepdims=True)
        return softmaxed

    def get_pos_mask(self, y_pred):
        b = tf.shape(y_pred)[0]
        n = tf.shape(y_pred)[-1]

        samples_mask = tf.eye(b, dtype=tf.bool)[..., None, None]  # (loss over diagonal)
        models_mask = (~tf.eye(n, dtype=tf.bool))[None, None]  # (loss over every pair)

        positive_mask = samples_mask & models_mask
        pos_n = tf.cast(tf.reduce_sum(tf.cast(positive_mask, tf.int32)), y_pred.dtype)
        return positive_mask, pos_n

    def call(self, y_true, y_pred):
        probs = self.calculate_probs(y_pred)
        positive_mask, pos_n = self.get_pos_mask(y_pred)
        return 1 - tf.reduce_sum(tf.where(positive_mask, probs, 0)) / pos_n


class GeneralPullPushGraphLoss(ContrastiveSoftmaxLoss):
    def __init__(self, *args, a_pull, a_push, log_eps=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_pull = tf.constant(a_pull)
        self.a_push = tf.constant(a_push)
        self.is_pull = tf.reduce_any(a_pull)
        self.is_push = tf.reduce_any(a_push)
        self.log_eps = log_eps

    def map_rep_dev(self, similarity):
        b = tf.shape(similarity)[0]
        n = tf.shape(similarity)[-1]
        self_sim = tf.reshape(similarity[~tf.eye(b, dtype=tf.bool)[..., None, None] & tf.eye(n, dtype=tf.bool)[None, None]], (b-1, b, n))
        map_rep = tf.transpose(self_sim / tf.reduce_sum(self_sim, axis=0, keepdims=True), [1,2,0])  # (b, n, b-1)
        log_map_rep = tf.math.log2(tf.maximum(map_rep, self.log_eps))
        cross_ent = tf.linalg.matmul(map_rep, log_map_rep, transpose_b=True)  # (b, n, n)
        entropy = tf.reshape(cross_ent[tf.eye(n)[None]], (b, n, 1))
        dkl = entropy - cross_ent   # (b, n, n)
        return dkl

    def call(self, y_true, y_pred):
        logits = self.calculate_logits(y_pred)
        loss = 0.

        if self.is_pull:
            probs = self.calculate_probs(None, logits)[tf.eye(tf.shape(logits)[0])]     # (b, n, n)
            mean_probs = tf.reduce_mean(probs, axis=0)
            pull_loss = tf.tensordot(self.a_pull, 1 - mean_probs, axes=(0, 1))
            loss += pull_loss

        if self.is_push:
            mrdev = self.map_rep_dev(logits)   # (b, n, n)
            mean_mrdev = tf.reduce_mean(mrdev, axis=0)
            push_loss = tf.tensordot(self.a_push, -mean_mrdev, axes=(0, 1))
            loss += push_loss

        return loss


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
