from utils.tf_utils import serialize
import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np
import scipy

A_PULL = None
A_PUSH = None

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
    def __init__(self, *args, temperature=10, eps=0, stable=True, cosine=False, **kwargs):
        super(ContrastiveSoftmaxLoss, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.eps = eps
        self.stable = stable
        self.cosine = cosine

    def calculate_logits(self, embedding):
        if self.cosine:
            normed_embedding = embedding / tf.linalg.norm(tf.stop_gradient(embedding), axis=1, keepdims=True)
            cosine_sim = tf.einsum('bdn,BdN->bBnN', normed_embedding, normed_embedding)
            logits = cosine_sim / self.temperature
        else:
            dist = tf.reduce_sum(tf.pow(embedding[:, None, ..., None, :] - embedding[None, :, ..., None], 2), axis=2)
            if self.eps:
                dist = tf.minimum(dist, self.eps)
            logits = -dist / self.temperature
        return logits

    def calculate_exp_logits(self, embedding, logits=None):
        logits = self.calculate_logits(embedding) if logits is None else logits
        if self.stable:
            logits = logits - tf.reduce_max(logits, axis=0, keepdims=True)
        return tf.exp(logits)

    def calculate_likelihood(self, embedding, exp_logits=None, logits=None):
        exp_logits = self.calculate_exp_logits(embedding, logits=logits) if exp_logits is None else exp_logits
        softmaxed = exp_logits / tf.reduce_sum(exp_logits, axis=0, keepdims=True)
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
        probs = self.calculate_likelihood(y_pred)
        positive_mask, pos_n = self.get_pos_mask(y_pred)
        return 1 - tf.reduce_sum(tf.where(positive_mask, probs, 0)) / pos_n


class GeneralPullPushGraphLoss(ContrastiveSoftmaxLoss):
    def __init__(self, *args, a_pull, a_push, w_push=1, log_eps=1e-10, log_pull=False, contrastive=True, remove_diag=True, **kwargs):
        super().__init__(*args, **kwargs)
        global A_PULL
        global A_PUSH
        A_PULL = tf.constant(eval(a_pull) if isinstance(a_pull, str) else a_pull, dtype=tf.float32)
        A_PUSH = tf.constant(eval(a_push) if isinstance(a_push, str) else a_push, dtype=tf.float32)
        self.a_pull = A_PULL
        self.a_push = A_PUSH * w_push
        self.w_push = w_push
        self.is_pull = tf.reduce_any(self.a_pull != 0).numpy()
        self.is_push = tf.reduce_any(self.a_push != 0).numpy()
        self.log_eps = log_eps
        self.log_pull = log_pull
        self.contrastive = contrastive
        self.remove_diag = remove_diag

    def map_rep_dev(self, exp_logits=None, logits=None):
        assert (logits is not None) or (exp_logits is not None)
        cur = exp_logits if exp_logits is not None else logits # Will be converted to exp_logits later
        b = tf.shape(cur)[0]
        n = tf.shape(cur)[-1]
        if self.remove_diag:
            self_sim = tf.reshape(cur[~tf.eye(b, dtype=tf.bool)[..., None, None] & tf.eye(n, dtype=tf.bool)[None, None]], (b-1, b, n))
        else:
            self_sim = cur
        if exp_logits is None:
            self_sim = self.calculate_exp_logits(None, self_sim)
        map_rep = self_sim / tf.reduce_sum(self_sim, axis=0, keepdims=True)  # (b-1, b, n)
        log_map_rep = tf.experimental.numpy.log2(tf.maximum(map_rep, self.log_eps))
        cross_ent = tf.einsum('ibn,ibm->bnm', map_rep, log_map_rep)  # (b, n, n)
        entropy = tf.linalg.diag_part(cross_ent)[..., None]
        dkl = entropy - cross_ent   # (b, n, n)
        return dkl

    def distance(self, embedding):
        dist = tf.reduce_sum(tf.pow(embedding[..., None, :] - embedding[..., None], 2), axis=-3)
        return dist

    def call(self, y_true, y_pred):
        logits = self.calculate_logits(y_pred)
        if self.stable and (self.is_pull and not self.log_pull) and self.is_push:
            exp_logits = self.calculate_exp_logits(None, logits=logits)
        else:
            exp_logits = None
        loss = 0.

        if self.is_pull:
            if self.contrastive:
                if self.log_pull:
                    log_likelihood = logits[tf.eye(tf.shape(logits)[0], dtype=tf.bool)] - tf.math.reduce_logsumexp(logits, axis=0)
                else:
                    # (b, n, n)
                    likelihood = self.calculate_likelihood(None, exp_logits=exp_logits, logits=logits)[tf.eye(tf.shape(logits)[0], dtype=tf.bool)]
                mean_gain = tf.reduce_mean(log_likelihood if self.log_pull else likelihood, axis=0)
                pull_loss = tf.tensordot(self.a_pull, (0 if self.log_pull else 1) - mean_gain, axes=[[0, 1], [0, 1]])
            else:
                mean_dist = tf.reduce_mean(self.distance(embedding=y_pred), axis=0)
                pull_loss = tf.tensordot(self.a_pull, mean_dist, axes=[[0, 1], [0, 1]])
            loss += pull_loss

        if self.is_push:
            mrdev = self.map_rep_dev(exp_logits=exp_logits, logits=logits)   # (b, n, n)
            mean_mrdev = tf.reduce_mean(mrdev, axis=0)
            push_loss = tf.tensordot(self.a_push, -mean_mrdev, axes=[[0, 1], [0, 1]])
            loss += push_loss

        return loss


class ProbabilisticPullPushGraphLoss(GeneralPullPushGraphLoss):
    def __init__(self, num_pathways, *args, p_pull=1., p_push=0., depend=False, **kwargs):
        a_pull = (np.random.rand(num_pathways, num_pathways) <= p_pull) & ~np.eye(num_pathways, dtype=np.bool)
        a_push = (np.random.rand(num_pathways, num_pathways) <= p_push) & ~np.eye(num_pathways, dtype=np.bool)
        if depend:
            a_push = a_push & ~a_pull
        super(ProbabilisticPullPushGraphLoss, self).__init__(*args, a_pull=a_pull, a_push=a_push, **kwargs)


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
