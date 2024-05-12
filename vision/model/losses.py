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

    def calculate_dists(self, embedding, self_only=False, stop_grad=False):
        # TODO: try to figure if the gradient flow in this case with top_k != 0 is correct
        if stop_grad:
            if self_only:
                dist = tf.reduce_sum(tf.pow(embedding[:, None] - tf.stop_gradient(embedding)[None, :], 2), axis=2)
            else:
                dist = tf.reduce_sum(tf.pow(embedding[:, None, ..., :, None] - tf.stop_gradient(embedding)[None, :, ..., None, :], 2),
                                     axis=2)
        else:
            if self_only:
                dist = tf.reduce_sum(tf.pow(embedding[:, None] - embedding[None, :], 2), axis=2)
            else:
                dist = tf.reduce_sum(tf.pow(embedding[:, None, ..., None, :] - embedding[None, :, ..., None], 2), axis=2)
        return dist

    def calculate_logits(self, embedding, dist=None, self_only=False):
        if self.cosine:
            normed_embedding = embedding / tf.linalg.norm(tf.stop_gradient(embedding), axis=1, keepdims=True)
            cosine_sim = tf.einsum('bdn,Bdn->bBn' if self_only else 'bdn,BdN->bBnN', normed_embedding, normed_embedding)
            logits = cosine_sim / self.temperature
        else:
            dist = dist if dist is not None else self.calculate_dists(embedding, self_only=self_only)
            if self.eps:
                dist = tf.minimum(dist, self.eps)
            logits = -dist / self.temperature
        return logits

    def calculate_exp_logits(self, embedding, logits=None):
        logits = self.calculate_logits(embedding) if logits is None else logits
        if self.stable:
            logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=0, keepdims=True)
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
    def __init__(self, *args, a_pull, a_push, w_push=1, log_eps=1e-10, log_pull=False, contrastive=True,
                 remove_diag=True, corr=False, use_dists=False, naive_push=False, naive_push_max=None, naive_djs=False,
                 top_k=0, stop_grad_dist=False, push_linear_predictivity=None, push_linear_predictivity_normalize=True,
                 linear_predictivity_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        global A_PULL
        global A_PUSH
        eval_a_push = eval(a_push) if isinstance(a_push, str) else a_push
        A_PULL = tf.constant(eval(a_pull) if isinstance(a_pull, str) else a_pull, dtype=tf.float32)
        A_PUSH = tf.constant(eval_a_push, dtype=tf.float32)
        self.a_pull = A_PULL
        self.a_push = A_PUSH * w_push
        self.neg_in_pull = tf.reduce_any(self.a_pull < 0)
        self.a_pull_neg_mask = self.a_pull < 0
        self.w_push = w_push
        self.is_pull = tf.reduce_any(self.a_pull != 0).numpy()
        self.is_push = tf.reduce_any(self.a_push != 0).numpy()
        self.log_eps = log_eps
        self.log_pull = log_pull
        self.naive_push = naive_push
        self.naive_djs = naive_djs
        self.naive_push_max = naive_push_max
        self.contrastive = contrastive
        self.remove_diag = remove_diag
        self.corr = corr
        self.use_dists = use_dists
        self.top_k = top_k
        self.stop_grad_dist = stop_grad_dist
        self.push_linear_predictivity = LinearPredictivity([[-w * w_push for w in vec] for vec in eval_a_push],
                                                           normalize=push_linear_predictivity_normalize, **linear_predictivity_kwargs) if push_linear_predictivity else None

    def map_rep_dev(self, exp_logits=None, logits=None):
        assert (logits is not None) or (exp_logits is not None)
        cur = exp_logits if exp_logits is not None else logits # Will be converted to exp_logits later
        b = tf.shape(cur)[0]
        n = tf.shape(cur)[-1]
        if self.remove_diag:
            self_sim = tf.reshape(cur[tf.tile(~tf.eye(b, dtype=tf.bool)[..., None], [1, 1, n])], (b-1, b, n))
        else:
            self_sim = cur  # (b, b, n)
        if exp_logits is None:
            self_sim = self.calculate_exp_logits(None, self_sim)
        map_rep = self_sim / tf.reduce_sum(self_sim, axis=0, keepdims=True)  # (b-remove_diag, b, n)
        log_map_rep = tf.experimental.numpy.log2(tf.maximum(map_rep, self.log_eps))
        cross_ent = tf.einsum('ibn,ibm->bnm', map_rep, log_map_rep)  # (b, n, n)
        entropy = tf.linalg.diag_part(cross_ent)[..., None]
        dkl = entropy - cross_ent   # (b, n, n)
        return dkl

    def calculate_correlation(self, exp_logits=None, logits=None, dists=None):
        using_dist = self.use_dists and dists is not None
        assert (logits is not None) or (exp_logits is not None) or using_dist
        cur = dists if using_dist else (exp_logits if exp_logits is not None else logits)  # Will be converted to exp_logits later
        b = tf.shape(cur)[0]
        n = tf.shape(cur)[-1]
        if self.remove_diag:
            cur = tf.reshape(cur[tf.tile(~tf.eye(b, dtype=tf.bool)[..., None], [1, 1, n])], (b-1, b, n))
        if exp_logits is None and not using_dist:
            cur = self.calculate_exp_logits(None, cur)

        size = float(b * (b-self.remove_diag))
        _mean = tf.math.reduce_mean(tf.stop_gradient(cur), axis=(0, 1))    # (n, )
        _std = tf.math.reduce_std(tf.stop_gradient(cur), axis=(0, 1))      # (n, )
        mult = tf.einsum('ijn,ijm->nm', cur, cur)                          # (n, n)

        correlation = (mult / size - _mean[None] * _mean[:, None]) / (_std[None] * _std[:, None])

        return correlation

    def distance(self, embedding):
        dist = tf.reduce_sum(tf.pow(embedding[..., None, :] - embedding[..., None], 2), axis=-3)
        return dist

    def call(self, y_true, y_pred):
        dists = self.calculate_dists(y_pred, self_only=not self.is_pull, stop_grad=self.stop_grad_dist)
        logits = exp_logits = None
        if self.is_pull or not self.use_dists:
            logits = self.calculate_logits(y_pred, dist=dists, self_only=not self.is_pull)
            if self.stable and (self.is_pull and not self.log_pull) and self.is_push:
                exp_logits = self.calculate_exp_logits(None, logits=logits)
        loss = 0.
        b = tf.shape(y_pred)[0]
        if self.is_pull:
            if self.contrastive:
                if self.log_pull:
                    gain = logits[tf.eye(tf.shape(logits)[0], dtype=tf.bool)] - tf.math.reduce_logsumexp(logits, axis=0)
                else:
                    # (b, n, n)
                    gain = self.calculate_likelihood(None, exp_logits=exp_logits, logits=logits)[tf.eye(tf.shape(logits)[0], dtype=tf.bool)]

                if self.neg_in_pull:
                    gain = tf.where(self.a_pull_neg_mask, tf.maximum(gain, tf.cast(1/b, tf.float32)), gain)

                if self.top_k:
                    top_k_vals = tf.reduce_min(tf.math.top_k(gain, k=self.top_k, sorted=False).values,
                                               axis=-1, keepdims=True)
                    top_k_mask = gain >= top_k_vals
                    gain = tf.where(top_k_mask, gain, 0.)

                mean_gain = tf.reduce_mean(gain, axis=0)
                pull_loss = tf.tensordot(self.a_pull, (0 if self.log_pull else 1) - mean_gain, axes=[[0, 1], [0, 1]])
            else:
                mean_dist = tf.reduce_mean(self.distance(embedding=y_pred), axis=0)
                pull_loss = tf.tensordot(self.a_pull, mean_dist, axes=[[0, 1], [0, 1]])
            loss += pull_loss

        if self.is_push:
            if self.push_linear_predictivity is not None:
                loss += self.push_linear_predictivity(y_pred)
            else:
                if self.naive_push:
                    if self.is_pull:
                        dists = dists[tf.eye(tf.shape(dists)[0], dtype='bool')]                                 # (b, n, n)
                    else:
                        dists = tf.reduce_sum(tf.pow(y_pred[..., None, :] - y_pred[..., None], 2), axis=2)      # (b, n, n)
                    normalized_dists = dists / float(tf.shape(y_pred)[1])
                    if self.naive_push_max is not None:
                        normalized_dists = tf.minimum(normalized_dists, self.naive_push_max)
                    paths_loss = tf.reduce_mean(-normalized_dists, axis=0)
                elif self.naive_djs:
                    exps = tf.math.exp(y_pred / self.temperature)
                    ps = exps / tf.reduce_sum(exps, axis=1, keepdims=True)  # (b, dim, n)
                    m = (ps[..., None, :] + ps[..., None]) / 2              # (b, dim, n, n)
                    log_ps = tf.math.log(ps)                                # (b, dim, n)
                    log_m = tf.math.log(m)                                  # (b, dim, n, n)
                    minus_entropy = tf.einsum('bin,bin->bn', ps, log_ps)    # (b, n)
                    cross_entropy = tf.einsum('bin,binm->bnm', ps, log_m)   # (b, n, n)
                    djs = minus_entropy - cross_entropy
                    paths_loss = -tf.reduce_mean(djs, axis=0)
                else:
                    if self.is_pull:
                        if self.use_dists:
                            dists = tf.linalg.diag_part(dists)  # (b, b, n)
                        else:
                            if exp_logits is None:
                                logits = tf.linalg.diag_part(logits)
                            else:
                                exp_logits = tf.linalg.diag_part(exp_logits)

                    if self.corr:
                        paths_loss = tf.pow(self.calculate_correlation(exp_logits=None if self.use_dists else exp_logits,
                                                                       logits=None if self.use_dists else logits,
                                                                       dists=dists if self.use_dists else None), 2)  # R^2
                    else:
                        mrdev = self.map_rep_dev(exp_logits=exp_logits, logits=logits)   # (b, n)
                        paths_loss = -tf.reduce_mean(mrdev, axis=0)
                push_loss = tf.tensordot(self.a_push, paths_loss, axes=[[0, 1], [0, 1]])
                loss += push_loss

        return loss


class ProbabilisticPullPushGraphLoss(GeneralPullPushGraphLoss):
    def __init__(self, num_pathways, *args, p_pull=1., p_push=0., depend=False, **kwargs):
        a_pull = (np.random.rand(num_pathways, num_pathways) <= p_pull) & ~np.eye(num_pathways, dtype=np.bool)
        a_push = (np.random.rand(num_pathways, num_pathways) <= p_push) & ~np.eye(num_pathways, dtype=np.bool)
        if depend:
            a_push = a_push & ~a_pull
        super(ProbabilisticPullPushGraphLoss, self).__init__(*args, a_pull=a_pull, a_push=a_push, **kwargs)


class CommunitiesLoss(GeneralPullPushGraphLoss):
    def __init__(self, num_pathways, num_communities, *args, **kwargs):
        assert not (num_pathways % num_communities)
        pathways_per_community = int(num_pathways / num_communities)
        a_pull = scipy.linalg.block_diag(*[1-np.eye(pathways_per_community).astype(np.float32)]*num_communities)
        if (a_pull != 0).any():
            a_pull /= num_communities * pathways_per_community **2 - num_pathways
        a_push = ((np.eye(num_pathways) == 0) & (a_pull == 0)).astype(np.float32)
        if (a_push != 0).any():
            a_push /= a_push.sum()
        super(CommunitiesLoss, self).__init__(*args, a_pull=a_pull, a_push=a_push, **kwargs)


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


# class BasicBatchContrastiveLoss(tf.keras.losses.Loss):
#     def __init__(self, temperature=10, partition_along_axis=1):
#         super(BasicBatchContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.partition_along_axis = partition_along_axis
#
#     def call(self, y_true, y_pred):
#         # embd shape: (B, dim)
#         dists_sqr = tf.reduce_sum(tf.pow(y_true[:, None] - y_pred[None], 2), axis=-1)  # (B1, B2)
#         logits = -dists_sqr / self.temperature
#         exps = tf.math.exp(logits)
#         partition = tf.reduce_sum(exps, axis=self.partition_along_axis)
#         gain = tf.linalg.diag_part(logits) - tf.math.log(partition)
#         mean_loss = -tf.reduce_mean(gain)
#         return mean_loss


class LateralPredictiveLoss(tf.keras.losses.Loss):
    def __init__(self, graph, name='lateral_pred_loss', temperature=10, partition_along_axis=1, **kwargs):
        super(LateralPredictiveLoss, self).__init__(name=name)
        self.graph = eval(graph) if isinstance(graph, str) else graph
        self.temperature = temperature
        self.partition_along_axis = partition_along_axis
        self.n = len(self.graph)

    def basic_batch_contrastive_loss(self, y_true, y_pred):
        dists_sqr = tf.reduce_sum(tf.pow(y_true[:, None] - y_pred[None], 2), axis=-1)  # (B1, B2)
        logits = -dists_sqr / self.temperature
        logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=self.partition_along_axis, keepdims=True)
        gain = tf.linalg.diag_part(logits) - tf.math.reduce_logsumexp(logits, axis=self.partition_along_axis)
        mean_loss = -tf.reduce_mean(gain)
        return mean_loss

    def call(self, y_true, y_pred):
        loss = 0.
        for j in range(self.n):
            target = y_pred[..., j, j]
            for i in range(self.n):
                if self.graph[i][j]:
                    pred = y_pred[..., i, j]
                    loss += self.graph[i][j] * self.basic_batch_contrastive_loss(target, pred)
        return loss


# class LinearPredictivityLoss(tf.keras.losses.Loss):
#     """
#     Say we have X and Y (samples by pathway i and pathway j).
#     Using linear regression we get M=(XT @ X)^-1 @ XT @ Y that minimizes the squared loss.
#     This loss uses a trick to calculate the least squares (or something similar) without calculating M.
#     E[NORM[(XT @ X)(Mx - y)]^2] over (x,y)
#     = E[NORM[(XT @ X) @ (XT @ X)^-1 @ XT @ Y @ x - XT @ X @ y]] over (x,y)
#     = E[NORM[XT @ Y @ x - XT @ X @ y]] over (x,y)
#     """
#     def __init__(self, graph, normalize=True, **kwargs):
#         super(LinearPredictivityLoss, self).__init__(**kwargs)
#         self.graph = eval(graph) if isinstance(graph, str) else graph
#         self.n = len(self.graph)
#         self.normalize = normalize
#
#     def call(self, y_true, y_pred):
#         if self.normalize:
#             y_pred = y_pred - tf.reduce_mean(y_pred, axis=0, keepdims=True)
#             y_pred = y_pred / tf.reduce_mean(tf.linalg.norm(tf.stop_gradient(y_pred), axis=1, keepdims=True),
#                                              axis=0, keepdims=True)
#         loss = 0.
#         for i in range(self.n):
#             if any(self.graph[i]):
#                 X = y_pred[..., i]                          # (B, dim)
#                 XT = tf.transpose(X)                        # (dim, B)
#                 for j in range(i+1, self.n):
#                     if self.graph[i][j]:
#                         Y = y_pred[..., j]                  # (B, dim)
#                         YXT = Y @ XT                        # (B, B)
#                         XYT = tf.transpose(YXT)             # (B, B)
#                         diff = XT @ (YXT - XYT)             # (dim, B)
#                         sample_loss = tf.reduce_sum(tf.pow(diff, 2), axis=0)        # (B, )
#                         mean_loss = tf.reduce_mean(sample_loss)     # (1, )
#                         loss += self.graph[i][j] * mean_loss
#         return loss


class LinearPredictivity:
    """
    Say we have X and Y (samples by pathway i and pathway j).
    Using linear regression we get M=(XT @ X)^-1 @ XT @ Y that minimizes the squared loss.
    This loss uses a trick to calculate the least squares (or something similar) without calculating M.
    E[NORM[(XT @ X)(Mx - y)]^2] over (x,y)
    = E[NORM[(XT @ X) @ (XT @ X)^-1 @ XT @ Y @ x - XT @ X @ y]] over (x,y)
    = E[NORM[XT @ Y @ x - XT @ X @ y]] over (x,y)
    """
    def __init__(self, graph, normalize=True, trick=True, ridge=0):
        self.graph = eval(graph) if isinstance(graph, str) else graph
        self.n = len(self.graph)
        self.normalize = normalize
        self.trick = trick
        self.ridge = ridge

    def __call__(self, arr):
        """
        :param arr: (B, dim, P)
        """
        if self.normalize:
            arr = arr - tf.reduce_mean(arr, axis=0, keepdims=True)
            arr = arr / tf.reduce_mean(tf.linalg.norm(tf.stop_gradient(arr), axis=1, keepdims=True),
                                       axis=0, keepdims=True)
        loss = 0.
        for i in range(self.n):
            X = arr[..., i]                             # (B, dim)
            XT = tf.transpose(X)                        # (dim, B)
            for j in range(i+1, self.n):
                if self.graph[i][j]:
                    Y = arr[..., j]                     # (B, dim)
                    if self.trick:
                        YXT = Y @ XT                        # (B, B)
                        XYT = tf.transpose(YXT)             # (B, B)
                        diff = XT @ (YXT - XYT)             # (dim, B)
                    else:
                        to_inverse = XT @ X
                        if self.ridge:
                            to_inverse += tf.eye(tf.shape(to_inverse)[0]) * self.ridge
                        w = tf.linalg.pinv(XT @ X + to_inverse) @ XT @ Y  # (dim, dim)
                        diff = X @ w - Y
                    sample_loss = tf.reduce_sum(tf.pow(diff, 2), axis=0)        # (B, )
                    mean_loss = tf.reduce_mean(sample_loss)     # (1, )
                    loss += self.graph[i][j] * mean_loss
        return loss
