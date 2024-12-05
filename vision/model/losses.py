import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np
import scipy
from utils.model.losses import *
from utils.model.metrics import LossMonitors


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

    def calculate_exp_logits(self, embedding=None, logits=None):
        assert embedding is not None or logits is not None
        logits = self.calculate_logits(embedding) if logits is None else logits
        if self.stable:
            logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=0, keepdims=True)
        return tf.exp(logits)

    def calculate_likelihood(self, embedding=None, exp_logits=None, logits=None):
        assert embedding is not None or exp_logits is not None or logits is not None
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
                 linear_predictivity_kwargs={}, entropy_w=0, cka=False, **kwargs):
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
        self.entropy_w = entropy_w
        self.cka = cka
        assert (self.entropy_w and self.log_pull) or not self.entropy_w, "entropy loss only for log pull"
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

    def calculate_cka(self, embedding=None, logits=None, exp_logits=None):
        """
        A measure of independence. 0 means independent.
        https://arxiv.org/pdf/1905.00414
        https://arxiv.org/pdf/2202.07757
        """
        assert embedding is not None or logits is not None or exp_logits is not None
        if exp_logits is None:
            exp_logits = self.calculate_exp_logits(embedding=embedding, logits=logits)

        exp_logits = tf.linalg.diag_part(exp_logits)  # (b, b, n)
        centered_exp_logits = exp_logits - tf.reduce_mean(exp_logits, axis=1, keepdims=True)
        b = tf.shape(exp_logits)[0]
        n = tf.shape(exp_logits)[2]

        # (N, N)
        denom = tf.cast((b - 1) ** 2, dtype=centered_exp_logits.dtype)
        hsic = tf.reduce_sum(tf.reshape(tf.einsum('bBn,BkN->bknN',
                                                  centered_exp_logits,
                                                  centered_exp_logits)[tf.tile(tf.eye(b, dtype=tf.bool)[..., None, None],
                                                                          [1, 1, n, n])], (b, n, n)), axis=0) / denom
        cka = hsic / tf.sqrt(tf.linalg.diag_part(hsic)[..., None] * tf.linalg.diag_part(hsic)[None])
        return cka

    def call(self, y_true, y_pred):
        if not self.cosine or self.is_push:
            dists = self.calculate_dists(y_pred, self_only=not self.is_pull, stop_grad=self.stop_grad_dist)
        else:
            dists = None
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
                    log_z = tf.math.reduce_logsumexp(logits, axis=0)
                    gain = logits[tf.eye(tf.shape(logits)[0], dtype=tf.bool)] - log_z

                    if self.entropy_w:
                        b = tf.shape(logits)[0]
                        n = tf.shape(logits)[-1]
                        logits_without_self = tf.reshape(logits[tf.tile(~tf.eye(b, dtype=tf.bool)[..., None, None], [1, 1, n, n])],
                                                         (b - 1, b, n, n))    # (B-1, B, N, N)
                        likelihood_without_self = self.calculate_likelihood(None, logits=logits_without_self)
                        log_likelihood_without_self = tf.math.log(likelihood_without_self)
                        minus_entropy_without_self = tf.einsum('bBnN,bBnN->BnN', likelihood_without_self, log_likelihood_without_self)
                        mean_minus_entropy_without_self = tf.reduce_mean(minus_entropy_without_self, axis=0)  # (N, N)

                        entropy_loss = tf.tensordot(self.a_pull, mean_minus_entropy_without_self, axes=[[0, 1], [0, 1]])
                        loss += entropy_loss * self.entropy_w

                else:
                    # (b, n, n)
                    likelihood = self.calculate_likelihood(None, exp_logits=exp_logits, logits=logits)
                    gain = likelihood[tf.eye(tf.shape(logits)[0], dtype=tf.bool)]

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
                elif self.cka:
                    paths_loss = self.calculate_cka(embedding=y_pred, exp_logits=exp_logits)
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
        a_pull = (np.random.rand(num_pathways, num_pathways) <= p_pull) & (np.eye(num_pathways) == 0)
        a_push = (np.random.rand(num_pathways, num_pathways) <= p_push) & (np.eye(num_pathways) == 0)
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


class ConfidenceContrastiveLoss(ContrastiveSoftmaxLoss):
    def __init__(self, *args, c_w=1, squared=False, threshold=0, implicit=False, stop_gradient=False, **kwargs):
        super(ConfidenceContrastiveLoss, self).__init__(*args, **kwargs)
        self.c_w = c_w
        self.squared = squared
        self.threshold = threshold
        self.implicit = implicit
        self.stop_gradient = stop_gradient

    def call(self, y_true, y_pred):
        b = tf.shape(y_pred)[0]
        n = tf.shape(y_pred)[2]
        if self.implicit:
            embedding = y_pred
            confidence = None
        else:
            embedding = y_pred[:, :-1]                      # (B, dim, N)
            confidence = tf.math.sigmoid(y_pred[:, -1])     # (B, N)
            confidence = confidence[..., None, :] * confidence[..., None]

        likelihood = self.calculate_likelihood(embedding=embedding)  # (B, B, N, N)
        if self.implicit:
            # confidence based on entropy
            confidence = 1 + tf.einsum('bBnN,bBnN->BnN', likelihood, tf.math.log(likelihood)) / tf.math.log(tf.cast(b, dtype=likelihood.dtype))

        if self.stop_gradient:
            confidence = tf.stop_gradient(confidence)

        likelihood = tf.reshape(likelihood[tf.tile((tf.eye(b, dtype=tf.bool))[None, None], [1, 1, n, n])], (b, n, n))     # (B, N, N)
        weighted_non_likelihood = confidence * (1 - likelihood - self.threshold)
        loss = tf.reduce_mean(weighted_non_likelihood[tf.tile((~tf.eye(n, dtype=tf.bool))[None], [b, 1, 1])])
        if self.squared:
            loss = loss + self.c_w * tf.reduce_mean((1-confidence)**2)
        else:
            loss = loss - self.c_w * tf.reduce_mean(confidence)
        return loss


class CrossEntropyAgreement(tf.keras.losses.Loss):
    def __init__(self, *args, w_ent=0., w_ent_mean=0., stable=True, **kwargs):
        super(CrossEntropyAgreement, self).__init__(*args, **kwargs)
        self.w_ent = w_ent
        self.w_ent_mean = w_ent_mean
        self.stable = stable

    def call(self, y_true, y_pred):
        embedding = y_pred      # (B, dim, N)
        b = tf.shape(embedding)[0]
        n = tf.shape(embedding)[-1]
        if self.stable:
            embedding = embedding - tf.reduce_max(embedding, axis=0, keepdims=True)
        exps = tf.exp(embedding)
        probs = exps / tf.reduce_sum(exps, axis=1, keepdims=True)   # (B, dim, N)
        log_probs = tf.math.log(probs)
        minus_cross_entropy = tf.einsum("bdn,bdN->bnN", probs, log_probs)
        loss = -tf.reduce_mean(minus_cross_entropy[tf.tile((~tf.eye(n, dtype=tf.bool))[None], [b, 1, 1])])

        if self.w_ent:
            loss = loss - self.w_ent * tf.reduce_mean(tf.linalg.diag_part(minus_cross_entropy))

        if self.w_ent_mean:
            mean_probs = tf.reduce_mean(probs, axis=0)
            log_mean_probs = tf.math.log(mean_probs)
            minus_entropy_mean_probs = tf.einsum('dn,dn->n', mean_probs, log_mean_probs)
            loss = loss + self.w_ent_mean * tf.reduce_mean(minus_entropy_mean_probs)
        return loss


class LogLikelihoodIterativeSoftmax(Loss):
    def __init__(self, *args, a=None, temperature=10, **kwargs):
        super(LogLikelihoodIterativeSoftmax, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.a = a

    def call(self, y_true, y_pred):
        # b = tf.shape(y_pred)[0]
        # dim = tf.shape(y_pred)[1]
        n = tf.shape(y_pred)[2]
        loss = 0.

        for i in range(n):
            for j in range(i+1, n):
                if i == j:
                    continue
                if self.a is not None and not self.a[i][j] and not self.a[j][i]:
                    continue

                sqr_dist = tf.reduce_sum(tf.pow(y_pred[:, None, ..., i] - y_pred[None, ..., j], 2), axis=-1)      # (B, B)
                logits = -sqr_dist / self.temperature
                diag_logits = tf.linalg.diag_part(logits)   # (B, )
                for side in range(2):
                    if self.a is None or (self.a is not None and (self.a[i][j] if not side else self.a[j][i])):
                        log_z = tf.math.reduce_logsumexp(logits, axis=side)     # (B, )
                        log_likelihood = diag_logits - log_z                    # (B, )
                        loss = loss - tf.reduce_mean(log_likelihood)
        return loss / (tf.cast(n ** 2, dtype=loss.dtype) if self.a is None else tf.reduce_sum(tf.cast(self.a, dtype=loss.dtype)))


class BasicDisagreement(tf.keras.losses.Loss):
    def __init__(self, entropy_w=None, name="basic_disagreement"):
        super().__init__(name=name)
        self.entropy_w = entropy_w
        self.disagreement = None
        self.koleo = None

    def call(self, y_true, y_pred):
        # y_pred shape (B, DIM, P)
        dist = tf.linalg.norm(y_pred[..., None] - y_pred[..., None, :], axis=-3)    # (B, P, P)
        mask = tf.tile(~tf.eye(tf.shape(dist)[-1], dtype=tf.bool)[None],
                       [tf.shape(dist)[0], 1, 1])

        self.disagreement = tf.reduce_mean(dist[mask])
        loss = self.disagreement
        if self.entropy_w is not None:
            self.koleo = koleo(y_pred, axis=-2)
            loss = loss + self.koleo * self.entropy_w
        return loss


class DinoLoss(tf.keras.losses.Loss):
    def __init__(self, entropy_w=0, taus=0.1, taut=0.05, alpha=0.9, sn=False, eps=1e-4, local=True, name="dino_loss"):
        super().__init__(name=name)
        self.entropy_w = entropy_w
        self.eps = eps
        self.sn = sn
        self.alpha = alpha
        self.taus = taus
        self.taut = taut
        self.local = local
        self.center = None
        losses = ['dino']
        if self.entropy_w:
            losses.append('koleo')
        self.monitor = LossMonitors(*losses)

    def update_center(self, embd):
        center = tf.reduce_mean(embd, axis=0, keepdims=True)
        if self.center is None:
            self.center = tf.Variable(center, trainable=False)
        else:
            self.center.assign(self.center * (1-self.alpha) + center * self.alpha)

    def softmax(self, embd, axis=-2, stable=True, tau=1.):
        if stable:
            embd = embd - tf.reduce_max(embd, axis=axis, keepdims=True)
        exps = tf.maximum(tf.math.exp(embd / tau), self.eps)
        softmaxed = exps / tf.reduce_sum(exps, axis=axis, keepdims=True)
        return softmaxed

    def sinkhorn_knopp(self, embd, axis=-2):
        raise NotImplementedError()

    def dino(self, embd):
        s = embd
        t = tf.stop_gradient(embd)
        P = s.shape[-1]

        ps = self.softmax(s, axis=-2, tau=self.taus)  # (B, DIM, P)
        log_ps = tf.math.log(ps)

        center = self.center if self.center is not None else tf.reduce_mean(tf.stop_gradient(t), axis=0, keepdims=True)
        if self.sn:
            pt = self.sinkhorn_knopp(t, axis=-2)
        else:
            pt = self.softmax(t - center, axis=-2, tau=self.taut)

        mean_ce = tf.reduce_sum(tf.reduce_mean(pt[..., None] * log_ps[..., None, :], axis=0), axis=0)  # (P, P)
        mean_path_ce = -tf.reduce_mean(mean_ce[~tf.eye(P, dtype=tf.bool)])
        if self.monitor is not None:
            self.monitor.update_monitor("dino", mean_path_ce)

        return mean_path_ce

    def koleo(self, embd):
        normed = embd / tf.linalg.norm(embd, axis=-2, keepdims=True)    # (B, DIM, P)
        dists = tf.linalg.norm(normed[:, None]-normed[None], axis=-2)   # (B, B, P)
        dists = tf.where(tf.eye(tf.shape(embd)[0], dtype=tf.bool)[..., None],
                         tf.reduce_max(dists),
                         dists)                                         # (B, B, P)
        log_min_dists = tf.math.log(tf.reduce_min(dists, axis=1))       # (B, P)
        return tf.reduce_mean(log_min_dists)

    def call(self, y_true, y_pred):
        if self.local:
            self.update_center(tf.stop_gradient(y_pred))
        loss = self.dino(y_pred)
        if self.entropy_w is not None:
            loss = loss + self.entropy_w * self.koleo(y_pred)
        return loss


class AgreementAndSTD(tf.keras.losses.Loss):
    def __init__(self, std_w=1, corr_w=10, alpha=0.1, l1=False, local=True, name='agreement_and_std'):
        super().__init__(name=name)
        self.std_w = std_w
        self.corr_w = corr_w
        self.monitor = LossMonitors("distance", "var", "cov", name="")
        self.first_moment = None
        self.second_moment = None
        self.cov_est = None
        self.alpha = alpha
        self.l1 = l1
        self.local = local

    def update_estimation(self, x):
        x = tf.stop_gradient(x)
        if self.first_moment is None:
            self.first_moment = tf.Variable(tf.reduce_mean(x, axis=0), trainable=False)
        else:
            self.first_moment.assign(self.first_moment * (1-self.alpha) + tf.reduce_mean(x, axis=0) * self.alpha)

        if self.std_w:
            if self.second_moment is None:
                self.second_moment = tf.Variable(tf.reduce_mean(x**2, axis=0), trainable=False)
            else:
                self.second_moment.assign(self.second_moment * (1-self.alpha) + tf.reduce_mean(x**2, axis=0) * self.alpha)

        if self.corr_w:
            centered = x - self.first_moment[None]  # (B, DIM, P)
            current_cov_est = tf.einsum('bip,bjp->ijp', centered, centered)/tf.cast(tf.shape(x)[0] - 1, dtype=x.dtype)
            if self.cov_est is None:
                self.cov_est = tf.Variable(current_cov_est, trainable=False)
            else:
                self.cov_est.assign(self.cov_est * (1-self.alpha) + current_cov_est * self.alpha)

    def distance(self, embedding):
        # (B, DIM, P)
        diff = embedding[..., None] - tf.stop_gradient(embedding[..., None, :])
        if self.l1:
            dist = tf.math.abs(tf.math.reduce_sum(diff, axis=1))
        else:
            dist = tf.reduce_mean(diff**2, axis=1)
        batch_mean_dist = tf.reduce_mean(dist, axis=0)        # (P, P)
        P = embedding.shape[-1]
        mean_dist = tf.reduce_mean(batch_mean_dist[~tf.eye(P, dtype=tf.bool)])#tf.tensordot(batch_mean_dist, (1-tf.eye(P, dtype=dist.dtype))/tf.cast(P * (P - 1), dist.dtype),  axes=[[0, 1], [0, 1]])
        if self.monitor is not None:
            self.monitor.update_monitor("distance", mean_dist)
        return mean_dist

    def neg_log_std(self, embd):
        variance = tf.reduce_sum((embd - tf.reduce_mean(tf.stop_gradient(embd), axis=0, keepdims=True))**2, axis=0) / tf.cast((tf.shape(embd)[0] - 1), embd.dtype) # (DIM, P, )
        if self.monitor is not None:
            self.monitor.update_monitor("var", tf.reduce_mean(variance))
        out = tf.reduce_mean(-tf.math.log(variance + 1e-6))
        return out

    def local_neg_log_std(self, embd, eps=1e-4):
        # a different calculation, but the same derivative with a stale std
        var = self.second_moment - self.first_moment**2 + eps
        if self.monitor is not None:
            self.monitor.update_monitor("var", tf.reduce_mean(var))
        return -tf.reduce_mean(tf.reduce_sum((embd - self.first_moment)**2, axis=0) / var) / tf.cast(tf.shape(embd)[0] - 1, dtype=var.dtype)

    def decorrelate_nonlocal(self, embd):
        centered = (embd - tf.reduce_mean(tf.stop_gradient(embd), axis=0, keepdims=True))
        co = centered[..., :, None, :] * centered[..., None, :, :]
        cov = tf.reduce_sum(co, axis=(0, )) / tf.cast(tf.shape(embd)[0] - 1, co.dtype) # (DIM, DIM, P)
        cov_sqr = cov ** 2
        mean_cov_sqr = tf.reduce_mean(cov_sqr, axis=-1)
        mean_feat_cov = tf.reduce_mean(mean_cov_sqr[~tf.eye(embd.shape[1], dtype=tf.bool)])
        self.monitor.update_monitor("cov", mean_feat_cov)
        return mean_feat_cov

    def decorrelate(self, embd):
        # If the same sign as the current estimation, minimize square (x_i-\bar{x_i})*(x_j-\bar{x_j}), else maximize
        centered = embd - self.first_moment[None]
        co = centered[..., :, None, :] * centered[..., None, :, :]  # (B, DIM, DIM, P)

        sign_est = tf.math.sign(self.cov_est)  # (DIM, DIM, P)
        sign_cur = tf.math.sign(tf.stop_gradient(co))  # (B, DIM, DIM, P)
        signs_agree = tf.cast(sign_cur == sign_est[None], dtype=co.dtype)

        cov_loss = signs_agree * co ** 2
        mean_over_p_cov_loss = tf.reduce_mean(cov_loss, axis=-1)
        mean_cov_loss = tf.reduce_mean(mean_over_p_cov_loss[~tf.eye(embd.shape[1], dtype=tf.bool)])
        self.monitor.update_monitor("cov", mean_cov_loss)
        return mean_cov_loss

    def call(self, y_true, y_pred):
        if self.local:
            self.update_estimation(y_pred)
        mean_dist = self.distance(embedding=y_pred)

        loss = mean_dist
        if self.std_w:
            std = (self.local_neg_log_std if self.local else self.neg_log_std)(y_pred)
            loss = loss + self.std_w * std
        if self.corr_w:
            corr = (self.decorrelate if self.local else self.decorrelate_nonlocal)(y_pred)
            loss = loss + self.corr_w * corr
        return loss
