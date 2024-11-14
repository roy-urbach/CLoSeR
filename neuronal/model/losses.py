from utils.model.losses import GeneralLossByKey, koleo
import tensorflow as tf

from utils.model.metrics import LossMonitors


class CrossPathwayTemporalContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, *args, a=None, contrast_t=1, start_t=0, temperature=10, cosine=False, stable=True,
                 eps=1e-3, triplet=None, other2self=False, **kwargs):
        super(CrossPathwayTemporalContrastiveLoss, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.a = a
        self.contrast_t = contrast_t
        self.start_t = start_t
        self.cosine = cosine
        self.eps = eps
        self.stable = stable
        self.triplet = triplet
        self.other2self = other2self

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

    def val_func(self, anchor, other, axis=-1):
        if self.triplet is not None:
            val = tf.linalg.norm(anchor - other, axis=axis)
        elif self.cosine:
            assert axis == -1
            val = tf.einsum('...i,...i->...', anchor, other) / self.temperature
        else:
            val = -(tf.linalg.norm(anchor - other, axis=axis) ** 2)
        return val

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)
        n = tf.shape(y_pred)[-1]
        loss = 0.
        if self.cosine:
            y_pred = y_pred / tf.maximum(tf.linalg.norm(tf.stop_gradient(y_pred), axis=-2, keepdims=True), self.eps)

        if self.other2self:
            # (B, T-cont_t-start_t, DIM, P)
            neg_val = self.val_func(y_pred[:, self.start_t+self.contrast_t:],
                                    y_pred[:, self.start_t:-self.contrast_t], axis=-2)

        for i in range(n):
            for j in range(i+1, n):
                if self.a is not None and not self.a[i][j] and not self.a[j][i]:
                    continue
                anchor_i = y_pred[:, self.start_t + self.contrast_t:, :, i]      # (B, T-cont_t-start_t, DIM)
                anchor_j = y_pred[:, self.start_t + self.contrast_t:, :, j]    # (B, T-cont_t-start_t, DIM)
                pos_val = self.val_func(anchor_i, anchor_j)  # (B, T-cont_t-start_t)

                if self.a is None or (self.a is not None and self.a[i][j]):
                    if self.other2self:
                        neg_val_i = neg_val[..., i]
                    else:
                        negative_j = y_pred[:, self.start_t:-self.contrast_t, :, j]  # (B, T-cont_t-start_t, DIM)
                        neg_val_i = self.val_func(anchor_i, negative_j)   # (B, T-cont_t-start_t)
                    loss += tf.reduce_mean(self.loss_func(pos_val, neg_val_i))

                if self.a is None or (self.a is not None and self.a[j][i]):
                    if self.other2self:
                        neg_val_j = neg_val[..., j]
                    else:
                        negative_i = y_pred[:, self.start_t:-self.contrast_t, :, i]  # (B, T-cont_t-start_t, DIM)
                        neg_val_j = self.val_func(anchor_j, negative_i)   # (B, T-cont_t-start_t)
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


class AngularTrajectoryDisagreement(tf.keras.losses.Loss):
    def __init__(self, a=None, entropy_w=None, name='angular_traj_disagreement'):
        super().__init__(name=name)
        self.entropy_w = entropy_w
        self.a = a
        if self.a is not None:
            self.a = self.a / tf.reduce_sum(self.a)

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)
        total_loss = 0.

        n = tf.shape(y_pred)[-1]
        movement_vecs = y_pred[:, 1:] - y_pred[:, :-1]         # (B, T-1, DIM, P)
        movement_vecs_norm = tf.linalg.norm(tf.stop_gradient(movement_vecs), axis=-2, keepdims=True)
        normed_movement_vecs = movement_vecs / movement_vecs_norm
        temporal_cosine_sim = tf.einsum('btdp,btdp->btp',
                                        normed_movement_vecs[:, 1:],
                                        normed_movement_vecs[:, :-1])  # (B, T-2, P)
        t_angles = tf.math.acos(temporal_cosine_sim)    # (B, T-2, P)
        if self.a is None:
            # (B, T-2)
            angle_loss = tf.reduce_sum(tf.where(tf.eye(n) < 1,
                                                tf.math.abs(t_angles[..., None, :] - t_angles[..., None]),
                                                0.),
                                       axis=-1) / tf.cast(n * (n-1), t_angles.dtype)
        else:
            angle_loss = 0.

            for i in range(n):
                for j in range(i+1, n):
                    if self.a is None or (self.a is not None and self.a[i][j]):
                        angle_loss += tf.math.abs(t_angles[..., i] - tf.stop_gradient(t_angles[..., j])) * self.a[i][j]
                    if self.a is None or (self.a is not None and self.a[j][i]):
                        angle_loss += tf.math.abs(tf.stop_gradient(t_angles[..., i]) - t_angles[..., j]) * self.a[i][j]
        total_loss += tf.reduce_mean(angle_loss)

        if self.entropy_w is not None:
            total_loss += koleo(t_angles, None) * self.entropy_w

        return total_loss


class VectorTrajectoryDisagreement(tf.keras.losses.Loss):
    def __init__(self, a=None, entropy_w=0, name='vector_traj_disagreement'):
        super().__init__(name=name)
        self.entropy_w = entropy_w
        self.a = a
        if self.a is not None:
            self.a = self.a / tf.reduce_sum(self.a)

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)
        n = tf.shape(y_pred)[-1]

        total_loss = 0.

        movement_vecs = y_pred[:, 1:] - y_pred[:, :-1]  # (B, T-1, DIM, P)
        movement_vecs_norm = tf.linalg.norm(tf.stop_gradient(movement_vecs), axis=-2, keepdims=True)
        normed_movement_vecs = movement_vecs / movement_vecs_norm

        if self.a is None:
            # (B, T-1)
            cross_path_cosine_sim = tf.einsum('btdp,btdk->btp',
                                              normed_movement_vecs[..., None],
                                              normed_movement_vecs[..., None, :])  # (B, T-1, P, P)
            vector_loss = tf.where(tf.eye(n) < 1, 1. - cross_path_cosine_sim, 0.) / tf.cast(n * (n - 1), cross_path_cosine_sim.dtype)
        else:
            raise NotImplementedError()  # TODO: if it looks promising

        total_loss += tf.reduce_mean(vector_loss)

        if self.entropy_w is not None:
            total_loss += koleo(y_pred, -2) * self.entropy_w

        return total_loss


class ContinuousLoss(tf.keras.losses.Loss):
    def __init__(self, softmax=False, l1=False, cosine=False, mse=False, continuous_w=1., entropy_w=None, crosspath_w=None, nonlocal_w=None, nonlocal_kwargs={}, eps=None,
                 contrast_in_time_w=None, contrast_in_time_kwargs={}, continuous_kwargs={}, push_corr_w=None, predictive_w=None,
                 adversarial_w=None, adversarial_pred_w=None, pe_w=None, pe_push_w=None, neg_log_std_w=None, adversarial_kwargs={},
                 log_dist=False, monitor=False, centering=False, name='continuous_loss'):
        super().__init__(name=name)
        self.l1 = l1
        self.cosine = cosine
        self.mse = mse
        self.softmax = softmax
        self.update_running_mean = centering or neg_log_std_w is not None
        self.centering = centering
        self.running_mean = None
        self.continuous_w = continuous_w
        self.continuous_kwargs = continuous_kwargs
        self.entropy_w = entropy_w
        self.neg_log_std_w = neg_log_std_w
        self.crosspath_w = crosspath_w
        self.nonlocal_w = nonlocal_w
        self.nonlocal_kwargs = nonlocal_kwargs
        self.eps = eps if eps is not None else 0.
        self.contrast_in_time_w = contrast_in_time_w
        self.contrast_in_time_kwargs = contrast_in_time_kwargs
        self.push_corr_w = push_corr_w
        self.adversarial_w = adversarial_w
        self.pe_w = pe_w
        self.pe_push_w = pe_push_w
        if adversarial_pred_w is not None:
            self.adversarial_pred_w = adversarial_pred_w
        elif adversarial_w is not None:
            self.adversarial_pred_w = adversarial_w
        elif predictive_w is not None:
            self.adversarial_pred_w = predictive_w
        elif pe_w is not None:
            self.adversarial_pred_w = pe_w
        elif pe_push_w is not None:
            self.adversarial_pred_w = pe_push_w
        else:
            self.adversarial_pred_w = None
        self.predictive_w = predictive_w
        self.adversarial_kwargs = adversarial_kwargs
        self.P = None
        self.T = None
        self.DIM = None
        self.log_dist = log_dist

        if monitor:
            losses = []
            if continuous_w:
                losses.append('continuous')
            if entropy_w:
                losses.append("koleo")
            if neg_log_std_w:
                losses.append("std")
            if crosspath_w:
                losses.append("distlocal_inter")
            if nonlocal_w:
                losses.append("nonlocal_inter")
            if contrast_in_time_w:
                losses.append("templocal_inter")
            if push_corr_w:
                losses.append("push_corr")
            if adversarial_w:
                losses.append("adv_inter")
            if pe_w:
                losses.append("pe_weighted_cross_distance")
            if pe_push_w:
                losses.append("pe_weighted_push_cross_distance")
            if adversarial_w or predictive_w or pe_w:
                losses.append("pred_distance")

            self.monitor = LossMonitors(*losses, name="ContinuousLossMonitor")
        else:
            self.monitor = None

    def continuous_disagreement(self, embd, stopgrad=False):
        # embd shape (B, T, DIM, P)
        dist = tf.maximum(self.dist2logdist(tf.linalg.norm(embd[:, 1:] - (tf.stop_gradient(embd[:, :-1]) if stopgrad else embd[:, :-1]), axis=-2)), self.eps)  # (B, T-1, P)
        disagreement = tf.reduce_mean(dist)
        if self.monitor is not None:
            self.monitor.update_monitor("continuous", disagreement)
        return disagreement

    def crosspath_disagreement(self, embd):
        dist = self.dist2logdist(tf.linalg.norm(embd[..., None] - embd[..., None, :], axis=-3))  # (B, T, P, P)
        mask = tf.tile(~tf.eye(tf.shape(dist)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(dist)[0], tf.shape(dist)[1], 1, 1])

        disagreement = tf.reduce_mean(dist[mask])
        if self.monitor is not None:
            self.monitor.update_monitor("distlocal_inter", disagreement)
        return disagreement

    def nonlocal_contrast(self, embd, temperature=1., eps=1e-4):
        # y_pred shape (B, T, DIM, P)
        dists = tf.maximum(self.dist2logdist(tf.linalg.norm(embd[:, None, ..., None] - embd[None, ..., None, :],
                                                            axis=-3)), self.eps)    # (B, B, T, P, P)
        # self.calculate_cross_path_agreement(dists)
        sim = tf.maximum(tf.math.exp(-(dists**2)/temperature), eps) # The bug is here
        log_z = tf.reduce_logsumexp(sim, axis=1)
        all_pair_loss = log_z - tf.math.log(sim[tf.eye(tf.shape(sim)[0]) != 0])   # -log(pi)=-log(simii/zi)=-log(simii)+log(zi)
        mask = tf.tile(~tf.eye(tf.shape(all_pair_loss)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(all_pair_loss)[0], tf.shape(all_pair_loss)[1], 1, 1])
        relevant_pairs_loss = all_pair_loss[mask]
        loss = tf.reduce_mean(relevant_pairs_loss)
        if self.monitor is not None:
            self.monitor.update_monitor("nonlocal_inter", tf.reduce_mean(tf.math.exp(-relevant_pairs_loss)))
        return loss

    def nonlocal_contrast_without_bug(self, embd, temperature=10.):
        # y_pred shape (B, T, DIM, P)
        dists = self.dist2logdist(tf.maximum(tf.linalg.norm(embd[:, None, ..., None] - embd[None, ..., None, :],
                                                            axis=-3), self.eps))    # (B, B, T, P, P)
        # self.calculate_cross_path_agreement(dists)
        sim_logits = -(dists**2)/temperature
        log_z = tf.reduce_logsumexp(sim_logits, axis=1)
        all_pair_loss = log_z - sim_logits[tf.eye(tf.shape(sim_logits)[0]) != 0]   # -log(p_i)=-log(sim_ii/z_i)=-log(sim_ii)+log(z_i)=logsumexp_j(sim_logits_ij)-sim_logits_ii
        mask = tf.tile(~tf.eye(tf.shape(all_pair_loss)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(all_pair_loss)[0], tf.shape(all_pair_loss)[1], 1, 1])
        relevant_pairs_loss = all_pair_loss[mask]
        loss = tf.reduce_mean(relevant_pairs_loss)
        if self.monitor is not None:
            self.monitor.update_monitor("nonlocal_inter", tf.reduce_mean(tf.math.exp(-relevant_pairs_loss)))
        return loss

    def contrast_in_time(self, embd, temperature=10., eps=1e-4, triplet=False):
        # embd shape (B, T, DIM, P)
        b = tf.shape(embd)[0]
        t = tf.shape(embd)[1]
        p = tf.shape(embd)[3]

        pos_sub = embd[:, 1:, ..., None] - embd[:, 1:, ..., None, :]    # (B, T-1, DIM, P, P)
        neg_sub = embd[:, 1:, ..., None] - embd[:, :-1, ..., None, :]   # (B, T-1, DIM, P, P)
        pos_dist = self.dist2logdist(tf.maximum(tf.linalg.norm(pos_sub, axis=-3), eps))
        neg_dist = self.dist2logdist(tf.maximum(tf.linalg.norm(neg_sub, axis=-3), eps))

        if triplet:
            all_pair_loss = tf.maximum(pos_dist - neg_dist, 0.)
        else:
            dist_to_logit = lambda dist: -(dist**2) / temperature
            pos_logit = dist_to_logit(pos_dist)   # (B, T-1, P, P)
            neg_logit = dist_to_logit(neg_dist)   # (B, T-1, P, P)

            log_z = tf.reduce_logsumexp(tf.stack([pos_logit, neg_logit], axis=-1), axis=-1)     # (B, T-1, P, P)
            all_pair_loss = log_z - pos_logit

        mask = tf.tile(~tf.eye(p, dtype=tf.bool)[None, None], [b, t-1, 1, 1])
        relevant_pairs_loss = all_pair_loss[mask]
        loss = tf.reduce_mean(relevant_pairs_loss)

        if self.monitor is not None:
            self.monitor.update_monitor("templocal_inter", tf.reduce_mean(tf.math.exp(-relevant_pairs_loss)))

        return loss

    def dkl(self, arr1, arr2, axis=-1):
        return tf.reduce_sum(arr1 * tf.math.log(arr2), axis=axis)

    def jsd(self, arr1, arr2, sqrt=False, axis=-1):
        m = (arr1 + arr2) / 2
        div = (self.dkl(arr1, m, axis=axis) + self.dkl(arr2, m, axis=axis)) / 2
        if sqrt:
            return tf.sqrt(div)
        else:
            return div

    def distance(self, arr1, arr2, axis, log=False, use_eps=True, hard_log=False):
        if self.softmax:
            arr1 = tf.nn.softmax(arr1, axis=axis)
            arr2 = tf.nn.softmax(arr2, axis=axis)
            dist = self.jsd(arr1, arr2, axis=axis)
        elif self.l1:
            dist = tf.reduce_sum(tf.abs(arr1 - arr2), axis=axis)
        elif self.cosine:
            norm = lambda arr: arr / tf.linalg.norm(tf.stop_gradient(arr), axis=axis, keepdims=True)
            cosine_sim = tf.reduce_sum(norm(arr1) * norm(arr2), axis=axis)
            dist = 1-cosine_sim/2
        elif self.mse:
            dist = tf.reduce_mean((arr1 - arr2)**2, axis=axis)
        else:
            dist = tf.linalg.norm(arr1 - arr2, axis=axis)
        if self.eps is not None and use_eps:
            dist = tf.maximum(dist, self.eps)
        if log:
            dist = self.dist2logdist(dist)
        if hard_log and not self.log_dist:
            dist = tf.math.log(dist)
        return dist

    def dist2logdist(self, dist):
        if self.log_dist:
            return tf.math.log(dist)
        else:
            return dist

    def _predictor_loss(self, embd, pred, axis=-2, return_mat=False):
        pred_dist = self.distance(tf.stop_gradient(embd), pred, axis=axis)
        predictivity_loss = tf.reduce_mean(pred_dist)
        if self.monitor is not None:
            self.monitor.update_monitor("pred_distance", predictivity_loss)
        return (predictivity_loss, pred_dist) if return_mat else predictivity_loss

    def adversarial_loss(self, embd, pred_embd, temperature=10., stop_grad_j=False, **kwargs):
        # (B, T, DIM, P)

        last_embd = embd[:, -1]     # (B, DIM, P)
        last_pred_embd = pred_embd[:, -1]
        last_embd_j = last_embd
        if stop_grad_j:
            last_embd_j = tf.stop_gradient(last_embd_j)

        # (B, P)
        predictivity_loss = self._predictor_loss(last_embd, last_pred_embd, axis=-2)

        # (B, P, P)
        pos_dist_sqr_normed = (self.distance(last_embd[..., None], last_embd_j[..., None, :],
                                             axis=-3, log=False)**2) / temperature

        # (B, P)
        neg_dist_sqr_normed = (self.distance(last_embd_j, tf.stop_gradient(last_pred_embd),
                                             axis=-2, log=False)**2) / temperature

        pe_contrast_loss = pos_dist_sqr_normed + tf.reduce_logsumexp(-tf.stack([pos_dist_sqr_normed,
                                                                                tf.tile(neg_dist_sqr_normed[..., None, :], [1, self.P, 1])],
                                                                                axis=-1),
                                                                     axis=-1)  # (B, P, P)
        mask = tf.tile(~tf.eye(self.P, dtype=tf.bool)[None], [tf.shape(pe_contrast_loss)[0], 1, 1])
        pe_contrast_relevant = pe_contrast_loss[mask]
        pe_contrast_loss = tf.reduce_mean(pe_contrast_relevant)

        if self.monitor is not None:
            self.monitor.update_monitor("adv_inter", tf.reduce_mean(tf.math.exp(-pe_contrast_relevant)))

        return self.adversarial_pred_w * predictivity_loss + self.adversarial_w * pe_contrast_loss

    def predictive_loss(self, embd, pred_embd, pe=1):
        # (B, T, DIM, P)

        last_embd = embd[:, -1]  # (B, DIM, P)
        last_pred_embd = pred_embd[:, -1]

        # (B, P)
        predictivity_loss = self._predictor_loss(last_embd, last_pred_embd, axis=-2)

        pred_pe = self.distance(last_embd, tf.stop_gradient(last_pred_embd), log=True, axis=-2)
        pe_loss = (tf.reduce_mean(pred_pe) - pe)**2

        return self.adversarial_pred_w * predictivity_loss + self.predictive_w * pe_loss

    def update_center(self, embd):
        if self.running_mean is None:
            self.running_mean = tf.Variable(tf.reduce_mean(tf.stop_gradient(embd), axis=0), trainable=False)
        else:
            self.running_mean.assign(self.running_mean * 0.9 + tf.reduce_mean(tf.stop_gradient(embd), axis=0) * 0.1)

    def prediction_error_loss(self, embd, pred_embd, _max=1, push_pe_diff_min=None):
        b = tf.shape(embd)[0]

        last_embd = embd[:,-1]  # (B, DIM, P)
        last_pred_embd = pred_embd[:, -1]
        mask = tf.tile(~tf.eye(self.P, dtype=bool)[None], [b, 1, 1])
        shape = [b, self.P, self.P-1]

        mean_pe, pe = self._predictor_loss(last_embd, last_pred_embd, axis=-2, return_mat=True)     # (1, ), (B, P)

        loss = self.adversarial_pred_w * mean_pe

        if self.pe_w is not None or self.pe_push_w is not None:

            other_embd = tf.stop_gradient(last_embd)
            if self.centering:
                other_embd = other_embd - self.running_mean[None]

            dist = self.distance(last_embd[..., None], other_embd[..., None, :],
                                 axis=-3, log=True)  # (B, P, P)
            dist_no_diag = tf.reshape(dist[mask], shape)

            pe_diff = tf.stop_gradient(pe[..., None] - pe[:, None])     # (B, P, P)
            pe_diff_no_diag = tf.maximum(tf.reshape(pe_diff[mask]**2, shape), self.eps)
            exps = tf.math.exp(-pe_diff_no_diag)

            if self.pe_w is not None:
                z = tf.reduce_sum(exps, axis=-1, keepdims=True)
                w = exps / z
                pe_weighted_cross = tf.tensordot(w, dist_no_diag, [[0,1,2], [0,1,2]]) / tf.cast(self.P * b, dtype=exps.dtype)
                if self.monitor is not None:
                    self.monitor.update_monitor("pe_weighted_cross_distance", pe_weighted_cross)

                loss = loss + self.pe_w * pe_weighted_cross

            if self.pe_push_w is not None:
                push_exps = 1/exps
                push_z = tf.reduce_sum(push_exps, axis=-1, keepdims=True)
                push_w = push_exps / push_z
                if _max is not None:
                    dist_no_diag = tf.minimum(dist_no_diag, _max)
                if push_pe_diff_min is not None:
                    push_w = tf.where(tf.math.abs(pe_diff_no_diag) < push_pe_diff_min, 0., push_w)
                push_pe_weighted_cross = tf.tensordot(push_w, dist_no_diag, [[0, 1, 2],
                                                                             [0, 1, 2]]) / tf.cast(self.P * b, dtype=exps.dtype)
                if self.monitor is not None:
                    self.monitor.update_monitor("pe_weighted_push_cross_distance", push_pe_weighted_cross)
                loss = loss + self.pe_push_w * push_pe_weighted_cross

        return loss

    def nonlocal_push(self, embd, last_step=True, exp=True, temperature=10.):
        if last_step:
            embd = embd[..., -1:, :]
        inenc_dists = self.distance(embd[:, None], embd[None], axis=-2, log=True)  # (B, B, T, P)

        b = tf.shape(inenc_dists)[0]

        inenc_dists = tf.reshape(inenc_dists[tf.tile(~tf.eye(b, dtype=tf.bool)[..., None, None], [1, 1, self.T, self.P])],
                                 (b - 1, b, self.T, self.P))
        if exp:
            inenc_dists = tf.math.exp(-inenc_dists**2 / temperature)

        corr_size = float(b*(b-1))
        _mean = tf.math.reduce_mean(tf.stop_gradient(inenc_dists), axis=(0, 1))  # (T, P, )
        _std = tf.math.reduce_std(tf.stop_gradient(inenc_dists), axis=(0, 1))  # (T, P, )
        mult = tf.einsum('ijtn,ijtm->tnm', inenc_dists, inenc_dists)  # (T, P, P)

        temporal_mean_correlation = tf.reduce_mean((mult / corr_size - _mean[:, None] * _mean[..., None]) / (_std[:, None] * _std[..., None]), axis=0)

        mean_corr = tf.reduce_mean(temporal_mean_correlation[~tf.eye(self.P, dtype=tf.bool)])

        if self.monitor is not None:
            self.monitor.update_monitor("push_corr", mean_corr)
        loss = mean_corr**2
        return loss

    def koleo(self, embd, eps=1e-5):
        b = tf.shape(embd)[0]
        dist = tf.maximum(self.distance(embd[None], embd[:, None], axis=-2, log=False, use_eps=False), eps)
        shape_without_b = dist.get_shape().as_list()[2:]
        mask = tf.tile(tf.reshape(tf.eye(b) < 1, [b] * 2 + [1] * len(shape_without_b)), [1] * 2 + shape_without_b)
        log_dist = tf.where(mask, dist, tf.reduce_max(dist))

        min_dist = tf.reduce_min(log_dist, axis=1)
        out = -tf.reduce_mean(tf.math.log(min_dist))

        if self.monitor is not None:
            self.monitor.update_monitor("koleo", out)

        return out

    def neg_log_std(self, embd):
        last_embd = embd[:,-1]  # (B, DIM, P)
        deviation = (last_embd - self.running_mean[None])**2
        if self.monitor is not None:
            self.monitor.update_monitor("std", tf.reduce_mean(deviation))
        out = tf.reduce_mean(-tf.math.log(tf.maximum(deviation, 1e-4)))
        return out

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)

        if self.adversarial_pred_w is not None:
            DIM = tf.shape(y_pred)[-2]//2
            embd = y_pred[..., :DIM, :]
            pred_embd = y_pred[..., DIM:, :]

        else:
            embd = y_pred
            pred_embd = None

        self.T = tf.shape(embd)[1]
        self.DIM = tf.shape(embd)[2]
        self.P = tf.shape(embd)[3]

        if self.update_running_mean:
            self.update_center(embd[:, -1])

        loss = 0.
        if self.continuous_w is not None:
            loss = loss + self.continuous_w * self.continuous_disagreement(embd, **self.continuous_kwargs)
        if self.crosspath_w is not None:
            loss = loss + self.crosspath_w * self.crosspath_disagreement(embd)
        if self.entropy_w is not None:
            loss = loss + self.entropy_w * self.koleo(embd)
        if self.neg_log_std_w is not None:
            loss = loss + self.neg_log_std_w * self.neg_log_std(embd)
        if self.nonlocal_w is not None:
            use_buggy_version = self.nonlocal_kwargs.pop("bug", True)
            if use_buggy_version:
                nonlocal_contrast = self.nonlocal_contrast
            else:
                nonlocal_contrast = self.nonlocal_contrast_without_bug
            loss = loss + self.nonlocal_w * nonlocal_contrast(embd, **self.nonlocal_kwargs)
        if self.contrast_in_time_w is not None:
            loss = loss + self.contrast_in_time_w * self.contrast_in_time(embd, **self.contrast_in_time_kwargs)
        if self.adversarial_w is not None:
            loss = loss + self.adversarial_loss(embd, pred_embd, **self.adversarial_kwargs)
        if self.predictive_w is not None:
            loss = loss + self.predictive_loss(embd, pred_embd, **self.adversarial_kwargs)
        if self.pe_w is not None or self.pe_push_w is not None:
            loss = loss + self.prediction_error_loss(embd, pred_embd, **self.adversarial_kwargs)
        if self.push_corr_w is not None:
            loss = loss + self.push_corr_w * self.nonlocal_push(embd)
        return loss


class BasicDisagreement(tf.keras.losses.Loss):
    def __init__(self, entropy_w=None, name="basic_disagreement"):
        super().__init__(name=name)
        self.entropy_w = entropy_w
        self.disagreement = None
        self.koleo = None

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)
        dist = tf.linalg.norm(y_pred[..., None] - y_pred[..., None, :], axis=-3)    # (B, T, P, P)
        mask = tf.tile(~tf.eye(tf.shape(dist)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(dist)[0], tf.shape(dist)[1], 1, 1])

        self.disagreement = tf.reduce_mean(dist[mask])
        loss = self.disagreement
        if self.entropy_w is not None:
            self.koleo = koleo(y_pred, axis=-2)
            loss = loss + self.koleo * self.entropy_w
        return loss

    def get_metrics(self):
        return {"disagreement": self.disagreement, "koleo": self.koleo}


class DinoLoss(tf.keras.losses.Loss):
    def __init__(self, entropy_w=0.1, sn=False, eps=1e-3, name="dino_loss"):
        super().__init__(name=name)
        self.entropy_w = entropy_w
        self.eps = eps
        self.cross_entropy = None
        self.koleo = None
        self.sn = sn

    def softmax(self, embd, axis=-2, stable=True):
        if stable:
            embd = embd - tf.reduce_max(embd, axis=axis, keepdims=True)
        exps = tf.maximum(tf.math.exp(embd), self.eps)
        softmaxed = exps / tf.reduce_sum(exps, axis=axis, keepdims=True)
        return softmaxed

    def sinkhorn_knopp(self, embd, axis=-2):
        raise NotImplementedError()

    def call(self, y_true, y_pred):
        ps = self.softmax(y_pred, axis=-2)  # (B, T, DIM, P)
        if self.sn:
            pt = self.sinkhorn_knopp(tf.stop_gradient(y_pred), axis=-2)
        else:
            pt = tf.stop_gradient(ps)

        log_ps = tf.math.log(ps)

        pair_ce = -tf.einsum('btdp,btdk->btpk', pt, log_ps)     # (B, T, P, P)
        mask = tf.tile(~tf.eye(tf.shape(pair_ce)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(pair_ce)[0], tf.shape(pair_ce)[1], 1, 1])

        loss = self.cross_entropy = tf.reduce_mean(pair_ce[mask])
        if self.entropy_w is not None:
            self.koleo = koleo(y_pred, axis=-2)
            loss = loss + self.koleo * self.entropy_w
        return loss

    def get_metrics(self):
        return {"cross_entropy": self.cross_entropy, "koleo": self.koleo}


class NonLocalContrastive(tf.keras.losses.Loss):
    def __init__(self, temperature=10., eps=1e-4, name='nonlocal_contrastive'):
        super().__init__(name=name)
        self.temperature = temperature
        self.cross_path_agreement = None
        self.eps = eps

    # def calculate_cross_path_agreement(self, dists):
    #     argmin = tf.math.argmin(dists, axis=1)
    #     where = argmin == tf.range(tf.shape(dists)[0])[..., None, None, None]
    #     self.cross_path_agreement = tf.reduce_mean(tf.cast(where, dtype=dists.dtype), axis=0)

    def call(self, y_true, y_pred):
        # y_pred shape (B, T, DIM, P)

        # (B, B, T, P, P)
        dists = tf.maximum(tf.linalg.norm(y_pred[:, None, ..., None] - y_pred[None, ..., None, :], axis=-3), self.eps)

        # self.calculate_cross_path_agreement(dists)
        sim = tf.maximum(tf.math.exp(-(dists**2)/self.temperature), self.eps)
        log_z = tf.reduce_logsumexp(sim, axis=1)
        all_pair_loss = log_z - tf.math.log(sim[tf.eye(tf.shape(sim)[0]) != 0])   # -log(pi)=-log(simii/zi)=-log(simii)+log(zi)
        mask = tf.tile(~tf.eye(tf.shape(all_pair_loss)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(all_pair_loss)[0], tf.shape(all_pair_loss)[1], 1, 1])
        relevant_pairs_loss = all_pair_loss[mask]
        return tf.reduce_mean(relevant_pairs_loss)

    def get_metrics(self):
        return {"cross_path_agreement": self.cross_path_agreement}


class LPL(tf.keras.losses.Loss):
    def __init__(self, std_w=1, corr_w=10, cross_w=None, cov_sqr=True, alpha=0.1, l1=False, pe_w=None, local=True, eps=1e-4, name='agreement_and_std'):
        super().__init__(name=name)
        self.eps = eps
        self.std_w = std_w
        self.corr_w = corr_w
        self.cov_sqr = cov_sqr
        self.pe_w =pe_w
        losses = ['cont_mse', 'var', 'cov']
        self.first_moment = None
        self.second_moment = None
        self.alpha = alpha
        self.l1 = l1
        self.local = local
        self.cross_w = cross_w
        if cross_w:
            losses.append("cross")
        if self.pe_w:
            losses.append("pe_cross")
        self.monitor = LossMonitors(*losses, name="")

    def update_estimation(self, x):
        x = tf.stop_gradient(x)
        if self.first_moment is None:
            self.first_moment = tf.Variable(tf.reduce_mean(x, axis=0), trainable=False)
        else:
            self.first_moment = self.first_moment.assign(self.first_moment * (1-self.alpha) + tf.reduce_mean(x, axis=0) * self.alpha)

        if self.second_moment is None:
            self.second_moment = tf.Variable(tf.reduce_mean(x**2, axis=0), trainable=False)
        else:
            self.second_moment = self.second_moment.assign(self.second_moment * (1-self.alpha) + tf.reduce_mean(x**2, axis=0) * self.alpha)

    def continuous_loss(self, prev_embd, embd):
        # (B, DIM, P)
        diff = embd - tf.stop_gradient(prev_embd)
        if self.l1:
            dist = tf.math.abs(tf.math.reduce_sum(diff, axis=1))
        else:
            dist = tf.reduce_mean(diff**2, axis=1)
        mean_dist = tf.reduce_mean(dist)
        if self.monitor is not None:
            self.monitor.update_monitor("cont_mse", mean_dist)
        return mean_dist

    def neg_log_std(self, embd):
        if self.local:
            # a different calculation, but the same derivative with a stale std
            var = self.second_moment - self.first_moment ** 2 + self.eps
            if self.monitor is not None:
                self.monitor.update_monitor("var", tf.reduce_mean(var))
            return -tf.reduce_mean(tf.reduce_sum((embd - self.first_moment) ** 2, axis=0) / var) / tf.cast(
                tf.shape(embd)[0] - 1, dtype=var.dtype)
        else:
            var = tf.reduce_sum((embd - tf.reduce_mean(embd, axis=0, keepdims=True))**2, axis=0) / tf.cast((tf.shape(embd)[0] - 1), embd.dtype) # (DIM, P, )
            if self.monitor is not None:
                self.monitor.update_monitor("var", tf.reduce_mean(var))
            out = -tf.reduce_mean(tf.math.log(var + self.eps))
        return out

    def decorrelate(self, embd):
        if self.local:
            raise NotImplementedError()
            # centered = (embd - self.first_moment[None])**2
            # co = centered[..., :, None, :] * centered[..., None, :, :]    # (B, DIM, DIM, P)
            # cov = tf.reduce_sum(cov, axis=(0,)) / (tf.shape(embd)[0] - 1) # (DIM, DIM, P)
            # mean_cov_sqr = tf.reduce_mean(cov_sqr, axis=-1)  # (DIM, DIM)
            # mean_feat_cov = tf.reduce_mean(mean_cov_sqr[~tf.eye(embd.shape[1], dtype=tf.bool)])
        else:
            centered = (embd - tf.reduce_mean(embd, axis=0, keepdims=True))
            co = centered[..., :, None, :] * centered[..., None, :, :]
            cov = tf.reduce_sum(co, axis=(0,)) / tf.cast(tf.shape(embd)[0] - 1, co.dtype)  # (DIM, DIM, P)
            if self.cov_sqr:
                cov = cov ** 2
            else:
                cov = tf.math.abs(cov)
            mean_cov_sqr = tf.reduce_mean(cov, axis=-1)
            mean_feat_cov = tf.reduce_mean(mean_cov_sqr[~tf.eye(embd.shape[1], dtype=tf.bool)])
        self.monitor.update_monitor("cov", mean_feat_cov)
        return mean_feat_cov

    def crossdist(self, embd):
        # (B, DIM, P)
        mse = tf.reduce_mean((embd[..., None] - tf.stop_gradient(embd[..., None, :]))**2, axis=(0,1))  # (P, P)
        mse_nondiag = mse[~tf.eye(embd.shape[-1], dtype=tf.bool)]
        mean_mse = tf.reduce_mean(mse_nondiag)
        self.monitor.update_monitor("cross", mean_mse)
        return mean_mse

    def pe_weighted_crossdist(self, embd, pe):
        pe_diff = (pe[..., None] - pe[..., None, :])**2
        # pe_diff = pe_diff - tf.reduce_min(pe_diff, axis=-1, keepdims=True)    # diagonal is min
        exp_pe = tf.linalg.set_diagonal(tf.exp(-pe_diff), 0.)
        z_pe = tf.reduce_sum(exp_pe, axis=-1, keepdims=True)
        pe_weights = exp_pe / z_pe  # (B, P, P)

        dist = tf.reduce_mean((embd[..., None] - tf.stop_gradient(embd[..., None, :]))**2, axis=1)  # (B, P, P)
        pe_weighted_dist = tf.tensordot(pe_weights, dist, axis=[[0, 1, 2],
                                                                [0, 1, 2]]) / tf.cast(tf.shape(embd)[0], dtype=embd.dtype)
        self.monitor.update_monitor("pe_cross", pe_weighted_dist)
        return pe_weighted_dist

    def call(self, y_true, y_pred):
        # (B, T, DIM, P)
        embd = y_pred[:, -1]
        prev_embd = tf.stop_gradient(y_pred[:, -2])
        if self.pe_w:
            pred_start = embd.shape[-2]//2
            pred = embd[..., pred_start:, :]
            embd = embd[..., :pred_start, :]
            pe = tf.reduce_mean((tf.stop_gradient(embd) - pred)**2, axis=-2)  # (B, P)
            prev_embd = prev_embd[..., :pred_start, :]

        if not self.local:
            self.update_estimation(embd)
        loss = self.continuous_loss(prev_embd, embd)
        if self.std_w:
            loss = loss + self.std_w * self.neg_log_std(embd)
        if self.corr_w:
            loss = loss + self.corr_w * self.decorrelate(embd)
        if self.cross_w:
            loss = loss + self.cross_w * self.crossdist(embd)
        if self.pe_w:
            loss = loss + self.pe_w * self.pe_weighted_crossdist(embd, tf.stop_gradient(pe))

            loss = loss + tf.reduce_mean(pe)

        return loss

