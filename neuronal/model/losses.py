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
    def __init__(self, continuous_w=1., entropy_w=None, crosspath_w=None, nonlocal_w=None, nonlocal_kwargs={}, eps=None,
                 contrast_in_time_w=None, contrast_in_time_kwargs={}, push_corr_w=None, predictive_w=None,
                 adversarial_w=None, adversarial_pred_w=None, pe_w=None, adversarial_kwargs={}, monitor=False, name='continuous_loss'):
        super().__init__(name=name)
        self.continuous_w = continuous_w
        self.entropy_w = entropy_w
        self.crosspath_w = crosspath_w
        self.nonlocal_w = nonlocal_w
        self.nonlocal_kwargs = nonlocal_kwargs
        self.eps = eps if eps is not None else 0.
        self.contrast_in_time_w = contrast_in_time_w
        self.contrast_in_time_kwargs = contrast_in_time_kwargs
        self.push_corr_w = push_corr_w
        self.adversarial_w = adversarial_w
        self.pe_w = pe_w
        if adversarial_pred_w is not None:
            self.adversarial_pred_w = adversarial_pred_w
        elif adversarial_w is not None:
            self.adversarial_pred_w = adversarial_w
        elif predictive_w is not None:
            self.adversarial_pred_w = predictive_w
        elif pe_w is not None:
            self.adversarial_pred_w = pe_w
        else:
            self.adversarial_pred_w = None
        self.predictive_w = predictive_w
        self.adversarial_kwargs = adversarial_kwargs
        self.P = None
        self.T = None
        self.DIM = None

        if monitor:
            losses = []
            if continuous_w:
                losses.append('continuous')
            if entropy_w:
                losses.append("koleo")
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
                losses.append("pe_weighted_cross_distance", )
            if adversarial_w or predictive_w or pe_w:
                losses.append("pred_distance")

            self.monitor = LossMonitors(*losses, name="ContinuousLossMonitor")
        else:
            self.monitor = None

    def continuous_disagreement(self, embd):
        dist = tf.maximum(tf.linalg.norm(embd[:, 1:] - embd[:, :-1], axis=-2), self.eps)  # (B, T-1, P)
        disagreement = tf.reduce_mean(dist)
        if self.monitor is not None:
            self.monitor.update_monitor("continuous", disagreement)
        return disagreement

    def crosspath_disagreement(self, embd):
        dist = tf.linalg.norm(embd[..., None] - embd[..., None, :], axis=-3)  # (B, T, P, P)
        mask = tf.tile(~tf.eye(tf.shape(dist)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(dist)[0], tf.shape(dist)[1], 1, 1])

        disagreement = tf.reduce_mean(dist[mask])
        if self.monitor is not None:
            self.monitor.update_monitor("distlocal_inter", disagreement)
        return disagreement

    def nonlocal_contrast(self, embd, temperature=1., eps=1e-4):
        # y_pred shape (B, T, DIM, P)
        dists = tf.maximum(tf.linalg.norm(embd[:, None, ..., None] - embd[None, ..., None, :], axis=-3), self.eps)    # (B, B, T, P, P)
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
        dists = tf.maximum(tf.linalg.norm(embd[:, None, ..., None] - embd[None, ..., None, :], axis=-3), self.eps)    # (B, B, T, P, P)
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
        pos_dist = tf.maximum(tf.linalg.norm(pos_sub, axis=-3), eps)
        neg_dist = tf.maximum(tf.linalg.norm(neg_sub, axis=-3), eps)

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

    def _predictor_loss(self, embd, pred, axis=-2, return_mat=False):
        pred_dist = tf.maximum(tf.linalg.norm(tf.stop_gradient(embd) - pred, axis=axis), self.eps)
        predictivity_loss = tf.reduce_mean(pred_dist)
        if self.monitor is not None:
            self.monitor.update_monitor("pred_distance", predictivity_loss)
        return (predictivity_loss, pred_dist) if return_mat else pred_dist

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
        pos_dist_sqr_normed = (tf.maximum(tf.linalg.norm(last_embd[..., None] - last_embd_j[..., None, :], axis=-3), self.eps)**2) / temperature

        # (B, P)
        neg_dist_sqr_normed = (tf.maximum(tf.linalg.norm(last_embd_j - tf.stop_gradient(last_pred_embd), axis=-2), self.eps)**2) / temperature

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

        pred_pe = tf.maximum(tf.linalg.norm(last_embd - tf.stop_gradient(last_pred_embd), axis=-2), self.eps)
        pe_loss = (tf.reduce_mean(pred_pe) - pe)**2

        return self.adversarial_pred_w * predictivity_loss + self.predictive_w * pe_loss

    def prediction_error_loss(self, embd, pred_embd, _max=1):
        b = tf.shape(embd)[0]

        last_embd = embd[:,-1]
        last_pred_embd = pred_embd[:, -1]
        dist = tf.linalg.norm(last_embd[..., None] - tf.stop_gradient(last_pred_embd[..., None, :]), axis=-3)  # (B, T, P, P)
        mean_pe, pe = self._predictor_loss(last_embd, last_pred_embd, axis=-2, return_mat=True)     # (B, P)
        pe_diff = tf.stop_gradient(pe[..., None] - pe[..., None])     # (B, P, P)

        mask = tf.tile(~tf.eye(self.P, dtype=bool)[None], [tf.shape(pe_diff)[0], 1, 1])
        pe_diff_no_diag = tf.maximum(tf.reshape(pe_diff[mask],
                                                [tf.shape(pe_diff)[0], self.P, self.P-1]), self.eps)
        exps = tf.math.exp(-pe_diff_no_diag**2)
        z = tf.reduce_sum(exps, axis=-1, keepdims=True)
        w = exps / z

        pe_weighted_cross = tf.einsum('bij,bij->', w, dist[mask].reshape(tf.shape(exps))) / (self.P * b)
        if self.monitor is not None:
            self.monitor.update_monitor("pe_weighted_cross_distance", pe_weighted_cross)

        return self.pe_w * pe_weighted_cross + self.predictive_w * mean_pe

    def nonlocal_push(self, embd, last_step=True, exp=True, temperature=10.):
        if last_step:
            embd = embd[..., -1:, :]
        inenc_dists = tf.maximum(tf.linalg.norm(embd[:, None] - embd[None], axis=-2), self.eps)  # (B, B, T, P)

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

        loss = 0.
        if self.continuous_w is not None:
            loss = loss + self.continuous_w * self.continuous_disagreement(embd)
        if self.crosspath_w is not None:
            loss = loss + self.crosspath_w * self.crosspath_disagreement(embd)
        if self.entropy_w is not None:
            koleo_loss = koleo(embd, axis=-2)
            if self.monitor is not None:
                self.monitor.update_monitor("koleo", koleo_loss)
            loss = loss + self.entropy_w * koleo_loss
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
        if self.pe_w is not None:
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
