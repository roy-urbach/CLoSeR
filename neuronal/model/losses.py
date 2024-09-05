from utils.model.losses import GeneralLossByKey, koleo
import tensorflow as tf


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


class BasicDisagreement(tf.keras.losses.Loss):
    def __init__(self, entropy_w=None, name="basic_disagreement"):
        super().__init__(name=name)
        self.entropy_w = entropy_w

    def call(self, y_true, y_pred):
        dist = tf.linalg.norm(y_pred[..., None] - y_pred[..., None, :], axis=-2)    # (B, T, P, P)
        loss = tf.reduce_mean(dist)
        if self.entropy_w is not None:
            loss += koleo(y_pred, axis=-2) * self.entropy_w
        return loss


class NonLocalContrastive(tf.keras.losses.Loss):
    def __init__(self, temperature=10., name='nonlocal_contrastive'):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        dists = tf.linalg.norm(y_pred[:, None, ..., None] - y_pred[None, ..., None, :], axis=-2)    # (B, B, T, P, P)
        sim = tf.math.exp(-(dists**2)/self.temperature)
        log_z = tf.reduce_logsumexp(sim, axis=1)
        all_pair_loss = log_z - tf.math.log(sim[tf.eye(tf.shape(sim)[0]) != 0])   # -log(pi)=-log(simii/zi)=-log(simii)+log(zi)
        mask = tf.tile(~tf.eye(tf.shape(all_pair_loss)[-1], dtype=tf.bool)[None, None],
                       [tf.shape(all_pair_loss)[0], tf.shape(all_pair_loss)[1], 1, 1])
        relevant_pairs_loss = all_pair_loss[mask]
        return tf.reduce_mean(relevant_pairs_loss)
