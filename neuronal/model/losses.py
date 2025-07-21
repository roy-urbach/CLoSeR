import tensorflow as tf

from utils.model.metrics import LossMonitors
from utils.utils import streval


class TemporalContiguityLoss(tf.keras.losses.Loss):
    def __init__(self, cont_w=1, var_w=1, cov_w=10, pull_w=None, tpull_w=None, eps=1e-4, name='temporal_contiguity_loss'):
        """
        :param cont_w: coefficient for the single encoder temporal contiguity loss term
        :param var_w: coefficient for the variance maximizing term
        :param cov_w: coefficient for the covariance^2 minimizing term
        :param pull_w: coefficient for cross-agreement at time t (not used in the paper)
        :param tpull_w: coefficient for cross-agreement in time
        :param eps: adding to the calculation of the variance for stability
        :param name: name of the loss
        """
        super().__init__(name=name)
        self.eps = streval(eps)
        self.cont_w = cont_w
        self.var_w = var_w
        self.cov_w = cov_w
        self.pull_w = pull_w
        self.tpull_w = tpull_w

        losses = []
        if self.cont_w:
            losses.append("L_cont")
        if self.var_w:
            losses.append("L_var")
        if self.cov_w:
            losses.append("L_cov")
        if self.pull_w:
            losses.append("L_pull")
        if self.tpull_w:
            losses.append("L_tpull")
        self.monitor = LossMonitors(*losses, name="")

    def continuous_loss(self, prev_embd, embd):
        # (B, DIM, P)
        diff = embd - tf.stop_gradient(prev_embd)
        mean_dist = tf.reduce_mean(diff**2)  # MSE
        if self.monitor is not None:
            self.monitor.update_monitor("L_cont", mean_dist)
        return mean_dist

    def neg_log_var(self, embd):
        var = tf.reduce_sum((embd - tf.reduce_mean(tf.stop_gradient(embd), axis=0, keepdims=True))**2, axis=0) / tf.cast(tf.shape(embd)[0] - 1, embd.dtype) # (DIM, P, )
        if self.monitor is not None:
            self.monitor.update_monitor("L_var", tf.reduce_mean(var))
        out = -tf.reduce_mean(tf.math.log(var + self.eps))
        return out

    def decorrelate(self, embd):
        centered = (embd - tf.reduce_mean(tf.stop_gradient(embd), axis=0, keepdims=True))
        co = centered[..., :, None, :] * centered[..., None, :, :]
        cov = tf.reduce_sum(co, axis=(0,)) / tf.cast(tf.shape(embd)[0] - 1, co.dtype)  # (DIM, DIM, P)
        cov_sqr = cov ** 2
        mean_cov_sqr = tf.reduce_mean(cov_sqr, axis=-1)
        mean_cov_loss = tf.reduce_mean(mean_cov_sqr[~tf.eye(embd.shape[1], dtype=tf.bool)])
        self.monitor.update_monitor("L_cov", mean_cov_loss)
        return mean_cov_loss

    def crossdist(self, embd):
        # (B, DIM, P)
        mse = tf.reduce_mean((embd[..., None] - tf.stop_gradient(embd[..., None, :]))**2, axis=(0,1))  # (P, P)
        mse_nondiag = mse[~tf.eye(embd.shape[-1], dtype=tf.bool)]
        mean_mse = tf.reduce_mean(mse_nondiag)
        self.monitor.update_monitor("L_pull", mean_mse)
        return mean_mse

    def crosscont(self,  prev_embd, embd):
        # (B, DIM, P)
        mse = tf.reduce_mean((embd[..., None] - tf.stop_gradient(prev_embd[..., None, :]))**2, axis=(0,1))  # (P, P)
        mse_nondiag = mse[~tf.eye(embd.shape[-1], dtype=tf.bool)]
        mean_mse = tf.reduce_mean(mse_nondiag)
        self.monitor.update_monitor("L_tpull", mean_mse)
        return mean_mse

    def call(self, y_true, y_pred):
        # (B, T, DIM, P)
        embd = y_pred[:, -1]
        prev_embd = y_pred[:, -2]
        prev_embd_sg = tf.stop_gradient(prev_embd)
        loss = 0.
        if self.cont_w:
            loss = loss + self.cont_w * self.continuous_loss(prev_embd_sg, embd)
        if self.var_w:
            loss = loss + self.var_w * self.neg_log_std(embd)
        if self.cov_w:
            loss = loss + self.cov_w * self.decorrelate(embd)
        if self.pull_w:
            loss = loss + self.pull_w * self.crossdist(embd)
        if self.tpull_w:
            loss = loss + self.tpull_w * self.crosscont(prev_embd_sg, embd)
        return loss


class LPL(TemporalContiguityLoss):
    def __init__(self, *args, cont_w=1, var_w=1, cov_w=10, **kwargs):
        super(LPL, self).__init__(*args, cont_w=cont_w, var_w=var_w, cov_w=cov_w, **kwargs)


class TemporalCLoSeR(TemporalContiguityLoss):
    def __init__(self, *args, cont_w=None, var_w=1, cov_w=10, tpull_w=2, **kwargs):
        super(TemporalCLoSeR, self).__init__(*args, cont_w=cont_w, var_w=var_w, cov_w=cov_w, tpull_w=tpull_w, **kwargs)
