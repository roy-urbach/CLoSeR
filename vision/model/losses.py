import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np
import scipy
from utils.model.metrics import LossMonitors

GRAPH = None


class GeneralGraphCLoSeRLoss(Loss):
    def __init__(self, num_pathways=None, graph=None, temperature=10, stable=True, stop_grad_dist=False,
                 other_stop_grad_dist=False, mse=False, **kwargs):
        """
        :param num_pathways: Number of nodes\encoders\pathways
        :param graph: a matrix of edges. If None, assumes all-to-all
        :param temperature: tau
        :param stable: removes max from similarity measure (assuming it will next be divided by the denominator)
        :param stop_grad_dist: if true, removes gradient from the j'th encoder when optimizing -log(p(mu_j(x_b)|mu_i(x_b))
                               defaults to False
        :param mse: if true, the distance calculated is the mse and not Euclidian distance. Equivalent to higher temperature
        :param kwargs: Loss kwargs
        """
        super().__init__(**kwargs)
        if graph is not None:
            global GRAPH
            GRAPH = tf.constant(eval(graph) if isinstance(graph, str) else graph, dtype=tf.float32)
            self.graph = GRAPH
        elif num_pathways is not None:
            self.graph = tf.constant((1 - np.eye(num_pathways)) / (num_pathways * (num_pathways - 1)))
        else:
            self.graph = None
        self.temperature = temperature
        self.stable = stable
        self.stop_grad_dist = stop_grad_dist
        self.other_stop_grad_dist = other_stop_grad_dist
        self.mse = mse
        self.monitor = LossMonitors("pull", name="")

    def calculate_dists(self, embedding, stop_grad=False, other_stop_grad=False, mean=False):
        """
        Euclidian distance squared
        :param embedding: (B, DIM, N)
        :param stop_grad: if true, (*B*, B, N, N) is calculated with stop grad
        :param other_stop_grad: if true, (B, *B*, N, N) is calculated with stop grad
        :param mean: if true, calculating the MSE, not squared Euclidian distance
        :return: (B, B, N, N) distance matrix
        """
        reduce_f = tf.reduce_mean if mean else tf.reduce_sum
        left_side = (tf.stop_gradient(embedding) if stop_grad else embedding)[:, None, ..., :, None]
        right_side = (tf.stop_gradient(embedding) if other_stop_grad else embedding)[None, :, ..., None, :]
        dist = reduce_f(tf.pow(left_side - right_side, 2), axis=2)
        return dist

    def calculate_logits(self, dist_squared):
        """
        calculates -dist^2 / tau
        :param dist_squared: squared distance shaped (B, B, N, N)
        :return: (B, B, N, N)
        """
        logits = -dist_squared / self.temperature
        return logits

    def calculate_exp_logits(self, logits):
        """
        Calculates psi (similarity)
        :param logits: (B, B, N, N)
        :return: (B, B, N, N)
        """
        if self.stable:
            logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=0, keepdims=True)
        return tf.exp(logits)

    def calculate_conditional_pseudo_likelihood(self, exp_logits):
        """
        Not used for the loss, because of tricks

        Calculates conditional pseudo-likelihood
        :param exp_logits: (B, B, N, N)
        :return: (B, B, N, N)
        """
        softmaxed = exp_logits / tf.reduce_sum(exp_logits, axis=0, keepdims=True)
        return softmaxed

    def call(self, y_true, y_pred):
        embd = y_pred
        dists_squared = self.calculate_dists(embd, stop_grad=self.stop_grad_dist, other_stop_grad=self.other_stop_grad_dist, mean=self.mse)
        logits = self.calculate_logits(dists_squared)

        log_denom = tf.math.reduce_logsumexp(logits, axis=0)    # (B, N, N)
        negative_log_likelihood = -(logits[tf.eye(tf.shape(logits)[0], dtype=tf.bool)] - log_denom) # (B, N, N)

        mean_nll = tf.reduce_mean(negative_log_likelihood, axis=0)  # (N, N)
        g = tf.cast(self.get_graph(num_pathways=mean_nll.shape[0], numpy=False), dtype=mean_nll.dtype)
        loss = tf.tensordot(g, mean_nll, axes=[[0, 1], [0, 1]])

        self.monitor.update_monitor("pull", loss)
        return loss

    @staticmethod
    def plot_graph(mat, ax=None, interaction_c='k', nointeraction_c='w',
                   ticks=True, cbar=True, labels=True, text_size=10,
                   figsize=(3,2), **kwargs):
        import matplotlib
        from matplotlib import pyplot as plt
        from utils.plot_utils import colorbar
        cmap = matplotlib.colors.ListedColormap([nointeraction_c, interaction_c])
        if ax is None:
            fig, ax = plt.subplots(figsize=(figsize))
        im = ax.imshow(mat > 0, cmap=cmap, vmax=1, vmin=0, origin='lower', **kwargs)
        if cbar:
            cbar = colorbar(im)
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(['no interaction', 'interaction'], size=text_size)
            for sp in cbar.ax.spines:
                cbar.ax.spines[sp].set_color(interaction_c)

        if labels:
            ax.set_xlabel(r"encoder $j$", size=text_size)
            ax.set_ylabel(r"encoder $i$", size=text_size)

        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        for sp in ax.spines:
            ax.spines[sp].set_color(interaction_c)

    def get_graph(self, num_pathways=None, numpy=True):
        if self.graph is not None:
            g = self.graph
        elif num_pathways is not None:
            g = tf.constant((1 - np.eye(num_pathways)) / (num_pathways * (num_pathways - 1)))
        else:
            return None
        if numpy:
            return g.numpy()
        else:
            return g

    def plot_pull(self, num_pathways=None, ax=None, interaction_c=(0.05, 0.3, 0.15, 0.8), nointeraction_c=(0.05, 0.3, 0.15, 0.1), **kwargs):
        graph = self.get_graph(num_pathways=num_pathways)
        if graph is not None:
            self.plot_graph(self.graph.numpy().T, ax=ax, interaction_c=interaction_c, nointeraction_c=nointeraction_c, **kwargs)
        else:
            raise Exception("Graph object doesn't have a graph set up")


class CLoSeRLoss(GeneralGraphCLoSeRLoss):
    """
    The basic CLoSeR loss L_CLoSeR, with all-to-all interactions
    """
    def __init__(self, *args, **kwargs):
        super(CLoSeRLoss, self).__init__(*args, **kwargs, graph=None)


class NumEdgesGraphLoss(GeneralGraphCLoSeRLoss):
    """
    G^k_N family, with N nodes and k random edges
    """
    def __init__(self, num_pathways, *args, num_edges_pull=None, p_pull=1., **kwargs):
        """
        :param num_pathways: N - the number of nodes
        :param args: GeneralGraphCLoSeRLoss args
        :param num_edges_pull: k - the number of edges. If None (like in the default), uses p_pull
        :param p_pull: if num_edges_pull is not specified, becomes Erdos-Reinyi graphs with G(num_pathways,p_pull)
        :param kwargs: GeneralGraphCLoSeRLoss kwargs
        """
        max_edges = num_pathways * (num_pathways - 1)
        num_edges_pull = int(max_edges * p_pull) if num_edges_pull is None else num_edges_pull
        pull_masked = np.array([True]*num_edges_pull + [False]*(max_edges - num_edges_pull))
        np.random.shuffle(pull_masked)

        graph = np.full((num_pathways, num_pathways), False)
        graph[~np.eye(num_pathways).astype(bool)] = pull_masked

        super(NumEdgesGraphLoss, self).__init__(*args, num_pathways=num_pathways, graph=graph, **kwargs)


class CirculantGraphLoss(GeneralGraphCLoSeRLoss):
    """
    Ring structure with a circulant adjacency matrix
    """
    def __init__(self, num_pathways, num_lateral=3, symmetric=False, *args, **kwargs):
        """
        :param num_pathways: N - the number of nodes
        :param num_lateral: number of clockwise interactions
        :param symmetric: if true, num_lateral counter-clockwise too (for example, bidirectional ring has num_lateral=1 and symmetric=True)
        :param args: GeneralGraphCLoSeRLoss args
        :param kwargs: GeneralGraphCLoSeRLoss kwargs
        """
        if symmetric:
            graph = scipy.linalg.circulant(
                np.concatenate([[0], np.ones(num_lateral), np.zeros(num_pathways - 2*num_lateral - 1), np.ones(num_lateral)])) / (
                                 2*num_lateral * num_pathways)
        else:
            graph = scipy.linalg.circulant(np.concatenate([[0], np.zeros(num_pathways - num_lateral - 1), np.ones(num_lateral)])) / (num_lateral * num_pathways)
        super(CirculantGraphLoss, self).__init__(*args, num_pathways=num_pathways, graph=graph, **kwargs)
        self.symmetric = symmetric
        self.num_lateral = num_lateral


class StarGraphLoss(GeneralGraphCLoSeRLoss):
    """
    Star structure with k nodes in the middle and N-k periphery
    """
    def __init__(self, num_pathways, num_center=1, *args, **kwargs):
        """
        :param num_pathways: N - the number of nodes
        :param num_center: K - the number of nodes at the center
        :param args: GeneralGraphCLoSeRLoss args
        :param kwargs: GeneralGraphCLoSeRLoss kwargs
        """
        graph = np.zeros((num_pathways, num_pathways))
        graph[:num_center] = 1
        graph[:, :num_center] = 1
        graph[np.arange(num_pathways), np.arange(num_pathways)] = 0

        graph = graph / graph.sum(axis=0, keepdims=True)    # it was found to be useful

        graph = graph / graph.sum()

        super(StarGraphLoss, self).__init__(*args, num_pathways=num_pathways, graph=graph, **kwargs)
        self.num_center = num_center
