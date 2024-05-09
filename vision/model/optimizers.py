import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K

# https://github.com/OverLordGoldDragon/keras-adamw/blob/master/keras_adamw


# Utils

def _apply_weight_decays(self, var, var_t):
    l1, l2 = self.weight_decays[var.name]
    if l1 == 0 and l2 == 0:
        if self.init_verbose and not self._init_notified:
            print("Both penalties are 0 for %s, will skip" % var.name)
        return var_t

    norm = math_ops.cast(math_ops.sqrt(1 / self.total_iterations_wd),
                         'float32')
    l1_normalized = l1 * norm
    l2_normalized = l2 * norm

    if l1 != 0 and l2 != 0:
        decay = l1_normalized * math_ops.sign(var) + l2_normalized * var
    elif l1 != 0:
        decay = l1_normalized * math_ops.sign(var)
    else:
        decay = l2_normalized * var
    var_t = var_t - self.eta_t * decay

    if self.init_verbose and not self._init_notified:
        norm_print = (1 / self.total_iterations_wd) ** (1 / 2)
        l1n_print, l2n_print = l1 * norm_print, l2 * norm_print
        decays_str = "{}(L1), {}(L2)".format(l1n_print, l2n_print)
        print('{} weight decay set for {}'.format(decays_str, var.name))
    return var_t


def _compute_eta_t(self):
    PI = 3.141592653589793
    t_frac = math_ops.cast(self.t_cur / (self.total_iterations - 1), 'float32')
    eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
        (1 + math_ops.cos(PI * t_frac))
    return eta_t


def _apply_lr_multiplier(self, lr_t, var):
    multiplier_name = [mult_name for mult_name in self.lr_multipliers
                       if mult_name in var.name]
    if multiplier_name != []:
        lr_mult = self.lr_multipliers[multiplier_name[0]]
    else:
        lr_mult = 1
    lr_t = lr_t * lr_mult

    if self.init_verbose and not self._init_notified:
        lr_print = self._init_lr * lr_mult
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
               '%.e' % round(lr_print, 5), var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(
                var.name, lr_print))
    return lr_t


def _update_t_cur_eta_t(self):  # keras
    self.updates.append(_update_t_cur(self))
    # Cosine annealing
    if self.use_cosine_annealing:
        # ensure eta_t is updated AFTER t_cur
        with tf.control_dependencies([self.updates[-1]]):
            self.updates.append(state_ops.assign(self.eta_t,
                                                 _compute_eta_t(self)))


def _update_t_cur_eta_t_v2(self, lr_t=None, var=None):  # tf.keras
    t_cur_update, eta_t_update = None, None  # in case not assigned

    # update `t_cur` if iterating last `(grad, var)`
    iteration_done = (self._updates_processed == (self._updates_per_iter - 1))
    if iteration_done:
        t_cur_update = _update_t_cur(self)
        self._updates_processed = 0  # reset
    else:
        self._updates_processed += 1

    # Cosine annealing
    if self.use_cosine_annealing and iteration_done:
        # ensure eta_t is updated AFTER t_cur
        with tf.control_dependencies([t_cur_update]):
            eta_t_update = state_ops.assign(self.eta_t, _compute_eta_t(self),
                                            use_locking=self._use_locking)
        self.lr_t = lr_t * self.eta_t  # for external tracking

    return iteration_done, t_cur_update, eta_t_update


def _update_t_cur(self):
    kw = {'use_locking': self._use_locking} if TF_KERAS else {}
    if self.autorestart:
        return control_flow_ops.cond(
            math_ops.equal(self.t_cur, self.total_iterations - 1),
            lambda: state_ops.assign(self.t_cur, 0, **kw),
            lambda: state_ops.assign_add(self.t_cur, 1, **kw),
        )
    return state_ops.assign_add(self.t_cur, 1, **kw)


def _set_autorestart(self, autorestart, use_cosine_annealing):
    if autorestart is None:
        self.autorestart = bool(use_cosine_annealing)
    elif autorestart and not use_cosine_annealing:
        raise ValueError("`autorestart` can only be used with "
                         "`use_cosine_annealing`")
    else:
        self.autorestart = autorestart


def _check_args(self, total_iterations, use_cosine_annealing, weight_decays):
    if use_cosine_annealing and total_iterations > 1:
        print('Using cosine annealing learning rates')
    elif (use_cosine_annealing or weight_decays) and total_iterations <= 1:
        print(WARN, "'total_iterations'==%s, must be >1" % total_iterations
              + " to use cosine annealing and/or weight decays; "
              "proceeding without either")
        self.use_cosine_annealing = False
        self.autorestart = False
        self.weight_decays = {}


def _init_weight_decays(model, zero_penalties, weight_decays):
    if not zero_penalties:
        print(WARN, "loss-based weight penalties should be set to zero. "
              "(set `zero_penalties=True`)")
    if weight_decays is not None and model is not None:
        print(WARN, "`weight_decays` is set automatically when "
              "passing in `model`; will override supplied")
    if model is not None:
        weight_decays = get_weight_decays(model, zero_penalties)
    return weight_decays


def get_weight_decays(model, zero_penalties=False):
    wd_dict = {}
    for layer in model.layers:
        layer_penalties = _get_layer_penalties(layer, zero_penalties)
        if layer_penalties:
            for p in layer_penalties:
                weight_name, weight_penalty = p
                if not all(wp == 0 for wp in weight_penalty):
                    wd_dict.update({weight_name: weight_penalty})
    return wd_dict


def _get_layer_penalties(layer, zero_penalties=False):
    if hasattr(layer, 'cell') or \
      (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
        return _rnn_penalties(layer, zero_penalties)
    elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
        layer = layer.layer

    penalties= []
    for weight_name in ['kernel', 'bias']:
        _lambda = getattr(layer, weight_name + '_regularizer', None)
        if _lambda is not None:
            l1l2 = _get_and_maybe_zero_penalties(_lambda, zero_penalties)
            penalties.append([getattr(layer, weight_name).name, l1l2])
    return penalties


def _rnn_penalties(layer, zero_penalties=False):
    penalties = []
    if hasattr(layer, 'backward_layer'):
        for layer in [layer.forward_layer, layer.backward_layer]:
            penalties += _cell_penalties(layer.cell, zero_penalties)
        return penalties
    else:
        return _cell_penalties(layer.cell, zero_penalties)


def _cell_penalties(rnn_cell, zero_penalties=False):
    cell = rnn_cell
    penalties = []  # kernel-recurrent-bias

    for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
        _lambda = getattr(cell, weight_type + '_regularizer', None)
        if _lambda is not None:
            weight_name = cell.weights[weight_idx].name
            l1l2 = _get_and_maybe_zero_penalties(_lambda, zero_penalties)
            penalties.append([weight_name, l1l2])
    return penalties


def _get_and_maybe_zero_penalties(_lambda, zero_penalties):
    if zero_penalties:
        if hasattr(_lambda, 'l1'):
            _lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
        if hasattr(_lambda, 'l2'):
            _lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
    return (float(getattr(_lambda, 'l1', 0.)),
            float(getattr(_lambda, 'l2', 0.)))


def fill_dict_in_order(_dict, values_list):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = values_list[idx]
    return _dict


def reset_seeds(reset_graph_with_backend=None, verbose=1):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    if verbose:
        print("RANDOM SEEDS RESET")


def KE(x, backend):
    K = backend
    try:
        return K.get_value(K.to_dense(x))
    except Exception:
        try:
            eval_fn = K.function([], [x])
            return eval_fn([])[0]
        except Exception:
            try:
                return K.eager(K.eval)(x)
            except Exception:
                return K.eval(x)



def K_eval(x):
    return KE(x, K)


@keras_export('keras.optimizers.AdamW')
class AdamW(OptimizerV2):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    For extended documentation, see optimizer_v2.Adam.__doc__.
    # Arguments
        model: keras.Model/tf.keras.Model. Pass as first positional argument
            to constructor (AdamW(model, ...)). If passed, automatically extracts
            weight penalties from layers and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        learning_rate: A Tensor or a floating point value.  The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".  @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each
            be a callable that takes no arguments and returns the actual value
            to use. This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for
            backward compatibility, recommended to use `learning_rate` instead.

        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)

    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None,
                 init_verbose=True, eta_min=0, eta_max=1, t_cur=0,
                 name="AdamW", **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)
        eta_t = kwargs.pop('eta_t', 1.)

        super(AdamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)

        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')
        self._updates_per_iter = len(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        m_t = state_ops.assign(m,
                               beta_1_t * m + (1.0 - beta_1_t) * grad,
                               use_locking=self._use_locking)
        v_t = state_ops.assign(v,
                               beta_2_t * v + (1.0 - beta_2_t
                                               ) * math_ops.square(grad),
                               use_locking=self._use_locking)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamW, self).set_weights(weights)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config


@keras_export('keras.optimizers.NadamW')
class NadamW(OptimizerV2):
    """Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the exponentially weighted infinity norm.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adamax".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [3]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [3]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)

    # <1> - if using 'warm restarts', then refers to total expected iterations
        for a given restart; can be an estimate, and training won't stop
        at iterations == total_iterations. [3]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)

    # References
        - [1][Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [2][On the importance of initialization and momentum in deep learning]
             (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
        - [3][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, name="NadamW", **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)

        # Backwards compatibility with keras NAdam optimizer.
        kwargs['decay'] = kwargs.pop('schedule_decay', 0.004)
        eta_t = kwargs.pop('eta_t', 1.)
        learning_rate = kwargs.get('lr', learning_rate)
        if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
            raise ValueError('The Nadam optimizer does not support '
                             'tf.keras.optimizers.LearningRateSchedules as the '
                             'learning rate.')

        super(NadamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self._m_cache = None

        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing
        self.epsilon = epsilon or backend_config.epsilon()

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    def _create_slots(self, var_list):
        var_dtype = var_list[0].dtype.base_dtype
        if self._m_cache is None:
            self._m_cache = self.add_weight('momentum_cache', shape=[],
                                            dtype=var_dtype, initializer='ones',
                                            trainable=False)
            self._weights.append(self._m_cache)
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        self._updates_per_iter = len(var_list)

    def _prepare(self, var_list):
        # Get the value of the momentum cache before starting to apply gradients.
        self._m_cache_read = array_ops.identity(self._m_cache)
        return super(NadamW, self)._prepare(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        next_step = math_ops.cast(self.iterations + 2, var_dtype)
        decay_base = math_ops.cast(0.96, var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * local_step)))
        momentum_cache_t_1 = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * next_step)))
        m_schedule_new = math_ops.cast(self._m_cache_read,
                                       var_dtype) * momentum_cache_t
        if var_dtype is self._m_cache.dtype:
            m_schedule_new = array_ops.identity(state_ops.assign(
                self._m_cache, m_schedule_new, use_locking=self._use_locking))
        m_schedule_next = m_schedule_new * momentum_cache_t_1

        # the following equations given in [1]
        g_prime = grad / (1. - m_schedule_new)
        m_t = beta_1_t * m + (1. - beta_1_t) * grad
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t = beta_2_t * v + (1. - beta_2_t) * math_ops.square(grad)
        v_t_prime = v_t / (1. - math_ops.pow(beta_2_t, local_step))
        m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t * m_t_prime)

        m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
        v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)

        var_t = math_ops.sub(var, self.eta_t * lr_t * m_t_bar / (
                math_ops.sqrt(v_t_prime + epsilon_t)))

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        next_step = math_ops.cast(self.iterations + 2, var_dtype)
        decay_base = math_ops.cast(0.96, var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        momentum_cache_t = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * local_step)))
        momentum_cache_t_1 = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * next_step)))
        m_schedule_new = math_ops.cast(self._m_cache_read,
                                       var_dtype) * momentum_cache_t
        if var_dtype is self._m_cache.dtype:
            m_schedule_new = array_ops.identity(state_ops.assign(
                self._m_cache, m_schedule_new, use_locking=self._use_locking))
        m_schedule_next = m_schedule_new * momentum_cache_t_1

        m_scaled_g_values = grad * (1. - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
            m_t_slice = array_ops.gather(m_t, indices)

        m_t_prime = m_t_slice / (1. - m_schedule_next)
        g_prime = grad / (1. - m_schedule_new)
        m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

        v_scaled_g_values = (grad * grad) * (1. - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)

        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
            v_t_slice = array_ops.gather(v_t, indices)

        v_t_prime_denominator = 1. - math_ops.pow(beta_2_t, local_step)
        v_t_prime = v_t_slice / v_t_prime_denominator
        v_prime_sqrt_plus_eps = math_ops.sqrt(v_t_prime) + epsilon_t

        var_t = self._resource_scatter_add(
            var, indices,
            -self.eta_t * lr_t * m_t_bar / v_prime_sqrt_plus_eps)

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t_bar, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(NadamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config


@keras_export("keras.optimizers.SGDW")
class SGDW(OptimizerV2):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        learning_rate: float hyperparameter >= 0. Learning rate.
        momentum: float hyperparameter >= 0 that accelerates SGDW in the relevant
          direction and dampens oscillations.
        nesterov: boolean. Whether to apply Nesterov momentum.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'SGDW'.
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for
            backward compatibility, recommended to use `learning_rate` instead.

        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)

    # <1> - if using 'warm restarts', then refers to total expected iterations
        for a given restart; can be an estimate, and training won't stop
        at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
        (https://github.com/OverLordGoldDragon/keras_adamw)

    # References
    - [1][Adam - A Method for Stochastic Optimization]
         (http://arxiv.org/abs/1412.6980v8)
    - [2][Fixing Weight Decay Regularization in Adam]
         (https://arxiv.org/abs/1711.05101)
    """
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, name="SGDW", **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)

        eta_t = kwargs.pop('eta_t', 1.)
        super(SGDW, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov
        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        self._updates_per_iter = len(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        if self._momentum:
            momentum = array_ops.identity(self._get_hyper('momentum', var_dtype))
            m = self.get_slot(var, 'momentum')
            v = momentum * m - self.eta_t * lr_t * grad  # velocity
            m = state_ops.assign(m, v, use_locking=self._use_locking)

            if self.nesterov:
                var_t = math_ops.sub(
                    var, -momentum * v + self.eta_t * lr_t * grad)
            else:
                var_t = var + v
        else:
            v = - self.eta_t * lr_t * grad
            var_t = var + v

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update]
        if self._momentum:
            updates += [m]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        if self._momentum:
            momentum = array_ops.identity(self._get_hyper('momentum', var_dtype))
            m = self.get_slot(var, 'momentum')
            v = momentum * m - self.eta_t * lr_t * grad
            m = state_ops.assign(m, v, use_locking=self._use_locking)

            if self.nesterov:
                var_t = self._resource_scatter_add(
                    var, indices, momentum * v - (self.eta_t * lr_t * grad))
            else:
                var_t = self._resource_scatter_add(var, indices, v)
        else:
            v = - self.eta_t * lr_t * grad
            var_t = var + v

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update]
        if self._momentum:
            updates += [m]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(SGDW, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config