# import random
# import tensorflow as tf
# from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
# import numpy as np
# from tensorflow.keras import backend as K
# from tensorflow.keras.legacy import interfaces
# from tensorflow.keras.optimizers import Optimizer
#
#
# # https://github.com/OverLordGoldDragon/keras-adamw/blob/master/keras_adamw
#
# TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
#
# # Utils
#
# def _apply_weight_decays(self, var, var_t):
#     l1, l2 = self.weight_decays[var.name]
#     if l1 == 0 and l2 == 0:
#         if self.init_verbose and not self._init_notified:
#             print("Both penalties are 0 for %s, will skip" % var.name)
#         return var_t
#
#     norm = math_ops.cast(math_ops.sqrt(1 / self.total_iterations_wd),
#                          'float32')
#     l1_normalized = l1 * norm
#     l2_normalized = l2 * norm
#
#     if l1 != 0 and l2 != 0:
#         decay = l1_normalized * math_ops.sign(var) + l2_normalized * var
#     elif l1 != 0:
#         decay = l1_normalized * math_ops.sign(var)
#     else:
#         decay = l2_normalized * var
#     var_t = var_t - self.eta_t * decay
#
#     if self.init_verbose and not self._init_notified:
#         norm_print = (1 / self.total_iterations_wd) ** (1 / 2)
#         l1n_print, l2n_print = l1 * norm_print, l2 * norm_print
#         decays_str = "{}(L1), {}(L2)".format(l1n_print, l2n_print)
#         print('{} weight decay set for {}'.format(decays_str, var.name))
#     return var_t
#
#
# def _compute_eta_t(self):
#     PI = 3.141592653589793
#     t_frac = math_ops.cast(self.t_cur / (self.total_iterations - 1), 'float32')
#     eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
#         (1 + math_ops.cos(PI * t_frac))
#     return eta_t
#
#
# def _apply_lr_multiplier(self, lr_t, var):
#     multiplier_name = [mult_name for mult_name in self.lr_multipliers
#                        if mult_name in var.name]
#     if multiplier_name != []:
#         lr_mult = self.lr_multipliers[multiplier_name[0]]
#     else:
#         lr_mult = 1
#     lr_t = lr_t * lr_mult
#
#     if self.init_verbose and not self._init_notified:
#         lr_print = self._init_lr * lr_mult
#         if lr_mult != 1:
#             print('{} init learning rate set for {} -- {}'.format(
#                '%.e' % round(lr_print, 5), var.name, lr_t))
#         else:
#             print('No change in learning rate {} -- {}'.format(
#                 var.name, lr_print))
#     return lr_t
#
#
# def _update_t_cur_eta_t(self):  # keras
#     self.updates.append(_update_t_cur(self))
#     # Cosine annealing
#     if self.use_cosine_annealing:
#         # ensure eta_t is updated AFTER t_cur
#         with tf.control_dependencies([self.updates[-1]]):
#             self.updates.append(state_ops.assign(self.eta_t,
#                                                  _compute_eta_t(self)))
#
#
# def _update_t_cur_eta_t_v2(self, lr_t=None, var=None):  # tf.keras
#     t_cur_update, eta_t_update = None, None  # in case not assigned
#
#     # update `t_cur` if iterating last `(grad, var)`
#     iteration_done = (self._updates_processed == (self._updates_per_iter - 1))
#     if iteration_done:
#         t_cur_update = _update_t_cur(self)
#         self._updates_processed = 0  # reset
#     else:
#         self._updates_processed += 1
#
#     # Cosine annealing
#     if self.use_cosine_annealing and iteration_done:
#         # ensure eta_t is updated AFTER t_cur
#         with tf.control_dependencies([t_cur_update]):
#             eta_t_update = state_ops.assign(self.eta_t, _compute_eta_t(self),
#                                             use_locking=self._use_locking)
#         self.lr_t = lr_t * self.eta_t  # for external tracking
#
#     return iteration_done, t_cur_update, eta_t_update
#
#
# def _update_t_cur(self):
#     kw = {'use_locking': self._use_locking} if TF_KERAS else {}
#     if self.autorestart:
#         return control_flow_ops.cond(
#             math_ops.equal(self.t_cur, self.total_iterations - 1),
#             lambda: state_ops.assign(self.t_cur, 0, **kw),
#             lambda: state_ops.assign_add(self.t_cur, 1, **kw),
#         )
#     return state_ops.assign_add(self.t_cur, 1, **kw)
#
#
# def _set_autorestart(self, autorestart, use_cosine_annealing):
#     if autorestart is None:
#         self.autorestart = bool(use_cosine_annealing)
#     elif autorestart and not use_cosine_annealing:
#         raise ValueError("`autorestart` can only be used with "
#                          "`use_cosine_annealing`")
#     else:
#         self.autorestart = autorestart
#
#
# def _check_args(self, total_iterations, use_cosine_annealing, weight_decays):
#     if use_cosine_annealing and total_iterations > 1:
#         print('Using cosine annealing learning rates')
#     elif (use_cosine_annealing or weight_decays) and total_iterations <= 1:
#         print("'total_iterations'==%s, must be >1" % total_iterations
#               + " to use cosine annealing and/or weight decays; "
#               "proceeding without either")
#         self.use_cosine_annealing = False
#         self.autorestart = False
#         self.weight_decays = {}
#
#
# def _init_weight_decays(model, zero_penalties, weight_decays):
#     if not zero_penalties:
#         print("loss-based weight penalties should be set to zero. "
#               "(set `zero_penalties=True`)")
#     if weight_decays is not None and model is not None:
#         print("`weight_decays` is set automatically when "
#               "passing in `model`; will override supplied")
#     if model is not None:
#         weight_decays = get_weight_decays(model, zero_penalties)
#     return weight_decays
#
#
# def get_weight_decays(model, zero_penalties=False):
#     wd_dict = {}
#     for layer in model.layers:
#         layer_penalties = _get_layer_penalties(layer, zero_penalties)
#         if layer_penalties:
#             for p in layer_penalties:
#                 weight_name, weight_penalty = p
#                 if not all(wp == 0 for wp in weight_penalty):
#                     wd_dict.update({weight_name: weight_penalty})
#     return wd_dict
#
#
# def _get_layer_penalties(layer, zero_penalties=False):
#     if hasattr(layer, 'cell') or \
#       (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
#         return _rnn_penalties(layer, zero_penalties)
#     elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
#         layer = layer.layer
#
#     penalties= []
#     for weight_name in ['kernel', 'bias']:
#         _lambda = getattr(layer, weight_name + '_regularizer', None)
#         if _lambda is not None:
#             l1l2 = _get_and_maybe_zero_penalties(_lambda, zero_penalties)
#             penalties.append([getattr(layer, weight_name).name, l1l2])
#     return penalties
#
#
# def _rnn_penalties(layer, zero_penalties=False):
#     penalties = []
#     if hasattr(layer, 'backward_layer'):
#         for layer in [layer.forward_layer, layer.backward_layer]:
#             penalties += _cell_penalties(layer.cell, zero_penalties)
#         return penalties
#     else:
#         return _cell_penalties(layer.cell, zero_penalties)
#
#
# def _cell_penalties(rnn_cell, zero_penalties=False):
#     cell = rnn_cell
#     penalties = []  # kernel-recurrent-bias
#
#     for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
#         _lambda = getattr(cell, weight_type + '_regularizer', None)
#         if _lambda is not None:
#             weight_name = cell.weights[weight_idx].name
#             l1l2 = _get_and_maybe_zero_penalties(_lambda, zero_penalties)
#             penalties.append([weight_name, l1l2])
#     return penalties
#
#
# def _get_and_maybe_zero_penalties(_lambda, zero_penalties):
#     if zero_penalties:
#         if hasattr(_lambda, 'l1'):
#             _lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
#         if hasattr(_lambda, 'l2'):
#             _lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
#     return (float(getattr(_lambda, 'l1', 0.)),
#             float(getattr(_lambda, 'l2', 0.)))
#
#
# def fill_dict_in_order(_dict, values_list):
#     for idx, key in enumerate(_dict.keys()):
#         _dict[key] = values_list[idx]
#     return _dict
#
#
# def reset_seeds(reset_graph_with_backend=None, verbose=1):
#     if reset_graph_with_backend is not None:
#         K = reset_graph_with_backend
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         if verbose:
#             print("KERAS AND TENSORFLOW GRAPHS RESET")
#
#     np.random.seed(1)
#     random.seed(2)
#     if tf.__version__[0] == '2':
#         tf.random.set_seed(3)
#     else:
#         tf.set_random_seed(3)
#     if verbose:
#         print("RANDOM SEEDS RESET")
#
#
# def KE(x, backend):
#     K = backend
#     try:
#         return K.get_value(K.to_dense(x))
#     except Exception:
#         try:
#             eval_fn = K.function([], [x])
#             return eval_fn([])[0]
#         except Exception:
#             try:
#                 return K.eager(K.eval)(x)
#             except Exception:
#                 return K.eval(x)
#
# def K_eval(x):
#     return KE(x, K)
#
#
# class AdamW(Optimizer):
#     """AdamW optimizer.
#     Default parameters follow those provided in the original paper.
#     # Arguments
#         model: keras.Model/tf.keras.Model. Pass as first positional argument
#             to constructor (AdamW(model, ...)). If passed, automatically extracts
#             weight penalties from layers and overrides `weight_decays`.
#         zero_penalties: bool. If True and `model` is passed, will zero weight
#             penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
#         learning_rate: float >= 0. Learning rate.
#         beta_1: float, 0 < beta < 1. Generally close to 1.
#         beta_2: float, 0 < beta < 1. Generally close to 1.
#         amsgrad: boolean. Whether to apply the AMSGrad variant of this
#             algorithm from the paper "On the Convergence of Adam and Beyond".
#
#         model: keras.Model/tf.keras.Model/None. If not None, automatically
#             extracts weight penalties from layers, and overrides `weight_decays`.
#         zero_penalties: bool. If True and `model` is passed, will zero weight
#             penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
#         total_iterations: int >= 0. Total expected iterations / weight updates
#                           throughout training, used for normalization; <1>
#         lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
#                           multipliers, as {<layer name>:<multiplier value>}; <2>
#         weight_decays:    dict / None. Name-value pairs specifying weight decays,
#                           as {<weight matrix name>:<weight decay value>}; <2>
#
#         use_cosine_annealing: bool. If True, multiplies lr each train iteration
#                               as a function of eta_min, eta_max, total_iterations,
#                               and t_cur (current); [2]-Appendix, 2
#         autorestart: bool / None. If True, will automatically do Warm Restarts
#                      by resetting `t_cur=0` after `total_iterations`. If None,
#                      will default to same as `use_cosine_annealing`. If True
#                      but `use_cosine_annealing` is False, will raise ValueError.
#                      Note: once optimizer is built (happens on first model fit),
#                      changing `autorestart` has no effect; optimizer needs to be
#                      re-built.
#         eta_min, eta_max: int, int. Min & max values of cosine annealing
#                           lr multiplier; [2]-Appendix, 2
#         t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
#                To be used together with use_cosine_annealing==True
#         total_iterations_wd: int / None. If not None, weight_decays will be
#                      applied according to total_iterations_wd instead of
#                      total_iterations, contrary to authors' scheme. Set to
#                      sum(total_iterations) over all restarts to normalize over
#                      all epochs. May yield improvement over `None`.
#         init_verbose: bool. If True, print weight-name--weight-decay, and
#                       lr-multiplier--layer-name value pairs set during
#                       optimizer initialization (recommended)
#
#     # <1> - if using 'warm restarts', then refers to total expected iterations
#             for a given restart; can be an estimate, and training won't stop
#             at iterations == total_iterations. [2]-Appendix, pg 1
#     # <2> - [AdamW Keras Implementation - Github repository]
#             (https://github.com/OverLordGoldDragon/keras_adamw)
#     # References
#         - [1][Adam - A Method for Stochastic Optimization]
#              (http://arxiv.org/abs/1412.6980v8)
#         - [2][Fixing Weight Decay Regularization in Adam]
#              (https://arxiv.org/abs/1711.05101)
#     """
#     def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
#                  amsgrad=False, model=None, zero_penalties=True,
#                  total_iterations=0, total_iterations_wd=None,
#                  use_cosine_annealing=False, lr_multipliers=None,
#                  weight_decays=None, autorestart=None, init_verbose=True,
#                  eta_min=0, eta_max=1, t_cur=0, **kwargs):
#         if total_iterations > 1:
#             weight_decays = _init_weight_decays(model, zero_penalties,
#                                                 weight_decays)
#
#         self.initial_decay = kwargs.pop('decay', 0.0)
#         self.epsilon = kwargs.pop('epsilon', K.epsilon())
#         learning_rate = kwargs.pop('lr', learning_rate)
#         eta_t = kwargs.pop('eta_t', 1.)
#         super(AdamW, self).__init__(**kwargs)
#
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.learning_rate = K.variable(learning_rate, name='learning_rate')
#             self.beta_1 = K.variable(beta_1, name='beta_1')
#             self.beta_2 = K.variable(beta_2, name='beta_2')
#             self.decay = K.variable(self.initial_decay, name='decay')
#             self.eta_min = K.constant(eta_min, name='eta_min')
#             self.eta_max = K.constant(eta_max, name='eta_max')
#             self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
#             self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
#
#         self.total_iterations = total_iterations
#         self.total_iterations_wd = total_iterations_wd or total_iterations
#         self.amsgrad = amsgrad
#         self.lr_multipliers = lr_multipliers
#         self.weight_decays = weight_decays or {}
#         self.init_verbose = init_verbose
#         self.use_cosine_annealing = use_cosine_annealing
#
#         _set_autorestart(self, autorestart, use_cosine_annealing)
#         _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
#         self._init_lr = learning_rate  # to print lr_mult setup
#         self._init_notified = False
#
#     @interfaces.legacy_get_updates_support
#     @K.symbolic
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#
#         lr = self.learning_rate
#         if self.initial_decay > 0:
#             lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
#                                                       K.dtype(self.decay))))
#
#         t = K.cast(self.iterations, K.floatx()) + 1
#         lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
#                      (1. - K.pow(self.beta_1, t)))
#
#         ms = [K.zeros(K.int_shape(p),
#               dtype=K.dtype(p),
#               name='m_' + str(i))
#               for (i, p) in enumerate(params)]
#         vs = [K.zeros(K.int_shape(p),
#               dtype=K.dtype(p),
#               name='v_' + str(i))
#               for (i, p) in enumerate(params)]
#
#         if self.amsgrad:
#             vhats = [K.zeros(K.int_shape(p),
#                      dtype=K.dtype(p),
#                      name='vhat_' + str(i))
#                      for (i, p) in enumerate(params)]
#         else:
#             vhats = [K.zeros(1, name='vhat_' + str(i))
#                      for i in range(len(params))]
#         self.weights = [self.iterations] + ms + vs + vhats
#
#         for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
#             # Learning rate multipliers
#             if self.lr_multipliers is not None:
#                 lr_t = _apply_lr_multiplier(self, lr_t, p)
#
#             m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
#             v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
#             if self.amsgrad:
#                 vhat_t = K.maximum(vhat, v_t)
#                 p_t = p - self.eta_t * lr_t * m_t / (
#                     K.sqrt(vhat_t) + self.epsilon)
#                 self.updates.append(K.update(vhat, vhat_t))
#             else:
#                 p_t = p - self.eta_t * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
#
#             self.updates.append(K.update(m, m_t))
#             self.updates.append(K.update(v, v_t))
#
#             # Weight decays
#             if p.name in self.weight_decays.keys():
#                 p_t = _apply_weight_decays(self, p, p_t)
#             new_p = p_t
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#
#             self.updates.append(K.update(p, new_p))
#
#         # Cosine annealing
#         _update_t_cur_eta_t(self)
#         self.lr_t = lr_t * self.eta_t  # for external tracking
#
#         self._init_notified = True
#         return self.updates
#
#     def get_config(self):
#         config = {
#             'learning_rate': float(K_eval(self.learning_rate)),
#             'beta_1': float(K_eval(self.beta_1)),
#             'beta_2': float(K_eval(self.beta_2)),
#             'decay': float(K_eval(self.decay)),
#             'total_iterations': int(self.total_iterations),
#             'weight_decays': self.weight_decays,
#             'lr_multipliers': self.lr_multipliers,
#             'use_cosine_annealing': self.use_cosine_annealing,
#             'autorestart': self.autorestart,
#             't_cur': int(K_eval(self.t_cur)),
#             'eta_t': float(K_eval(self.eta_t)),
#             'eta_min': float(K_eval(self.eta_min)),
#             'eta_max': float(K_eval(self.eta_max)),
#             'init_verbose': self.init_verbose,
#             'epsilon': self.epsilon,
#             'amsgrad': self.amsgrad
#         }
#         base_config = super(AdamW, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class NadamW(Optimizer):
#     """Nesterov Adam optimizer.
#
#     Much like Adam is essentially RMSprop with momentum,
#     Nadam is Adam RMSprop with Nesterov momentum.
#
#     Default parameters follow those provided in the paper.
#     It is recommended to leave the parameters of this optimizer
#     at their default values.
#
#     # Arguments
#         lr: float >= 0. Learning rate.
#         beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
#         epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
#
#         model: keras.Model/tf.keras.Model/None. If not None, automatically
#             extracts weight penalties from layers, and overrides `weight_decays`.
#         zero_penalties: bool. If True and `model` is passed, will zero weight
#             penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
#         total_iterations: int >= 0. Total expected iterations / weight updates
#                           throughout training, used for normalization; <1>
#         lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
#                           multipliers, as {<layer name>:<multiplier value>}; <2>
#         weight_decays:    dict / None. Name-value pairs specifying weight decays,
#                           as {<weight matrix name>:<weight decay value>}; <2>
#
#         use_cosine_annealing: bool. If True, multiplies lr each train iteration
#                               as a function of eta_min, eta_max, total_iterations,
#                               and t_cur (current); [3]-Appendix, 2
#         autorestart: bool / None. If True, will automatically do Warm Restarts
#                      by resetting `t_cur=0` after `total_iterations`. If None,
#                      will default to same as `use_cosine_annealing`. If True
#                      but `use_cosine_annealing` is False, will raise ValueError.
#                      Note: once optimizer is built (happens on first model fit),
#                      changing `autorestart` has no effect; optimizer needs to be
#                      re-built.
#         eta_min, eta_max: int, int. Min & max values of cosine annealing
#                           lr multiplier; [3]-Appendix, 2
#         t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
#                To be used together with use_cosine_annealing==True
#         total_iterations_wd: int / None. If not None, weight_decays will be
#                      applied according to total_iterations_wd instead of
#                      total_iterations, contrary to authors' scheme. Set to
#                      sum(total_iterations) over all restarts to normalize over
#                      all epochs. May yield improvement over `None`.
#         init_verbose: bool. If True, print weight-name--weight-decay, and
#                       lr-multiplier--layer-name value pairs set during
#                       optimizer initialization (recommended)
#
#     # <1> - if using 'warm restarts', then refers to total expected iterations
#         for a given restart; can be an estimate, and training won't stop
#         at iterations == total_iterations. [3]-Appendix, pg 1
#     # <2> - [AdamW Keras Implementation - Github repository]
#             (https://github.com/OverLordGoldDragon/keras_adamw)
#
#     # References
#         - [1][Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
#         - [2][On the importance of initialization and momentum in deep learning]
#              (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
#         - [3][Fixing Weight Decay Regularization in Adam]
#              (https://arxiv.org/abs/1711.05101)
#     """
#     def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999,
#                  model=None, zero_penalties=True,
#                  total_iterations=0, total_iterations_wd=None,
#                  use_cosine_annealing=False, lr_multipliers=None,
#                  weight_decays=None, autorestart=None, init_verbose=True,
#                  eta_min=0, eta_max=1, t_cur=0, **kwargs):
#         if total_iterations > 1:
#             weight_decays = _init_weight_decays(model, zero_penalties,
#                                                 weight_decays)
#
#         self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
#         self.epsilon = kwargs.pop('epsilon', K.epsilon())
#         learning_rate = kwargs.pop('lr', learning_rate)
#         eta_t = kwargs.pop('eta_t', 1.)
#         super(NadamW, self).__init__(**kwargs)
#
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.m_schedule = K.variable(1., name='m_schedule')
#             self.learning_rate = K.variable(learning_rate, name='learning_rate')
#             self.beta_1 = K.variable(beta_1, name='beta_1')
#             self.beta_2 = K.variable(beta_2, name='beta_2')
#             self.eta_min = K.constant(eta_min, name='eta_min')
#             self.eta_max = K.constant(eta_max, name='eta_max')
#             self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
#             self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
#
#         self.total_iterations = total_iterations
#         self.total_iterations_wd = total_iterations_wd or total_iterations
#         self.lr_multipliers = lr_multipliers
#         self.weight_decays = weight_decays or {}
#         self.use_cosine_annealing = use_cosine_annealing
#         self.init_verbose = init_verbose
#
#         _set_autorestart(self, autorestart, use_cosine_annealing)
#         _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
#         self._init_lr = learning_rate  # to print lr_mult setup
#         self._init_notified = False
#
#     @interfaces.legacy_get_updates_support
#     @K.symbolic
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#
#         t = K.cast(self.iterations, K.floatx()) + 1
#
#         # Due to the recommendations in [2], i.e. warming momentum schedule
#         momentum_cache_t = self.beta_1 * (1. - 0.5 * (
#             K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
#         momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
#             K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
#         m_schedule_new = self.m_schedule * momentum_cache_t
#         m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
#         self.updates.append((self.m_schedule, m_schedule_new))
#
#         shapes = [K.int_shape(p) for p in params]
#         ms = [K.zeros(shape, name='m_' + str(i))
#               for (i, shape) in enumerate(shapes)]
#         vs = [K.zeros(shape, name='v_' + str(i))
#               for (i, shape) in enumerate(shapes)]
#
#         self.weights = [self.iterations, self.m_schedule] + ms + vs
#
#         for p, g, m, v in zip(params, grads, ms, vs):
#             # Learning rate multipliers
#             lr_t = self.learning_rate
#             if self.lr_multipliers is not None:
#                 lr_t = _apply_lr_multiplier(self, lr_t, p)
#
#             # the following equations given in [1]
#             g_prime = g / (1. - m_schedule_new)
#             m_t = self.beta_1 * m + (1. - self.beta_1) * g
#             m_t_prime = m_t / (1. - m_schedule_next)
#             v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
#             v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
#             m_t_bar = (1. - momentum_cache_t) * g_prime + (
#                 momentum_cache_t_1 * m_t_prime)
#
#             self.updates.append(K.update(m, m_t))
#             self.updates.append(K.update(v, v_t))
#             p_t = p - self.eta_t * lr_t * m_t_bar / (
#                     K.sqrt(v_t_prime) + self.epsilon)
#
#             # Weight decays
#             if p.name in self.weight_decays.keys():
#                 p_t = _apply_weight_decays(self, p, p_t)
#             new_p = p_t
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#             self.updates.append(K.update(p, new_p))
#
#         # Cosine annealing
#         _update_t_cur_eta_t(self)
#         self.lr_t = lr_t * self.eta_t  # for external tracking
#
#         self._init_notified = True
#         return self.updates
#
#     def set_weights(self, weights):
#         params = self.weights
#         # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
#         # since it does not include m_schedule at head of the weight list. Set
#         # m_schedule to 1.
#         if len(params) == len(weights) + 1:
#             weights = [weights[0]] + [np.array(1.)] + weights[1:]
#         super(NadamW, self).set_weights(weights)
#
#     def get_config(self):
#         config = {
#             'learning_rate': float(K_eval(self.learning_rate)),
#             'beta_1': float(K_eval(self.beta_1)),
#             'beta_2': float(K_eval(self.beta_2)),
#             'epsilon': self.epsilon,
#             'schedule_decay': self.schedule_decay,
#             'total_iterations': int(self.total_iterations),
#             'weight_decays': self.weight_decays,
#             'lr_multipliers': self.lr_multipliers,
#             'use_cosine_annealing': self.use_cosine_annealing,
#             'autorestart': self.autorestart,
#             't_cur': int(K_eval(self.t_cur)),
#             'eta_t': float(K_eval(self.eta_t)),
#             'eta_min': float(K_eval(self.eta_min)),
#             'eta_max': float(K_eval(self.eta_max)),
#             'init_verbose': self.init_verbose
#         }
#         base_config = super(NadamW, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class SGDW(Optimizer):
#     """Stochastic gradient descent optimizer.
#
#     Includes support for momentum,
#     learning rate decay, and Nesterov momentum.
#
#     # Arguments
#         lr: float >= 0. Learning rate.
#         momentum: float >= 0. Parameter that accelerates SGD
#             in the relevant direction and dampens oscillations.
#         decay: float >= 0. Learning rate decay over each update.
#         nesterov: boolean. Whether to apply Nesterov momentum.
#
#         model: keras.Model/tf.keras.Model/None. If not None, automatically
#             extracts weight penalties from layers, and overrides `weight_decays`.
#         zero_penalties: bool. If True and `model` is passed, will zero weight
#             penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
#         total_iterations: int >= 0. Total expected iterations / weight updates
#                           throughout training, used for normalization; <1>
#         lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
#                           multipliers, as {<layer name>:<multiplier value>}; <2>
#         weight_decays:    dict / None. Name-value pairs specifying weight decays,
#                           as {<weight matrix name>:<weight decay value>}; <2>
#
#         use_cosine_annealing: bool. If True, multiplies lr each train iteration
#                               as a function of eta_min, eta_max, total_iterations,
#                               and t_cur (current); [2]-Appendix, 2
#         autorestart: bool / None. If True, will automatically do Warm Restarts
#                      by resetting `t_cur=0` after `total_iterations`. If None,
#                      will default to same as `use_cosine_annealing`. If True
#                      but `use_cosine_annealing` is False, will raise ValueError.
#                      Note: once optimizer is built (happens on first model fit),
#                      changing `autorestart` has no effect; optimizer needs to be
#                      re-built.
#         eta_min, eta_max: int, int. Min & max values of cosine annealing
#                           lr multiplier; [2]-Appendix, 2
#         t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
#                To be used together with use_cosine_annealing==True
#         total_iterations_wd: int / None. If not None, weight_decays will be
#                      applied according to total_iterations_wd instead of
#                      total_iterations, contrary to authors' scheme. Set to
#                      sum(total_iterations) over all restarts to normalize over
#                      all epochs. May yield improvement over `None`.
#         init_verbose: bool. If True, print weight-name--weight-decay, and
#                       lr-multiplier--layer-name value pairs set during
#                       optimizer initialization (recommended)
#
#     # <1> - if using 'warm restarts', then refers to total expected iterations
#         for a given restart; can be an estimate, and training won't stop
#         at iterations == total_iterations. [2]-Appendix, pg 1
#     # <2> - [AdamW Keras Implementation - Github repository]
#         (https://github.com/OverLordGoldDragon/keras_adamw)
#
#     # References
#     - [1][Adam - A Method for Stochastic Optimization]
#          (http://arxiv.org/abs/1412.6980v8)
#     - [2][Fixing Weight Decay Regularization in Adam]
#          (https://arxiv.org/abs/1711.05101)
#     """
#     def __init__(self, learning_rate=0.01, momentum=0., nesterov=False,
#                  model=None, zero_penalties=True,
#                  total_iterations=0, total_iterations_wd=None,
#                  use_cosine_annealing=False, lr_multipliers=None,
#                  weight_decays=None, autorestart=None, init_verbose=True,
#                  eta_min=0, eta_max=1, t_cur=0, **kwargs):
#         if total_iterations > 1:
#             weight_decays = _init_weight_decays(model, zero_penalties,
#                                                 weight_decays)
#
#         self.initial_decay = kwargs.pop('decay', 0.0)
#         learning_rate = kwargs.pop('lr', learning_rate)
#         eta_t = kwargs.pop('eta_t', 1.)
#         super(SGDW, self).__init__(**kwargs)
#
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.learning_rate = K.variable(learning_rate, name='learning_rate')
#             self.momentum = K.variable(momentum, name='momentum')
#             self.decay = K.variable(self.initial_decay, name='decay')
#             self.eta_min = K.constant(eta_min, name='eta_min')
#             self.eta_max = K.constant(eta_max, name='eta_max')
#             self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
#             self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
#
#         self.total_iterations = total_iterations
#         self.total_iterations_wd = total_iterations_wd or total_iterations
#         self.nesterov = nesterov
#         self.lr_multipliers = lr_multipliers
#         self.weight_decays = weight_decays or {}
#         self.init_verbose = init_verbose
#         self.use_cosine_annealing = use_cosine_annealing
#
#         _set_autorestart(self, autorestart, use_cosine_annealing)
#         _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
#         self._init_lr = learning_rate  # to print lr_mult setup
#         self._init_notified = False
#
#     @interfaces.legacy_get_updates_support
#     @K.symbolic
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#
#         lr = self.learning_rate
#         if self.initial_decay > 0:
#             lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
#                                                       K.dtype(self.decay))))
#         # momentum
#         shapes = [K.int_shape(p) for p in params]
#         moments = [K.zeros(shape, name='moment_' + str(i))
#                    for (i, shape) in enumerate(shapes)]
#         self.weights = [self.iterations] + moments
#
#         for p, g, m in zip(params, grads, moments):
#             # Learning rate multipliers
#             lr_t = self.learning_rate
#             if self.lr_multipliers is not None:
#                 lr_t = _apply_lr_multiplier(self, lr_t, p)
#
#             v = self.momentum * m - self.eta_t * lr_t * g  # velocity
#             self.updates.append(K.update(m, v))
#
#             if self.nesterov:
#                 p_t = p + self.momentum * v - self.eta_t * lr_t * g
#             else:
#                 p_t = p + v
#
#             # Weight decays
#             if p.name in self.weight_decays.keys():
#                 p_t = _apply_weight_decays(self, p, p_t)
#             new_p = p_t
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#             self.updates.append(K.update(p, new_p))
#
#         # Cosine annealing
#         _update_t_cur_eta_t(self)
#         self.lr_t = lr_t * self.eta_t  # for external tracking
#
#         self._init_notified = True
#         return self.updates
#
#     def get_config(self):
#         config = {
#             'learning_rate': float(K_eval(self.learning_rate)),
#             'momentum': float(K_eval(self.momentum)),
#             'decay': float(K_eval(self.decay)),
#             'nesterov': self.nesterov,
#             'total_iterations': int(self.total_iterations),
#             'weight_decays': self.weight_decays,
#             'lr_multipliers': self.lr_multipliers,
#             'use_cosine_annealing': self.use_cosine_annealing,
#             'autorestart': self.autorestart,
#             't_cur': int(K_eval(self.t_cur)),
#             'eta_t': float(K_eval(self.eta_t)),
#             'eta_min': float(K_eval(self.eta_min)),
#             'eta_max': float(K_eval(self.eta_max)),
#             'init_verbose': self.init_verbose
#         }
#         base_config = super(SGDW, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer


class Adam(Optimizer):
    """
    https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/adam.py
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adam",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="velocity"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="velocity_hat"
                    )
                )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(learning_rate, variable.dtype)
        gradient = tf.cast(gradient, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(
            tf.cast(self.beta_1, variable.dtype), local_step
        )
        beta_2_power = tf.pow(
            tf.cast(self.beta_2, variable.dtype), local_step
        )

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(
            m, tf.multiply(tf.subtract(gradient, m), 1 - self.beta_1)
        )
        self.assign_add(
            v,
            tf.multiply(
                tf.subtract(tf.square(gradient), v), 1 - self.beta_2
            ),
        )
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, tf.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            tf.divide(
                tf.multiply(m, alpha), tf.add(tf.sqrt(v), self.epsilon)
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


class AdamW(Adam):
    """
    https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/adamw.py
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamw",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )

        if self.weight_decay is None:
            raise ValueError(
                "Argument `weight_decay` must be a float. Received: "
                "weight_decay=None"
            )