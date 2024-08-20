from utils.model.layers import *
from tensorflow.keras import layers as tf_layers
import tensorflow as tf

from utils.tf_utils import serialize


class ViTEncoder:
    def __init__(self, block_kwargs={}, out_block_kwargs={}, layers=3, kernel_regularizer='l1_l2', ln=False,
                 out_regularizer=None, only_classtoken=False, name='ViTEncoder'):
        self.blocks = [ViTBlock(name=name + f'_block{l}', ln=ln, kernel_regularizer=kernel_regularizer, **block_kwargs) for l in range(layers)]
        self.only_classtoken = only_classtoken
        self.out_block = ViTOutBlock(name=name + '_outblock', **out_block_kwargs, ln=ln,
                                     kernel_regularizer=kernel_regularizer, activity_regularizer=out_regularizer)

    def __call__(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        if self.only_classtoken:
            x = x[:, 0]
        out = self.out_block(x)
        return out


@serialize
class MLPEncoder(MLP):
    def __init__(self, *args, out_dim=64, out_regularizer=None, kernel_regularizer='l1_l2', **kwargs):
        super(MLPEncoder, self).__init__(*args, kernel_regularizer=kernel_regularizer, **kwargs)
        self.flatten = tf_layers.Flatten(name=self.name + '_flatten')
        self.out_layer = tf_layers.Dense(out_dim, activation=None, kernel_regularizer=kernel_regularizer,
                                         activity_regularizer=out_regularizer, name=self.name + '_out')

    def call(self, inputs, **kwargs):
        return self.out_layer(super().call(self.flatten(inputs), **kwargs))


class BasicRNN:
    def __init__(self, residual=True, out_dim=None, name='rnn', **kwargs):
        self.residual = residual
        self.rnn = BasicRNNLayer(name=name + "_internal", **kwargs)
        self.out_proj = tf.keras.layer.Dense(out_dim, name=name + "_out") if out_dim is not None else None

    def __call__(self, inputs):
        # inputs shape =  (B, N, T)
        internal_state = tf.constant(tf.zeros((tf.shape(inputs)[0], self.rnn.internal_state_size), dtype=inputs.dtype))
        states = []
        for t in range(tf.shape(inputs)[-1]):
            internal_state = self.rnn(inputs[..., t], internal_state) + (internal_state if self.residual else 0)
            states.append(internal_state)
        states = tf.stack(states, axis=1)  # (B, T, INTERNAL_DIM)
        if self.out_proj is not None:
            out = self.out_proj(states)    # (B, T, OUTDIM)
        else:
            out = states
        return out
