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


class BasicRNN(tf.keras.layers.Layer):
    def __init__(self, residual=False, out_dim=None, name='rnn', kernel_regularizer=None, out_regularizer=None, **kwargs):
        super().__init__(name=name)
        self.residual = residual
        self.rnn = BasicRNNLayer(name=name + "_internal", kernel_regularizer=kernel_regularizer, **kwargs)
        self.out_proj = tf.keras.layer.Dense(out_dim, name=name + "_out", kernel_regularizer=kernel_regularizer,
                                             activity_regularizer=out_regularizer) if out_dim is not None else None

    def call(self, inputs):
        # inputs shape =  (B, N, T)
        initial_state = tf.zeros((tf.shape(inputs)[0], self.rnn.internal_state_size), dtype=inputs.dtype)

        inputs = tf.unstack(inputs, axis=-1)

        if self.residual:
            raise NotImplementedError("Naive implementation falls for some reason")
        else:
            states = tf.scan(lambda state, inp: self.rnn(inp, state), inputs, initializer=initial_state)

        states = tf.stack(states, axis=1)
        if self.out_proj is not None:
            out = self.out_proj(states)    # (B, T, OUTDIM)
        else:
            out = states
        return out
        #
        #
        # internal_state = tf.zeros((tf.shape(inputs)[0], self.rnn.internal_state_size), dtype=inputs.dtype)
        # states = []
        # for t in range(tf.shape(inputs)[-1]):
        #     cur_calc = self.rnn(inputs[..., t], internal_state)
        #     if self.residual:
        #         internal_state = tf.stop_gradient(internal_state) + cur_calc
        #     else:
        #         internal_state = cur_calc
        #     states.append(internal_state)
        # states = tf.stack(states, axis=1)  # (B, T, INTERNAL_DIM)
        # if self.out_proj is not None:
        #     out = self.out_proj(states)    # (B, T, OUTDIM)
        # else:
        #     out = states
        # return out
