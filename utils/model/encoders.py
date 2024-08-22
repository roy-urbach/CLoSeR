from tensorflow.keras import layers as tf_layers
import tensorflow as tf

from utils.model.layers import ViTBlock, MLP, BasicRNNLayer
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
    def __init__(self, *args, out_dim=64, out_regularizer=None, kernel_regularizer='l1_l2', flatten=True, **kwargs):
        super(MLPEncoder, self).__init__(*args, kernel_regularizer=kernel_regularizer, **kwargs)
        self.flatten = tf_layers.Flatten(name=self.name + '_flatten') if flatten else None
        self.out_layer = tf_layers.Dense(out_dim, activation=None, kernel_regularizer=kernel_regularizer,
                                         activity_regularizer=out_regularizer, name=self.name + '_out')

    def call(self, inputs, **kwargs):
        return self.out_layer(super().call(self.flatten(inputs) if self.flatten is not None else inputs, **kwargs))


class BasicRNN(tf.keras.layers.Layer):
    def __init__(self, residual=False, outdim=None, name='rnn', kernel_regularizer=None, out_regularizer=None,
                 **kwargs):
        super().__init__(name=name)
        self.residual = residual
        self.rnn = BasicRNNLayer(name=name + "_internal", kernel_regularizer=kernel_regularizer, **kwargs)
        self.out_proj = tf.keras.layers.Dense(outdim, name=name + "_out", kernel_regularizer=kernel_regularizer,
                                              activity_regularizer=out_regularizer) if outdim is not None else None
        self.initial_state = None

    def build(self, input_shape):
        self.initial_state = self.add_weight(
            name='initial_state',
            shape=(1, self.rnn.internal_state_size),
            initializer='zeros',
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape =  (B, N, T)
        initial_state = tf.tile(self.initial_state, [tf.shape(inputs)[0], 1])

        inputs = tf.unstack(inputs, axis=-1)

        states = [None] * len(inputs)
        for i in range(len(inputs)):
            cur_calc = self.rnn(inputs[i], states[i - 1] if i else initial_state)
            if self.residual and i:
                cur_calc = cur_calc + states[i - 1]
            states[i] = cur_calc

        stacked_states = tf.stack(states, axis=1)

        if self.out_proj is not None:
            out = self.out_proj(stacked_states)  # (B, T, OUTDIM)
        else:
            out = stacked_states
        return out


class TimeAgnosticMLP(MLPEncoder):
    def __init__(self, bins_per_frame, *args, **kwrags):
        super().__init__(*args, flatten=False, **kwrags)
        self.bins_per_frame = bins_per_frame

    def call(self, inputs, **kwargs):
        # (B, N, T)

        permuted = tf.transpose(inputs, [0, 2, 1])     # (B, T, N)

        if self.bins_per_frame != 1:
            shape = tf.shape(permuted)
            reshaped = tf.reshape(permuted, (shape[0], shape[1] // self.bins_per_frame,
                                             self.bins_per_frame * shape[-1]))  # (B, Frames, N*bins_per_frame)
                                                                                # N first, then bins_per_frame
        else:
            reshaped = permuted

        return super().call(reshaped)
