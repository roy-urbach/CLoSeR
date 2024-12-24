from tensorflow.keras import layers as tf_layers
import tensorflow as tf

from utils.model.layers import ViTBlock, MLP, BasicRNNLayer, ViTOutBlock
from utils.tf_utils import serialize
from utils.utils import streval


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
        act = super().call(self.flatten(inputs) if self.flatten is not None else inputs, **kwargs)
        out = self.out_layer(act)
        return out


class BasicRNN(tf.keras.layers.Layer):
    def __init__(self, residual=False, outdim=None, name='rnn', kernel_regularizer=None, out_regularizer=None,
                 only_last=False, **kwargs):
        super().__init__(name=name)
        self.residual = residual
        self.outdim = outdim
        self.rnn = BasicRNNLayer(name=name + "_internal", kernel_regularizer=kernel_regularizer, **kwargs)
        self.out_proj = tf.keras.layers.Dense(outdim, name=name + "_out", kernel_regularizer=kernel_regularizer,
                                              activity_regularizer=out_regularizer) if outdim is not None else None
        self.initial_state = None
        self.only_last = only_last

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
        if self.only_last:  # (B, 1, OUTDIM)
            return out[:, -1]
        return out


class LSTM(tf.keras.layers.Layer):
    def __init__(self, *args, name='lstm', out_regularizer=None, only_last=False, **kwargs):
        super().__init__(name=name)
        self.lstm = tf.keras.layers.LSTM(*args, return_sequences=not only_last, **kwargs)
        self.out_regularizer = out_regularizer
        self.outdim = self.lstm.units

    def call(self, inputs, training=False):
        out = self.lstm(tf.transpose(inputs, [0, 2, 1]), training=training)
        if self.out_regularizer is not None:
            self.add_loss(self.out_regularizer(out))
        return out


class TimeAgnosticMLP(MLPEncoder):
    def __init__(self, bins_per_frame, *args, local=False, **kwrags):
        super().__init__(*args, flatten=False, local=local, **kwrags)
        self.bins_per_frame = bins_per_frame
        self.reshape_shape = None

    def build(self, input_shape):
        # (B, N, T)
        B = -1
        N = input_shape[1]
        T = input_shape[2]
        frames = T // self.bins_per_frame

        self.reshape_shape = [B, frames, N*self.bins_per_frame]

    def call(self, inputs, **kwargs):
        # (B, N, T)

        permuted = tf.transpose(inputs, [0, 2, 1])     # (B, T, N)

        if self.bins_per_frame != 1:
            reshaped = tf.reshape(permuted, self.reshape_shape)  # (B, Frames, N*bins_per_frame)
                                                                                # N first, then bins_per_frame
        else:
            reshaped = permuted

        return super().call(reshaped)


class TimeAgnosticMLPTimeShift(TimeAgnosticMLP):
    def call(self, inputs, **kwargs):
        # (B, N, T)
        B, N = tf.shape(inputs)[0], inputs.shape[1]
        time_shift_out = super().call(inputs)
        return time_shift_out


class RecurrentAdversarial(tf.keras.layers.Layer):
    def __init__(self, encoder='BasicRNN', name='recurrent_adversarial', **kwargs):
        super().__init__(name=name)
        RNN = eval(encoder)
        self.rnn = RNN(name=name + "_rnn", **kwargs)
        self.advers_rnn = RNN(name=name + "_adversrnn", **kwargs)
        self.outdim = None

    def build(self, input_shape):
        # (B, N, T)
        self.outdim = self.rnn.outdim

    def call(self, inputs):
        embds = self.rnn(inputs)    # (B, T, OUTDIM)
        embds_as_inp = tf.transpose(embds[:, :-1], [0, 2, 1])   # (B, DIM, T-1)
        adverse_embd = self.advers_rnn(embds_as_inp)    # (B, T-1, OUTDIM)

        concat = tf.concat([embds,
                            tf.concat([tf.zeros([tf.shape(inputs)[0], 1, self.outdim], dtype=adverse_embd.dtype), adverse_embd], axis=1)],
                           axis=-1)
        return concat


class SplitFramesEncoderWrapper:
    def __init__(self, encoder, frames=2, split_axis=-1, stack_axis=None):
        self.encoder = encoder
        self.frames = frames
        self.split_axis = split_axis
        if stack_axis is None:
            stack_axis = split_axis
        self.stack_axis = stack_axis

    def __call__(self, inputs):
        # bins_in_split = inputs.shape[self.split_axis] // self.frames
        splits = tf.split(inputs, num_or_size_splits=self.frames, axis=self.split_axis)
        encs = [self.encoder(sp) for sp in splits]
        return tf.stack(encs, axis=self.stack_axis)


class TemporalConvNet:
    """
    A 1D convolutional neural network with convolutional hidden layers,
    max-pooling, and global average pooling.

    Args:
        width: The initial number of filters in the first convolutional layer.
        depth: The depth of the model (number of convolutional layers).
        kernel_size: The size of the convolutional kernel (default: 3).
        out_dim: The number of output units (default: 1).
    """

    def __init__(self, width=32, depth=3, kernel_size=3, out_dim=32, name='tempconv',
                 data_format='channels_first', kernel_regularizer=None, activity_regularizer=None,
                 out_regularizer=None):
        super(TemporalConvNet, self).__init__()
        self.depth = depth
        self.conv_layers = []
        self.pool_layers = []
        out_linear = bool(out_dim)
        for i in range(depth):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    filters=width * 2**i,
                    kernel_size=kernel_size,
                    activation='gelu' if i < depth -1 or out_linear else None,
                    padding='same',
                    name=name + f"_conv{i}",
                    data_format=data_format,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer
                )
            )
            if i < depth - 1:  # Add max-pooling for all layers except the last
                self.pool_layers.append(
                    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, name=name + f"_maxpool{i}",
                                                 data_format=data_format)
                )
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(name=name + "_avgpool", )
        self.flatten = tf.keras.layers.Flatten(name=name + "_flatten", activity_regularizer=None if out_linear else out_regularizer)
        self.output_layer = tf.keras.layers.Dense(out_dim, name=name + "_out",
                                                  kernel_regularizer=kernel_regularizer,
                                                  activity_regularizer=out_regularizer
                                                  ) if out_dim else None

    def __call__(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = self.conv_layers[i](x)
            if i < self.depth - 1:  # Apply max-pooling for all layers except the last
                x = self.pool_layers[i](x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x
