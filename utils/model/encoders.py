from tensorflow.keras import layers as tf_layers
import tensorflow as tf

from utils.model.layers import ViTBlock, MLP, ViTOutBlock


class ViTEncoder:

    def __init__(self, block_kwargs={}, out_block_kwargs={}, layers=3, kernel_regularizer='l1_l2', ln=False,
                 out_regularizer=None, only_classtoken=False, name='ViTEncoder'):
        """
        A Vision Transformer class

        :param block_kwargs: the kwargs for each ViTBlock
        :param out_block_kwargs: the kwargs for the ViTOutBlock
        :param layers: number of layers
        :param kernel_regularizer: kernel_regularizer given to both the blocks and the out block
        :param ln: whether to use layernorm or batchnorm
        :param out_regularizer: an activity_regularizer on the output of the out block
        :param only_classtoken: if true, outputs the accumelted activation of the class token after the out block
        :param name: name of the encoder, which will be used as prefix for the blocks and out block
        """
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


class MLPEncoder(MLP):
    def __init__(self, *args, out_dim=64, out_regularizer=None, kernel_regularizer='l1_l2', flatten=True, **kwargs):
        """
        A basic MLP encoder, which starts by flattening (if flatten is True)
        :param args: MLP args
        :param out_dim: output dimension
        :param out_regularizer: activity_regularizer on the output of the out layer
        :param kernel_regularizer: kernel regularizer for all layers of the MLP
        :param flatten: if true, flattens the input
        :param kwargs: MLP kwargs
        """
        super(MLPEncoder, self).__init__(*args, kernel_regularizer=kernel_regularizer, **kwargs)
        self.flatten = tf_layers.Flatten(name=self.name + '_flatten') if flatten else None
        self.out_layer = tf_layers.Dense(out_dim, activation=None, kernel_regularizer=kernel_regularizer,
                                         activity_regularizer=out_regularizer, name=self.name + '_out')

    def call(self, inputs, **kwargs):
        act = super().call(self.flatten(inputs) if self.flatten is not None else inputs, **kwargs)
        out = self.out_layer(act)
        return out


class TimeAgnosticMLP(MLPEncoder):
    def __init__(self, bins_per_frame, *args, **kwrags):
        """
        An MLP which is encoding each frame independently. Used for the Neuronal module
        :param bins_per_frame: number of bins in a single frame (usually 1)
        :param args: MLPEncoder args
        :param local:
        :param kwrags:
        """
        super().__init__(*args, flatten=False, **kwrags)
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

