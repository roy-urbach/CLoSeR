import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import backend as K

from utils.tf_utils import set_seed, serialize
from utils.utils import *


# ViT layers

@serialize
class MLP(tf_layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1, kernel_regularizer='l1_l2', **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dense = [tf_layers.Dense(units, activation=tf.nn.gelu, kernel_regularizer=kernel_regularizer, name=self.name + f'_fc{i}')
                      for i, units in enumerate(self.hidden_units)]
        self.dropout = [tf_layers.Dropout(dropout_rate, name=self.name + f'_do{i}')
                        for i, _ in enumerate(self.hidden_units)]
        self.depth = len(hidden_units)

    def call(self, inputs, training=None):
        x = inputs
        for l in range(self.depth):
            x = self.dense[l](x)
            x = self.dropout[l](x, training=training)
        return x

    def get_config(self):
        return dict(**super().get_config(), hidden_units=self.hidden_units)


@serialize
class Patches(tf_layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        return dict(super().get_config(), patch_size=self.patch_size)


@serialize
class PatchEncoder(tf_layers.Layer):
    def __init__(self, num_patches=196, projection_dim=768, num_class_tokens=1, kernel_regularizer='l1_l2', **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_class_tokens = num_class_tokens
        class_token = tf.random_normal_initializer()(shape=(self.num_class_tokens, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = tf_layers.Dense(units=projection_dim, kernel_regularizer=kernel_regularizer)
        self.position_embedding = tf_layers.Embedding(input_dim=num_patches + 1,
                                                      embeddings_regularizer=kernel_regularizer,
                                                      output_dim=projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embeddings
        class_token = tf.tile(self.class_token, multiples=[batch, 1])
        class_token = tf.reshape(class_token, (batch, self.num_class_tokens, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([class_token, patches_embed], 1)
        # calculate positional embeddings
        positions = tf.concat([tf.zeros(self.num_class_tokens, dtype=tf.int32),
                               tf.range(start=1, limit=self.num_patches + 1, delta=1)], axis=0)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded

    def get_config(self):
        return dict(**super().get_config(), num_patches=self.num_patches, projection_dim=self.projection_dim,
                    num_class_tokens=self.num_class_tokens)


@serialize
class ViTBlock(tf_layers.Layer):
    def __init__(self, num_heads=4, projection_dim=64, dropout_rate=0.1, kernel_regularizer='l1_l2', **kwargs):
        super(ViTBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.bn1 = tf_layers.BatchNormalization(name=self.name + '_bn1')
        self.mh_attn = tf_layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=projection_dim,
                                                    dropout=dropout_rate,
                                                    kernel_regularizer=kernel_regularizer,
                                                    name=self.name + '_mhattn')
        self.add1 = tf_layers.Add(name=self.name + '_add1')
        self.bn2 = tf_layers.BatchNormalization(name=self.name + '_bn2')
        self.mlp = MLP([projection_dim * 2, projection_dim], dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer)
        self.add2 = tf_layers.Add(name=self.name + '_add2')

    def get_config(self):
        return dict(super().get_config(), num_heads=self.num_heads,
                    projection_dim=self.projection_dim, dropout_rate=self.dropout_rate)

    def call(self, encoded_patches):
        x1 = self.bn1(encoded_patches)
        attention_output = self.mh_attn(x1, x1)
        x2 = self.add1([attention_output, encoded_patches])
        x3 = self.bn2(x2)
        x3 = self.mlp(x3)
        out = self.add2([x2, x3])
        return out


@serialize
class ViTOutBlock(tf_layers.Layer):
    def __init__(self, dropout_rate=0.1, output_dim=512, mlp_head_units=(2048, 1024), reg=0, kernel_regularizer='l1_l2', **kwargs):
        super(ViTOutBlock, self).__init__(**kwargs)
        # Create a [batch_size, projection_dim] tensor.
        self.bn = tf_layers.BatchNormalization()
        self.fl = tf_layers.Flatten()
        self.dropout = tf_layers.Dropout(dropout_rate)
        self.output_dim = output_dim
        self.mlp = MLP(mlp_head_units, dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer)
        self.dense = tf_layers.Dense(self.output_dim, kernel_regularizer=kernel_regularizer)
        self.reg = reg
        self.mlp_head_units = mlp_head_units

    def call(self, encoded_patches):
        x = self.bn(encoded_patches)
        x = self.fl(x)
        x = self.dropout(x)
        x = self.mlp(x)
        out = self.dense(x)
        return out

    def get_config(self):
        return {**super().get_config(),
                'activity_regularizer': self.activity_regularizer,
                'mlp_head_units': self.mlp_head_units,
                'output_dim': self.output_dim}


# "our" layers

@serialize
class SplitPathways(tf_layers.Layer):
    def __init__(self, num_patches, token_per_path=False, n=2, d=0.5, intersection=True, fixed=False,
                 seed=0, class_token=True, pathway_to_cls=None, **kwargs):
        super(SplitPathways, self).__init__(**kwargs)
        assert intersection or (n*d) <= 1
        self.n = n
        self.seed = seed
        self.fixed = fixed
        self.num_patches = num_patches
        self.token_per_path = token_per_path
        if pathway_to_cls is not None:
            if isinstance(pathway_to_cls, str):
                pathway_to_cls = eval(pathway_to_cls)
            self.pathway_to_cls = tf.constant(pathway_to_cls)
        if class_token and pathway_to_cls is None:
            if not token_per_path:
                self.pathway_to_cls = tf.zeros(n, dtype=tf.int32)
            else:
                self.pathway_to_cls = tf.range(n, dtype=tf.int32)
        self.num_patches_per_path = int(num_patches * d)
        self.intersection = intersection
        self.class_token = class_token
        self.shift = (tf.reduce_max(self.pathway_to_cls) + 1) if class_token else 0
        self.indices = None
        if fixed:
            set_seed(self.seed)
            self.get_indices()

    def get_config(self):
        return dict(**super().get_config(), n=self.n, seed=self.seed, fixed=self.fixed,
                    num_patches=self.num_patches, intersection=self.intersection,
                    shift=self.shift, class_token=self.class_token, pathway_to_cls=self.pathway_to_cls)

    def get_indices(self):
        if self.indices is None:
            if self.intersection:
                indices = tf.stack(
                    [tf.random.shuffle(tf.range(self.shift, self.num_patches + self.shift))[:self.num_patches_per_path]
                     for _ in range(self.n)],
                    axis=-1)
            else:
                indices = tf.reshape(
                    tf.random.shuffle(tf.range(self.shift,
                                               self.num_patches + self.shift))[:self.num_patches_per_path * self.n],
                    (-1, self.n))

            if self.fixed:
                self.indices = indices
        else:
            indices = self.indices

        # everyone gets the class token
        if self.class_token:
            # cls_tokens_to_add = tf.range(self.n, dtype=indices.dtype)[None] if self.token_per_path else tf.zeros(
            #     (1, self.n),
            #     dtype=indices.dtype)
            indices = tf.concat([self.pathway_to_cls[None], indices], axis=0)
        return indices

    def call(self, inputs, training=False):
        if not training:
            set_seed(self.seed)

        indices = self.get_indices()

        return tf.gather(inputs, indices, axis=-2, batch_dims=0)


@serialize
class LayerNormalization(keras.layers.Layer):
    """
    Taken from:
    https://github.com/CyberZHG/keras-layer-normalization/blob/master/keras_layer_normalization/layer_normalization.py
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs
