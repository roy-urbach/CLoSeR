import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tf_layers

from utils.tf_utils import set_seed, serialize
from utils.utils import *


# ViT layers

@serialize
class MLP(tf_layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dense = [tf_layers.Dense(units, activation=tf.nn.gelu, name=self.name + f'_fc{i}')
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


@serialize
class PatchEncoder(tf_layers.Layer):
    def __init__(self, num_patches=196, projection_dim=768, num_class_tokens=1, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_class_tokens = num_class_tokens
        class_token = tf.random_normal_initializer()(shape=(self.num_class_tokens, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = tf_layers.Dense(units=projection_dim)
        self.position_embedding = tf_layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embeddings
        class_token = tf.tile(self.class_token, multiples=[batch, 1])
        class_token = tf.reshape(class_token, (batch, self.num_class_tokens, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([class_token, patches_embed], 1)
        # calcualte positional embeddings
        positions = tf.concat(
            [tf.zeros(self.num_class_tokens, dtype=tf.int32), tf.range(start=0, limit=self.num_patches, delta=1)],
            axis=0)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded


@serialize
class ViTBlock(tf_layers.Layer):
    def __init__(self, num_heads=4, projection_dim=64, dropout_rate=0.1, **kwargs):
        super(ViTBlock, self).__init__(**kwargs)
        self.ln1 = tf_layers.LayerNormalization(epsilon=1e-6, name=self.name + '_ln1')
        self.mh_attn = tf_layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=projection_dim,
                                                    dropout=dropout_rate, name=self.name + '_mhattn')
        self.add1 = tf_layers.Add(name=self.name + '_add1')
        self.ln2 = tf_layers.LayerNormalization(epsilon=1e-6, name=self.name + '_ln2')
        self.mlp = MLP([projection_dim * 2, projection_dim], dropout_rate=dropout_rate)
        self.add2 = tf_layers.Add(name=self.name + '_add2')

    def call(self, encoded_patches):
        x1 = self.ln1(encoded_patches)
        attention_output = self.mh_attn(x1, x1)
        x2 = self.add1([attention_output, encoded_patches])
        x3 = self.ln2(x2)
        x3 = self.mlp(x3)
        out = self.add2([x2, x3])
        return out


@serialize
class ViTOutBlock(tf_layers.Layer):
    def __init__(self, dropout_rate=0.1, output_dim=512, mlp_head_units=(2048, 1024), reg=0, **kwargs):
        super(ViTOutBlock, self).__init__(**kwargs)
        # Create a [batch_size, projection_dim] tensor.
        self.ln = tf_layers.LayerNormalization(epsilon=1e-6)
        self.fl = tf_layers.Flatten()
        self.dropout = tf_layers.Dropout(dropout_rate)
        self.output_dim = output_dim
        self.mlp = MLP(mlp_head_units, dropout_rate=dropout_rate)
        self.dense = tf_layers.Dense(self.output_dim)
        self.reg = reg
        self.mlp_head_units = mlp_head_units

    def call(self, encoded_patches):
        x = self.ln(encoded_patches)
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
    def __init__(self, num_patches, token_per_path=False, n=2, d=0.5, intersection=True, fixed=False, seed=0, old=True,
                 **kwargs):
        super(SplitPathways, self).__init__(**kwargs)
        assert intersection or n == 2
        self.n = n
        self.seed = seed
        self.fixed = fixed
        self.num_patches = num_patches
        self.token_per_path = token_per_path
        self.num_patches_per_path = int(num_patches * d)
        self.intersection = intersection
        self.old = old
        self.indices = None
        if fixed:
            set_seed(self.seed)
            self.get_indices()

    def get_indices(self):
        if self.indices is None:
            shift = (1 - self.old) * (self.n if self.token_per_path else 1)
            if self.intersection:
                indices = tf.stack(
                    [tf.random.shuffle(tf.range(shift, self.num_patches + shift))[:self.num_patches_per_path] for _ in
                     range(self.n)],
                    axis=-1)
            else:
                indices = tf.reshape(
                    tf.random.shuffle(tf.range(shift, self.num_patches + shift))[
                    :self.num_patches - (self.num_patches % self.n)],
                    (-1, self.n))

            if self.fixed:
                self.indices = indices
        else:
            indices = self.indices

        # everyone gets the class token
        if not self.old:
            cls_tokens_to_add = tf.range(self.n, dtype=indices.dtype)[None] if self.token_per_path else tf.zeros(
                (1, self.n), dtype=indices.dtype)
            indices = tf.concat([cls_tokens_to_add, indices], axis=0)
        return indices

    def call(self, inputs, training=False):
        if not training or self.fixed:
            set_seed(self.seed)
            tf.keras.utils.set_random_seed(self.seed)

        indices = self.get_indices()

        return tf.gather(inputs, indices, axis=-2, batch_dims=0)
