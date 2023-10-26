from model.layers import *


class ViTEncoder:
    def __init__(self, block_kwargs, out_block_kwargs, layers=3, out_regularizer=None, only_classtoken=False, name='ViTEncoder'):
        self.blocks = [ViTBlock(name=name + f'_block{l}', **block_kwargs) for l in range(layers)]
        self.only_classtoken = only_classtoken
        self.out_block = ViTOutBlock(name=name + '_outblock', **out_block_kwargs, activity_regularizer=out_regularizer)

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
    def obsulete_method(self):
        return True
