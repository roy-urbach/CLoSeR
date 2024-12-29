import neuronal.model.model as neur_model
import neuronal.model.losses as neur_losses
from auditory.utils.consts import N_FREQS, SR
from auditory.utils.data import Labels
from mp3_to_spect import FMIN, FMAX
from utils.model.layers import SplitPathways
from utils.modules import Modules
from utils.model.augmentations import MelSpectrogramAugmenter
import tensorflow as tf


class SplitPathwaysAuditory(SplitPathways):
    """
    Call
    :param inputs: (B, S, T)
    :return: (B, d*S, N, DIM)
    """

    def __init__(self, num_units, spatial_k=1, n=2, d=0.5, intersection=True, fixed=False, seed=0, axis=-2, **kwargs):
        if isinstance(num_units, dict):
            raise NotImplementedError()
        import warnings
        if num_units % spatial_k:
            warnings.warn(f"num units ({num_units}) / spatial_k ({spatial_k}) isn't an int, still running")
        super().__init__(num_signals=num_units//spatial_k, n=n, d=d, intersection=intersection,
                         fixed=fixed, seed=seed, class_token=False, axis=axis, **kwargs)
        self.num_units = num_units // spatial_k
        self.full_units = num_units
        self.spatial_k = spatial_k
        self.expected_size = int(d * self.units) * self.spatial_k

    def call(self, inputs, training=False):
        # (B, N, T)
        T = inputs.shape[-1]
        inputs_reshape = tf.transpose(tf.reshape(inputs, (-1, self.num_units, self.spatial_k, T)),
                                      [0, 2, 1, 3])  # (B, K, N/K, T)
        paths = super().call(inputs_reshape)    # (B, K, d*N/K, P, T)
        flattened = tf.reshape(paths, (-1, self.expected_size, self.n, T))   # (B, d*N, P, T)
        return flattened


def create_model(input_shape, name='auditory_model', encoder='TimeAgnosticMLP',
                 labels=(Labels.BIRD, ), module=Modules.AUDITORY, augmentation_kwargs={}, **kwargs):
    pink_noise_w = augmentation_kwargs.get('pink_noise_w', 0)
    white_noise_w = augmentation_kwargs.get('white_noise_w', 0)
    if pink_noise_w or white_noise_w:
        augmentation_kwargs['augmentations'] = [MelSpectrogramAugmenter(sr=SR, n_mels=N_FREQS, f_min=FMIN, f_max=FMAX,
                                                                        pink_noise_factor=pink_noise_w,
                                                                        white_noise_factor=white_noise_w)]
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_model.create_model(input_shape, name=name, encoder=encoder, labels=labels, module=module,
                                   augmentation_kwargs=augmentation_kwargs, SplitClass=SplitPathwaysAuditory, **kwargs)


def compile_model(model, dataset, loss=neur_losses.LPL, labels=(Labels.BIRD, ), **kwargs):
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_model.compile_model(model, dataset, loss=loss, labels=labels, **kwargs)
