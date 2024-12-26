import neuronal.model.model as neur_model
import neuronal.model.losses as neur_losses
from auditory.utils.consts import N_FREQS, SR
from auditory.utils.data import Labels
from mp3_to_spect import FMIN, FMAX
from utils.modules import Modules
from utils.model.augmentations import MelSpectrogramAugmenter


def create_model(input_shape, name='auditory_model', encoder='TimeAgnosticMLP',
                 labels=(Labels.BIRD, ), module=Modules.AUDITORY, augmentation_kwargs={}, **kwargs):
    pink_noise_w = augmentation_kwargs.get('pink_noise_w', 0)
    white_noise_w = augmentation_kwargs.get('white_noise_w', 0)
    if pink_noise_w or white_noise_w:
        augmentation_kwargs['augmentations'] = [MelSpectrogramAugmenter(sr=SR, n_mels=N_FREQS, f_min=FMIN, f_max=FMAX,
                                                                        pink_noise_factor=pink_noise_w, white_noise_factor=white_noise_w)]
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_model.create_model(input_shape, name=name, encoder=encoder, labels=labels, module=module,
                                   augmentation_kwargs=augmentation_kwargs, **kwargs)


def compile_model(model, dataset, loss=neur_losses.LPL, labels=(Labels.BIRD, ), **kwargs):
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_model.compile_model(model, dataset, loss=loss, labels=labels, **kwargs)
