# ALL_BIRDS = np.array(list(filter(lambda bird: os.path.exists(os.path.join(base_path, 'train_spect', f"{bird}.npz")), sorted(os.listdir(os.path.join(base_path, 'train_audio'))))))
import abc
from enum import Enum

from auditory.utils.consts import ALL_BIRDS, N_FREQS
from utils.data import Label, CATEGORICAL, ComplicatedData, GeneratorDataset
from tqdm import tqdm as counter
from utils.modules import Modules
import os
import numpy as np

from utils.utils import streval
import random


class Labels(Enum):
    BIRD = Label("bird", CATEGORICAL, len(ALL_BIRDS))

    @staticmethod
    def get(name):
        if isinstance(name, Labels):
            return name
        else:
            relevant_labels = [label for label in Labels if name in (label.name, label.value, label.value.name)]
            if not len(relevant_labels):
                raise ValueError(f"No label named {name}")
            else:
                return relevant_labels[0]


class BirdDataset(ComplicatedData):
    PATH = os.path.join(Modules.AUDITORY.value, 'data', 'train_spect')

    def __init__(self, module: Modules = Modules.AUDITORY, birds=ALL_BIRDS, bins_per_sample=2, spects=None, files=None,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.bins_per_sample = bins_per_sample
        birds = streval(birds)
        if not isinstance(birds[0], str):
            if isinstance(birds[0], bool):
                birds = ALL_BIRDS[birds]
            else:
                birds = np.array([ALL_BIRDS[bird] for bird in birds])
        self.birds = birds
        self.spects = spects
        self.files = files

    def _set_label_to_dim(self):
        self.label_to_dim = {Labels.BIRD.value.name: len(self.birds)}

    def _set_x(self):
        if self.x is None:
            if self.spects is None:
                self.spects = {}
                # self.files = {}
                for bird in counter(self.birds):
                    with np.load(os.path.join(self.PATH, f'{bird}.npz'),
                                 allow_pickle=True) as data:
                        self.spects[bird] = data['spectrograms']
                        # self.files[bird] = data['files']

            x = []
            y = {Labels.BIRD.value.name: []}

            for birdid, bird in counter(enumerate(self.birds)):
                inds = np.arange(len(self.spects[bird]))
                if self.train:
                    mask = inds < inds.size * 0.6
                elif self.val:
                    mask = (inds >= inds.size * 0.6) & (inds < inds.size * 0.8)
                elif self.test:
                    mask = inds >= inds.size * 0.8
                else:
                    assert False, "train or val or test should be True"

                for i in np.where(mask)[0]:
                    spect = self.spects[bird][i].reshape(N_FREQS, -1)  # (N, T)
                    x.append(spect.T[np.arange(self.bins_per_sample)[None] + np.arange(
                        spect.shape[-1] - self.bins_per_sample)[:, None]])
                    y[Labels.BIRD.value.name].extend([birdid] * (spect.shape[-1] - self.bins_per_sample))

            self.x = np.concatenate(x, axis=0).transpose(0, 2, 1)
            self.y = {k: np.array(v) for k, v in y.items()}

    def get_config(self):
        return dict(**super().get_config(),
                    birds=self.birds,
                    bins_per_sample=self.bins_per_sample,
                    spects=self.spects,
                    files=self.files)


class BirdGenerator(GeneratorDataset):
    PATH = os.path.join(Modules.AUDITORY.value, 'data', 'train_spect')

    def __init__(self, birds=ALL_BIRDS, bins_per_sample=32, **kwargs):
        self.bins_per_sample = bins_per_sample
        birds = streval(birds)
        if not isinstance(birds[0], str):
            if isinstance(birds[0], bool):
                birds = ALL_BIRDS[birds]
            else:
                birds = np.array([ALL_BIRDS[bird] for bird in birds])
        self.birds = birds
        self.spects = None

        super().__init__(**kwargs)

    def _set_label_to_dim(self, *args, **kwargs):
        self.label_to_dim = {Labels.BIRD.value.name: len(self.birds)}

    def get_shape(self):
        return self.spects[self.birds[0]].shape[0], self.bins_per_sample

    def _set(self):
        if self.spects is None:
            self.spects = {}
            for bird in counter(self.birds):
                with np.load(os.path.join(self.PATH, f'{bird}.npz'), allow_pickle=True) as data:
                    self.spects[bird] = data['spectrograms']
                nspects = len(self.spects[bird])

                inds = np.arange(nspects)
                if self.train:
                    mask = inds < inds.size * 0.6
                elif self.val:
                    mask = (inds >= inds.size * 0.6) & (inds < inds.size * 0.8)
                elif self.test:
                    mask = inds >= inds.size * 0.8
                else:
                    assert False, "train or val or test should be True"

                self.spects[bird] = [spect.reshape(N_FREQS, -1) for i, spect in enumerate(self.spects[bird]) if mask[i]]

    def get_config(self):
        return dict(**super().get_config(), birds=self.birds, bins_per_sample=self.bins_per_sample)

    def __iter__(self):
        while True:
            # Sample a random batch of birds
            bird_batch = np.random.choice(len(self.birds), self.batch_size, replace=True)
            bird_batch_name = [self.birds[bird_id] for bird_id in bird_batch]
            spect_num = np.random.randint([len(self.spects[bird_name]) for bird_name in bird_batch_name])

            # Initialize batch of sequences and targets
            sequences = []

            for i, (bird_id, bird_name, spect_i) in enumerate(zip(bird_batch, bird_batch_name, spect_num)):
                spectrogram = self.spects[bird_name][spect_i]

                # Randomly sample a starting point for each spectrogram in the batch
                start_indices = random.randint(0, spectrogram.shape[-1] - self.bins_per_sample)
                sequences.append(spectrogram[:, start_indices:start_indices+self.bins_per_sample])

            y = {Labels.BIRD.value.name: bird_batch}

            actual_y = {}
            for name, label in self.name_to_label.items():
                actual_y[name] = np.array(y[(label.value if hasattr(label, 'value') else label).name] if label.value.name else np.zeros(self.batch_size))
            # Yield the batch of sequences and targets
            yield np.array(sequences), actual_y
