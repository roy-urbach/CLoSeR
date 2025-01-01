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

    def __init__(self, birds=ALL_BIRDS, bins_per_sample=32, normalize=False, **kwargs):
        self.bins_per_sample = bins_per_sample
        birds = streval(birds)
        if not isinstance(birds[0], str):
            if isinstance(birds[0], bool):
                birds = ALL_BIRDS[birds]
            else:
                birds = np.array([ALL_BIRDS[bird] for bird in birds])
        self.birds = birds
        self.spects = None
        self.spects_length = None
        self.num_spects = None
        self.normalize = normalize

        super().__init__(**kwargs)

    def get_output_dtypes(self):
        return {name: int for name in self.name_to_label}

    def get_output_shapes(self):
        pass

    def _set_label_to_dim(self, *args, **kwargs):
        self.label_to_dim = {Labels.BIRD.value.name: len(self.birds)}

    def get_shape(self):
        return self.spects[self.birds[0]][0].shape[0], self.bins_per_sample

    def _normalize(self, spect):
        spect -= np.nanmin(spect)
        spect /= np.quantile(spect, 0.95)
        return spect

    def _set(self):
        if self.spects is None:
            self.spects = {}
            self.spects_length = {}
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
                self.spects[bird] = [self._normalize(spect) if self.normalize else spect for spect in self.spects[bird] if spect.shape[-1] > self.bins_per_sample]
                self.spects_length[bird] = np.array([spect.shape[-1] for spect in self.spects[bird]])
            self.num_spects = np.array([len(self.spects[bird]) for bird in self.birds])

    def get_config(self):
        return dict(**super().get_config(), birds=self.birds, bins_per_sample=self.bins_per_sample)

    def __iter__(self):
        while True:

            # random sample for 1000 steps
            random_cache = 1000

            # Sample a random batch of birds
            bird_batch_cache = np.random.choice(len(self.birds), (random_cache, self.batch_size), replace=True)
            bird_batch_name_cache = np.reshape([self.birds[bird_id]
                                                for bird_id in bird_batch_cache.flatten()], bird_batch_cache.shape)
            spect_num_cache = np.random.randint(self.num_spects[bird_batch_cache])
            spects_length_cache = np.array([self.spects_length[bird_name][spect_id]
                                            for bird_name, spect_id in zip(bird_batch_name_cache.flatten(),
                                                                           spect_num_cache.flatten())]).reshape(bird_batch_cache.shape)
            start_indices_cache = np.random.randint(spects_length_cache - self.bins_per_sample)
            del spects_length_cache

            for step, (bird_batch, bird_batch_name, spect_num, start_indices) in enumerate(zip(bird_batch_cache,
                                                                                               bird_batch_name_cache,
                                                                                               spect_num_cache,
                                                                                               start_indices_cache
                                                                                               )):
                # Initialize batch of sequences and targets
                sequences = [self.spects[bird][spect_id][:, start_ind:start_ind+self.bins_per_sample]
                             for bird, spect_id, start_ind in zip(bird_batch_name, spect_num, start_indices)]

                y = {Labels.BIRD.value.name: bird_batch}

                if self.name_to_label:
                    actual_y = {}
                    for name, label in self.name_to_label.items():
                        actual_y[name] = np.array(y[(label.value if hasattr(label, 'value') else label).name] if label.value.name else np.zeros(self.batch_size))
                else:
                    actual_y = y
                # Yield the batch of sequences and targets
                yield np.array(sequences), actual_y
