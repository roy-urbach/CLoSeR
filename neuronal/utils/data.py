import numpy as np
import os
import pandas as pd
from enum import Enum

from neuronal.utils.consts import NEURONAL_BASE_DIR, NATURAL_MOVIES, NATURAL_MOVIES_FRAMES, \
    NATURAL_MOVIES_TRIALS, SESSIONS, BLOCKS
import tensorflow as tf

from utils.utils import streval

DATA_DIR = f"{NEURONAL_BASE_DIR}/data"
CATEGORICAL = "categorical"
CONTINUOUS = 'continuous'


class Label:
    def __init__(self, name, kind, dimension, meaning=None):
        self.name = name
        self.kind = kind
        self.dimension = dimension
        self.meaning = meaning


class Labels(Enum):
    STIMULUS = Label("stimulus", CATEGORICAL, 1 if len(NATURAL_MOVIES) <= 2 else len(NATURAL_MOVIES), NATURAL_MOVIES)
    TRIAL = Label("trial", CATEGORICAL, max(NATURAL_MOVIES_TRIALS.values()))
    FRAME = Label("normedframe", CONTINUOUS, 1)


def loadz(npz_path):
    dct = {k: v for k, v in np.load(npz_path, allow_pickle=True).items()}
    return dct


class Session:
    def __init__(self, session_id):
        assert session_id in SESSIONS
        self.session_id = session_id
        self.metadata = None
        self.start_time = None
        self.trials = None
        self.units = None
        self.probes = None
        self._path = os.path.join(DATA_DIR, str(self.session_id))
        self._load()

    def get_id(self):
        return self.session_id

    def _load(self):
        import json
        with open(os.path.join(self._path, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        with open(os.path.join(self._path, "start_time.txt"), 'r') as f:
            self.start_time = f.read()

        # Load trials
        self.trials = {}
        for stimulus in os.listdir(self._path):
            if "." in stimulus: continue
            trial_nums = sorted([int(num) for num in os.listdir(os.path.join(self._path, stimulus))])
            self.trials[stimulus] = [Trial(self, stimulus, trial_num) for trial_num in trial_nums]

        self.units = pd.read_csv(os.path.join(self._path, "units.csv"))
        self.probes = pd.read_csv(os.path.join(self._path, "probes.csv"))

    def get_trials(self, stimulus=None):
        if stimulus is not None:
            return self.trials.get(stimulus, [])[:]
        else:
            return {k: v[:] for k, v in self.trials.items()}

    def get_units(self):
        return self.units

    def get_area_units(self, area):
        return self.units[self.units.ecephys_structure_acronym == area].unit_id.to_numpy()

    def __repr__(self):
        return f"<Session {self.session_id}>"


class Trial:
    def __init__(self, session, stimulus, trial_num):
        self.session = session
        self.session_id = self.session.get_id()
        self._path = os.path.join(DATA_DIR, str(self.session_id), stimulus, str(trial_num))
        self.stimulus = stimulus
        self.trial_num = int(trial_num)
        self.spike_bins = {}
        self.bins = {}
        self.spike_times = None
        self.spike_amplitudes = None
        self.running_speed = None
        self.invalid_times = None
        self.frame_start = None
        self.frame_end = None

    def _load_spike_times(self):
        if self.spike_times is None:
            self.spike_times = loadz(os.path.join(self._path, "spike_times.npz"))
            self.spike_times = {int(k): v for k, v in self.spike_times.items()}

    def _load_spike_amplitudes(self):
        if self.spike_amplitudes is None:
            self.spike_amplitudes = loadz(os.path.join(self._path, "spike_amplitudes.npz"))
            self.spike_amplitudes = {int(k): v for k, v in self.spike_amplitudes.items()}

    def _load_running_speed(self):
        if self.running_speed is None:
            self.running_speed = pd.read_csv(os.path.join(self._path, "running_speed.csv"))

    def _load_invalid_times(self):
        if self.invalid_times is None:
            self.invalid_times = pd.read_csv(os.path.join(self._path, "invalid_times.csv"))

    def _load_frame_start(self):
        if self.frame_start is None:
            frame_times = loadz(os.path.join(self._path, "frame_times.npz"))
            self.frame_start = frame_times['start']
            self.frame_end = frame_times['end']

    def _load_frame_end(self):
        if self.frame_start is None:
            frame_times = loadz(os.path.join(self._path, "frame_times.npz"))
            self.frame_start = frame_times['start']
            self.frame_end = frame_times['end']

    def _filter_by_area(self, dct, area=None):
        if area is None:
            return dct
        else:
            area_units = self.session.get_area_units(area)
            return {unit: spikes for unit, spikes in dct.items() if unit in area_units}

    def get_spike_times(self, area=None):
        self._load_spike_times()
        return self._filter_by_area(self.spike_times, area=area)

    def get_spike_amplitudes(self, area=None):
        self._load_spike_amplitudes()
        return self._filter_by_area(self.spike_amplitudes, area=area)

    def get_running_speed(self):
        self._load_running_speed()
        return self.running_speed

    def get_invalid_times(self):
        self._load_invalid_times()
        return self.invalid_times

    def get_frame_start(self):
        self._load_frame_start()
        return self.frame_start

    def get_frame_end(self):
        self._load_frame_end()
        return self.frame_end

    def _load_spike_bins(self, bins_per_frame=3, override=False):
        if bins_per_frame not in self.spike_bins or override:
            binned_path = os.path.join(self._path, f"spikes_binned_bpf_{bins_per_frame}.npz")
            bins_path = os.path.join(self._path, f"spike_bins_bpf_{bins_per_frame}.npy")

            if os.path.exists(binned_path) and not override:
                self.spike_bins[bins_per_frame] = loadz(binned_path)
                self.spike_bins[bins_per_frame] = {int(unit): spikes for unit, spikes in self.spike_bins[bins_per_frame].items()}
                self.bins[bins_per_frame] = np.load(bins_path, allow_pickle=True)
            else:
                diff = np.diff(np.concatenate([self.get_frame_start(), self.get_frame_end()[-1:]]))
                self.bins[bins_per_frame] = np.concatenate([(self.get_frame_start()[..., None] + diff[:, None] * np.linspace(0., 1., bins_per_frame+1)[None, :-1]).flatten(),
                                                            self.get_frame_end()[-1:]])
                self.spike_bins[bins_per_frame] = {unit: np.histogram(spike_times, self.bins[bins_per_frame])[0]
                                   for unit, spike_times in self.get_spike_times().items()}
                with open(binned_path, 'wb') as f:
                    np.savez(f, **{str(unit): spikes for unit, spikes in self.spike_bins[bins_per_frame].items()})
                with open(bins_path, "wb") as f:
                    np.save(f, self.bins[bins_per_frame])

    def get_spike_bins(self, area=None, bins_per_frame=3, as_matrix=False, **kwargs):
        self._load_spike_bins(bins_per_frame, **kwargs)
        unit_to_bins = self._filter_by_area(self.spike_bins[bins_per_frame], area=area)
        if as_matrix:
            return np.stack([unit_to_bins[unit] for unit in sorted(unit_to_bins.keys())], axis=0)
        else:
            return unit_to_bins

    def get_bins(self, bins_per_frame=None):
        if bins_per_frame is None:
            assert len(self.bins) == 1
            return list(self.bins)[0]
        else:
            self._load_spike_bins(bins_per_frame)
            return self.bins[bins_per_frame]

    def __repr__(self):
        return f"<Trial {self.trial_num} (session {self.session_id}, stim {self.stimulus})>"


class SessionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, session_id, frames_per_sample=10, bins_per_frame=1,
                 stimuli=NATURAL_MOVIES, areas=None, train=True, val=False, test=False, binary=False, random=True):
        super(SessionDataGenerator, self).__init__()
        self.session_id = streval(session_id)
        if self.session_id not in SESSIONS:
            assert self.session_id in np.arange(len(SESSIONS))
            self.session_id = SESSIONS[self.session_id]
        self.session = Session(self.session_id)
        self.frames_per_sample = frames_per_sample
        self.bins_per_frame = bins_per_frame
        self.spikes = {}    # {stim: {area: List[trial_activity_mat]}} if areas else {stim: List[trial_activity_mat]}
        self.areas = list(areas) if areas is not None else areas
        self.single_area = areas[0] if areas and len(areas) == 1 else None
        self.stimuli = np.array(stimuli)
        self.bins_per_sample = frames_per_sample * bins_per_frame
        self.order = None
        self.num_units = None
        self.binary = binary
        self.possible_trials = {}
        self.random = random

        assert not (test and val)
        self.train = train
        self.test = test
        self.val = val
        self.x = None
        self.y = None

        self.name_to_label = {}

        self.__total_samples = None
        self.__load_spikes()

    @staticmethod
    def is_generator():
        return True

    def clone(self, **kwargs):
        self_kwargs = dict(session_id=self.session_id, frames_per_sample=self.frames_per_sample,
                           bins_per_frame=self.bins_per_frame, stimuli=self.stimuli, areas=self.areas, train=self.train,
                           val=self.val, test=self.test)
        self_kwargs.update(**kwargs)
        clone = SessionDataGenerator(**self_kwargs)
        clone.name_to_label = {k: v for k, v in self.name_to_label.items()}
        return clone

    def get_train(self):
        return self.clone(train=True, val=False, test=False)

    def get_validation(self):
        return self.clone(train=False, val=True, test=False)

    def get_test(self):
        return self.clone(train=False, val=False, test=True)

    def __len__(self):
        if self.__total_samples is None:
            total = 0
            for stim, act in self.spikes.items():
                if self.areas_in_spikes():
                    arr = act[self.areas[0]]
                else:
                    arr = act

                num_trials = len(arr)
                num_samples_in_trial = arr[0].shape[-1] - self.bins_per_sample + 1
                total += num_trials * num_samples_in_trial
            self.__total_samples = total
        return self.__total_samples

    def __load_spikes(self):
        for stimulus in self.stimuli:
            self.spikes[stimulus] = {area: [] for area in self.areas} if self.areas_in_spikes() else []

            trials = self.session.get_trials(stimulus)
            normed_inds = np.linspace(0, BLOCKS, len(trials)+1) % 1
            if self.train:
                trial_mask = normed_inds <= 0.6
            elif self.val:
                trial_mask = (normed_inds >= 0.6) & (normed_inds < 0.8)
            elif self.test:
                trial_mask = normed_inds >= 0.8
            else:
                # If we want the full session
                trial_mask = np.full_like(normed_inds, True)
            self.possible_trials[stimulus] = np.where(trial_mask)[0]

            self.num_units = {area: None for area in self.areas} if self.areas_in_spikes() else None

            for i, trial in enumerate(trials):
                if not trial_mask[i]: continue
                if self.areas is not None:
                    if self.single_area:
                        self.spikes[stimulus].append(trial.get_spike_bins(area=self.single_area,
                                                                          bins_per_frame=self.bins_per_frame,
                                                                          as_matrix=True))
                        if self.num_units is None:
                            self.num_units = len(self.spikes[stimulus][-1])
                    else:
                        for area in self.areas:
                            self.spikes[stimulus][area].append(trial.get_spike_bins(area=area,
                                                                                    bins_per_frame=self.bins_per_frame,
                                                                                    as_matrix=True))
                            if self.num_units[area] is None:
                                self.num_units[area] = len(self.spikes[stimulus][area][-1])

                else:
                    self.spikes[stimulus].append(trial.get_spike_bins(bins_per_frame=self.bins_per_frame,
                                                                      as_matrix=True))
                    if self.num_units is None:
                        self.num_units = len(self.spikes[stimulus][-1])

    def get_shape(self):
        if self.areas_in_spikes():
            return {area: (self.num_units[area], self.bins_per_sample) for area in self.areas}
        else:
            return self.num_units, self.bins_per_sample

    def areas_in_spikes(self):
        return self.areas is not None and not self.single_area

    def update_name_to_label(self, name, label):
        self.name_to_label[name] = label

    def get_activity_window(self, stim_name, trial_num, frame_num):

        last_bin = frame_num + self.bins_per_frame
        first_bin = last_bin - self.bins_per_sample

        if self.areas_in_spikes():
            spikes = {area: self.spikes[stim_name][area][trial_num][..., first_bin:last_bin] for area in self.areas}
        else:
            spikes = self.spikes[stim_name][trial_num][..., first_bin:last_bin]  # (N, T))

        return spikes

    def sample(self, idx):
        stim_ind = np.random.randint(len(self.stimuli))
        stim_name = self.stimuli[stim_ind]
        trial = np.random.randint(len(self.spikes[stim_name]) if not self.areas_in_spikes() else len(self.spikes[stim_name][self.areas[0]]))
        frame = np.random.randint(self.frames_per_sample, NATURAL_MOVIES_FRAMES[stim_name])

        spikes = self.get_activity_window(stim_name, trial, frame)

        labels = {Labels.STIMULUS.value.name: np.array(stim_ind),
                  Labels.TRIAL.value.name: np.array(self.possible_trials[stim_name][trial]),
                  Labels.FRAME.value.name: np.array(frame / NATURAL_MOVIES_FRAMES[stim_name])}

        y = {}
        for name, label in self.name_to_label.items():
            y[name] = labels[label.value.name]

        if self.binary:
            spikes = (spikes > 0).astype(np.float32)
        else:
            spikes = spikes.astype(np.float32)

        return spikes, y

    def get_x(self, labels=None):
        if self.x is None:
            spikes = {area: [] for area in self.areas} if self.areas_in_spikes() else []
            y = {k.value.name: [] for k in Labels}

            for stim_ind, stim_name in enumerate(self.stimuli):
                for trial_num in range(len(self.spikes[stim_name]) if not self.areas_in_spikes() else len(self.spikes[stim_name][self.areas[0]])):
                    for frame_num in range(self.frames_per_sample, NATURAL_MOVIES_FRAMES[stim_name]):
                        cur_spikes = self.get_activity_window(stim_name, trial_num, frame_num)
                        valid = True
                        if self.areas_in_spikes():
                            for area in spikes.keys():
                                valid = bool(spikes[area].size)
                                if not valid: break
                                spikes[area].append(cur_spikes[area])
                        else:
                            valid = bool(spikes.size)
                            if valid:
                                spikes.append(cur_spikes)

                        if valid:
                            y[Labels.STIMULUS.value.name].append(stim_ind)
                            y[Labels.TRIAL.value.name].append(self.possible_trials[stim_name][trial_num])
                            y[Labels.FRAME.value.name].append(frame_num / NATURAL_MOVIES_FRAMES[stim_name])

            if self.areas_in_spikes():
                spikes = {area: np.stack(activity, axis=0) for area, activity in spikes.items()}
            else:
                spikes = np.stack(spikes, axis=0)
            self.x = spikes

            actual_y = {}
            for name, label in self.name_to_label.items() if labels is None else {label.value.name: label
                                                                                  for label in labels}.items():
                actual_y[name] = np.array(y[label.value.name])
            self.y = y

        return self.x

    def get_y(self, labels=None):
        if self.y is None:
            self.get_x(labels=labels)
        return self.y

    def __getitem__(self, idx):
        if self.random:
            return self.sample(idx)
        else:
            x = self.get_x()
            y = self.get_y()
            if self.areas_in_spikes():
                cur_x = {area: x[idx] for area in self.areas}
            else:
                cur_x = x[idx]
            cur_y = y[idx]
            return cur_x, cur_y
