import numpy as np
import os
import pandas as pd
from enum import Enum

from neuronal.utils.consts import NEURONAL_BASE_DIR, NATURAL_MOVIES, NATURAL_MOVIES_FRAMES, NATURAL_MOVIES_TRIALS, SESSIONS
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
    STIMULUS = Label("stimulus", CATEGORICAL, len(NATURAL_MOVIES), NATURAL_MOVIES)
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
        self._path = os.path.join(DATA_DIR, self.session_id)
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
            self.trials[stimulus] = [Trial(self, stimulus, trial_num)
                                     for trial_num in sorted(os.listdir(os.path.join(self._path, stimulus)))]

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
        self._path = os.path.join(DATA_DIR, self.session_id, stimulus, trial_num)
        self.stimulus = stimulus
        self.trial_num = trial_num
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

            if os.path.exists(binned_path):
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
    def __init__(self, session_id, batch_size=32, frames_per_sample=10, bins_per_frame=1,
                 stimuli=NATURAL_MOVIES, areas=None, train=True, val=False, test=False):
        super(SessionDataGenerator, self).__init__()
        self.session_id = streval(session_id)
        self.session = Session(session_id)
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.bins_per_frame = bins_per_frame
        self.spikes = {}    # {stim: {area: List[trial_activity_mat]}} if areas else {stim: List[trial_activity_mat]}
        self.areas = list(areas) if areas is not None else areas
        self.single_area = areas[0] if len(areas) == 1 else None
        self.stimuli = list(stimuli)
        self.bins_per_sample = frames_per_sample * bins_per_frame
        self.order = None
        self.num_units = None

        assert not (test and val)
        self.train = train
        self.test = test
        self.val = val

        self.__total_samples = None
        self.__load_spikes()

    @staticmethod
    def is_generator():
        return True

    def clone(self, **kwargs):
        self_kwargs = dict(session_id=self.session_id, batch_size=self.batch_size,
                           frames_per_sample=self.frames_per_sample, bins_per_frame=self.bins_per_frame,
                           stimuli=self.stimuli, areas=self.areas, train=self.train, val=self.val, test=self.test)
        self_kwargs.update(**kwargs)
        return SessionDataGenerator(**self_kwargs)

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
                if self.areas_in_spikes() is not None:
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
            normed_inds = np.arange(0, 1, len(trials)+1)
            if self.train:
                trial_mask = normed_inds <= 0.6
            elif self.val:
                trial_mask = (normed_inds >= 0.6) & (normed_inds < 0.8)
            elif self.test:
                trial_mask = normed_inds >= 0.8
            else:
                # If we want the full session
                trial_mask = np.full_like(normed_inds, True)

            self.num_units = {area: None for area in self.areas} if self.areas_in_spikes() else None

            for i, trial in trials:
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

    def __getitem__(self, idx):
        stimuli_inds = np.random.randint(len(self.stimuli), size=self.batch_size)
        spikes = {area: [] for area in self.areas} if self.areas_in_spikes() else []
        trials = np.empty(self.batch_size, dtype=int)
        frames = np.empty(self.batch_size, dtype=int)
        num_trials = {stim: len(list(self.spikes[stim].values())[0] if self.areas_in_spikes() else self.spikes[stim])
                      for stim in self.stimuli}
        for b in range(self.batch_size):
            cur_stimulus = self.stimuli[stimuli_inds[b]]
            activity_dct_or_mat = self.spikes[cur_stimulus]
            trial = np.random.choice(list(range(num_trials[cur_stimulus])))
            trials[b] = trial
            if self.areas_in_spikes():
                start_bin = None
                for area in self.areas:
                    cur_spikes = activity_dct_or_mat[area][trial]
                    if start_bin is None:
                        start_bin = np.random.randint(0, cur_spikes.shape[-1] - self.bins_per_sample + 1)
                    sample = cur_spikes[..., start_bin:start_bin + self.bins_per_sample]    # (N, T)
                    spikes[area].append(sample)
            else:
                cur_spikes = activity_dct_or_mat[trial]
                start_bin = np.random.randint(0, cur_spikes.shape[-1] - self.bins_per_sample)
                sample = cur_spikes[..., start_bin:start_bin + self.bins_per_sample]    # (N, T)
                spikes.append(sample)
            frames[b] = (start_bin + self.bins_per_sample - 1) / NATURAL_MOVIES_FRAMES[cur_stimulus]

        if self.areas_in_spikes():
            spikes = tf.convert_to_tensor(np.stack(spikes, axis=0))     # (B, N, T)
        else:
            spikes = {area: tf.convert_to_tensor(np.stack(activity, axis=0)) for area, activity in spikes.items()}

        trials = tf.convert_to_tensor(np.array(trials))
        frames = tf.convert_to_tensor(np.stack(frames, axis=0))

        return spikes, {Labels.STIMULUS.value.name: stimuli_inds,
                        Labels.TRIAL.value.name: trials,
                        Labels.FRAME.value.name: frames}
