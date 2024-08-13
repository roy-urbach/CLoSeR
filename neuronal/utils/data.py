import numpy as np
from utils.consts import *
import os
import pandas as pd


DATA_DIR = "data"


def loadz(npz_path):
    dct = {k: v for k, v in np.load(npz_path, allow_pickle=True).items()}
    return dct


class Session:
    def __init__(self, id):
        self.id = id
        self.metadata = None
        self.start_time = None
        self.trials = None
        self.units = None
        self.probes = None
        self._path = os.path.join(DATA_DIR, self.id)
        self._load()

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
        return f"<Session {self.id}>"


class Trial:
    def __init__(self, session, stimulus, trial_num):
        self.session = session
        self.session_id = self.session.id
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

    def _load_spike_bins(self, bins_per_frame=3):
        if bins_per_frame not in self.spike_bins:
            binned_path = os.path.join(self._path, f"spikes_binned_bpf_{bins_per_frame}.npz")
            bins_path = os.path.join(self._path, f"spike_bins_bpf_{bins_per_frame}.npy")

            if os.path.exists(binned_path):
                self.spike_bins[bins_per_frame] = loadz(binned_path)
                self.bins[bins_per_frame] = np.load(bins_path, allow_pickle=True)
            else:
                diff = np.diff(np.concatenate([self.get_frame_start(), self.get_frame_end()[-1:]]))
                self.bins[bins_per_frame] = np.concatenate([(self.get_frame_start()[..., None] + diff[:, None] * np.linspace(0., 1., bins_per_frame+1)[None, :-1]).flatten(),
                                                            self.get_frame_end()[-1:]])
                self.spike_bins[bins_per_frame] = {unit: np.histogram(spike_times, self.bins)[0] > 0
                                   for unit, spike_times in self.get_spike_times()}
                with open(binned_path, 'wb') as f:
                    np.savez(f, **self.spike_bins)
                with open(bins_path, "wb") as f:
                    np.save(f, self.bins)

    def get_spike_bins(self, area=None, bins_per_frame=3):
        self._load_spike_bins(bins_per_frame)
        return self._filter_by_area(self.spike_bins[bins_per_frame], area=area)

    def get_bins(self, bins_per_frame=None):
        if bins_per_frame is None:
            assert len(self.bins) == 1
            return list(self.bins)[0]
        else:
            self._load_spike_bins(bins_per_frame)
            return self.bins[bins_per_frame]

    def __repr__(self):
        return f"<Trial {self.trial_num} (session {self.session_id}, stim {self.stimulus})>"
