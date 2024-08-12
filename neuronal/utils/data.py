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
        self._path = os.path.join(DATA_DIR, self.id)
        self._load()

    def _load(self):
        import json
        with open(os.path.join(self._path, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        with open(os.path.join(self._path, "start_time.txt"), 'r') as f:
            self.start_time = eval(f.read().strip())

        # Load trials
        self.trials = {}
        for stimulus in os.listdir(self._path):
            if "." in stimulus: continue
            self.trials[stimulus] = [Trial(self.id, stimulus, trial_num)
                                     for trial_num in sorted(os.listdir(os.path.join(self._path, stimulus)))]

    def get_trials(self, stimulus=None):
        if stimulus is not None:
            return {k: v.copy() for k, v in self.trials.get(stimulus, {}).items()}
        else:
            return {k: {subk: subv.copy() for subk, subv in v.items()} for k, v in self.trials.items()}


class Trial:
    def __init__(self, session_id, stimulus, trial_num):
        self._path = os.path.join(DATA_DIR, session_id, stimulus, trial_num)
        self.stimulus = stimulus
        self.session_id = session_id
        self.trial_num = trial_num
        self.spike_times = None
        self.spike_amplitudes = None
        self.running_speed = None
        self.invalid_times = None
        self.frame_start = None
        self.frame_end = None
        self._load()

    def _load(self):
        self.spike_times = loadz(os.path.join(self._path, "spike_times.npz"))
        self.spike_amplitudes = loadz(os.path.join(self._path, "spike_amplitudes.npz"))
        self.running_speed = pd.read_csv(os.path.join(self._path, "running_speed.csv"))
        self.invalid_times = pd.read_csv(os.path.join(self._path, "invalid_times.csv"))
        frame_times = loadz(os.path.join(self._path, "frame_times.npz"))
        self.frame_start = frame_times['start']
        self.frame_end = frame_times['end']




