from utils.io_utils import load_json, save_json
from utils.io_utils import get_file_time
from utils.modules import Modules
import os
import numpy as np

RESULTS_FILE_NAME = 'classification_eval'
CS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])


@Modules.add_method
def load_evaluation_json(model_name):
    results = load_json(os.path.join(model_name, RESULTS_FILE_NAME))
    return results


@Modules.add_method
def save_evaluation_json(model_name, dct):
    save_json(os.path.join(model_name, RESULTS_FILE_NAME), dct)


@Modules.add_method
def get_evaluation_time(model_name, raw=True):
    return get_file_time(os.path.join(model_name, RESULTS_FILE_NAME) + '.json', raw=raw)
