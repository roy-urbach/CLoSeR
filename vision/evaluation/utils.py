from utils.io_utils import load_json, save_json
from utils.io_utils import get_file_time
from vision.utils.consts import VISION_MODELS_DIR

RESULTS_FILE_NAME = 'classification_eval'


def load_evaluation_json(model_name):
    results = load_json(f'{VISION_MODELS_DIR}/{model_name}/{RESULTS_FILE_NAME}')
    return results


def save_evaluation_json(model_name, dct):
    save_json(f'{VISION_MODELS_DIR}/{model_name}', dct)


def get_evaluation_time(model_name, raw=True):
    return get_file_time(f'{VISION_MODELS_DIR}/{model_name}/{RESULTS_FILE_NAME}.json', raw=raw)
