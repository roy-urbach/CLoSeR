from utils.io_utils import load_json, save_json

RESULTS_FILE_NAME = 'classification_eval'


def load_evaluation_json(model_name):
    base_path = f'models/{model_name}'
    results = load_json(RESULTS_FILE_NAME, base_path=base_path)
    return results


def save_evaluation_json(model_name, dct):
    base_path = f'models/{model_name}'
    save_json(RESULTS_FILE_NAME, dct, base_path=base_path)
