import os
import utils.io_utils as general_io_utlis
from vision.utils.consts import VISION_CONFIG_DIR, VISION_MODELS_DIR


def load_json(fn, base_path=VISION_CONFIG_DIR):
    return general_io_utlis.load_json(os.path.join(base_path, fn))


def save_json(fn, dct, base_path=VISION_CONFIG_DIR, indent=4, **kwargs):
    return general_io_utlis.save_json(os.path.join(base_path, fn), dct, indendt=indent, **kwargs)


def load_output(model_name):
    return general_io_utlis.load_output(f"{VISION_MODELS_DIR}/{model_name}")


def get_output_time(model_name, raw=True):
    return general_io_utlis.get_output_time(f"{VISION_MODELS_DIR}/{model_name}", raw=raw)
