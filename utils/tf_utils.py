import numpy as np
import tensorflow as tf
import os
import random

from utils.io_utils import load_json
from utils.modules import Modules


def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)


def get_model_fn(model_or_name, module:Modules):
    if isinstance(model_or_name, str):
        model_name = model_or_name
    else:
        assert hasattr(model_or_name, 'name')
        model_name = model_or_name.name
    fn = os.path.join(module.get_models_path(), model_name, 'model.h5')
    return fn


def get_weights_fn(model_or_name, module:Modules):
    if isinstance(model_or_name, str):
        model_name = model_or_name
    else:
        assert hasattr(model_or_name, 'name')
        model_name = model_or_name.name
    fn = os.path.join(module.get_models_path(), model_name, "checkpoints", 'model_weights_{epoch}')
    return fn


@Modules.add_method
def history_fn_name(model):
    return os.path.join(model, "checkpoints/history.json")


@Modules.add_method
def load_history(model):
    return load_json(history_fn_name(model))

