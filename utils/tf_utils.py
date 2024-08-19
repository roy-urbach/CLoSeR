import numpy as np
import tensorflow as tf
import os
import random

from utils.io_utils import load_json
from utils.modules import Modules
from utils.utils import printd

CUSTOM_OBJECTS = {}


def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)


def get_custom_objects():
    return CUSTOM_OBJECTS# keras.saving.get_custom_objects()


def serialize(c, package=''):
    get_custom_objects()[package + '>' * bool(package) + c.__name__] = c
    # print(f"Added {c.__name__} to custom layers")
    return c


def save_model(model, module:Modules):
    fn = get_model_fn(model, module)
    model.save(fn)


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


def load_model(fn):
    reconstructed_model = tf.keras.models.load_model(fn, custom_objects=get_custom_objects())
    # with keras.saving.custom_object_scope(get_custom_objects()):
    #     reconstructed_model = tf.keras.models.load_model(fn)
    return reconstructed_model


@Modules.add_method
def load_checkpoint(model):
    checkpoint_name = os.path.join(model.name, 'weights.ckpt')
    if os.path.exists(checkpoint_name):
        printd("checkpoint found. loading...", end='\t')
        model.load(checkpoint_name)
        printd("done!")
    else:
        printd("no checkpoint found")


@Modules.add_method
def history_fn_name(model):
    return os.path.join(model, "checkpoints/history.json")


@Modules.add_method
def load_history(model):
    return load_json(history_fn_name(model))

