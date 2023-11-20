import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import random
from utils.utils import printd

CUSTOM_OBJECTS = {}


def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)
    tf.keras.utils.set_random_seed(s)


def get_custom_objects():
    return CUSTOM_OBJECTS# keras.saving.get_custom_objects()


def serialize(c, package=''):
    get_custom_objects()[package + '>' * bool(package) + c.__name__] = c
    print(f"Added {c.__name__} to custom layers")
    return c


def save_model(model):
    fn = get_model_fn(model)
    model.save(fn)


def get_model_fn(model_or_name):
    if isinstance(model_or_name, str):
        model_name = model_or_name
    else:
        assert hasattr(model_or_name, 'name')
        model_name = model_or_name.name
    fn = os.path.join('models', model_name, 'model.h5')
    return fn


def get_weights_fn(model_or_name):
    if isinstance(model_or_name, str):
        model_name = model_or_name
    else:
        assert hasattr(model_or_name, 'name')
        model_name = model_or_name.name
    fn = os.path.join('models', model_name, "checkpoints", 'model_weights_{epoch:02d}')
    return fn


def load_model(fn):
    reconstructed_model = tf.keras.models.load_model(fn, custom_objects=get_custom_objects())
    # with keras.saving.custom_object_scope(get_custom_objects()):
    #     reconstructed_model = tf.keras.models.load_model(fn)
    return reconstructed_model


def load_checkpoint(model):
    checkpoint_name = f'/models/{model.name}/weights.ckpt'
    if os.path.exists(checkpoint_name):
        printd("checkpoint found. loading...", end='\t')
        model.load(checkpoint_name)
        printd("done!")
    else:
        printd("no checkpoint found")
