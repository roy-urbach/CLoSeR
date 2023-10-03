import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import random


def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)
    tf.keras.utils.set_random_seed(s)


def get_custom_objects():
    return keras.saving.get_custom_objects()


def serialize(c, package=''):
    get_custom_objects()[package + '>' * bool(package) + c.__name__] = c
    print(f"Added {c.__name__} to custom layers")
    return c


def save_model(model, name=''):
    from datetime import date
    model_name = 'models/' + (model.name if not name else name) + str(date.today()) + '.h5'
    model.save(model_name)


def load_model(fn):
    with keras.saving.custom_object_scope(get_custom_objects()):
        reconstructed_model = tf.keras.models.load_model(fn)
    return reconstructed_model


def load_model_from_json(model_name):
    return load_model(f'models/{model_name}.h5')
