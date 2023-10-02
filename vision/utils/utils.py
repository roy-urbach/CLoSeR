import numpy as np
import tensorflow as tf
from tensorflow import keras


def serialize(c, package=''):
    dct = keras.saving.get_custom_objects()
    dct[package + '>' * bool(package) + c.__name__] = c
    print(f"Added {c.__name__} to custom layers")
    return c
