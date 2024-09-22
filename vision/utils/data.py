import tensorflow as tf
import numpy as np
from utils.data import Data


class Cifar10(Data):
    LABELS = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

    def __init__(self, *args, val_split=0.1, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, val_split=val_split, **kwargs)


class Cifar100(Data):
    def __init__(self, *args, val_split=0.1, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, val_split=val_split, **kwargs)


class MNIST(Data):
    def __init__(self, *args, val_split=0.1, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        super().__init__(x_train[..., None], y_train, x_test[..., None],  y_test, val_split=val_split, *args, **kwargs)


class FMNIST(Data):
    def __init__(self, *args, val_split=0.1, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        super().__init__(x_train[..., None], y_train, x_test[..., None],  y_test, val_split=val_split, *args, **kwargs)
