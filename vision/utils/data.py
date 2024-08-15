import tensorflow as tf

from utils.data import Data


class Cifar10(Data):
    LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, **kwargs)


class Cifar100(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, **kwargs)


class MNIST(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        super().__init__(x_train[..., None], y_train, x_test[..., None],  y_test, *args, **kwargs)


class FMNIST(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        super().__init__(x_train[..., None], y_train, x_test[..., None],  y_test, *args, **kwargs)
