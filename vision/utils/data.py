import tensorflow as tf


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, val_split=0.1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.val_split = val_split
        self.shape = x_train[0].shape

    def get_all(self):
        return self.get_train(), self.get_test()

    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_val_split(self):
        return self.val_split

    def get_shape(self):
        return self.shape


class Cifar10(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, **kwargs)


class MNIST(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, **kwargs)
