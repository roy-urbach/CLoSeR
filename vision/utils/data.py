import tensorflow as tf


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, val_split=0.1, normalize=False, img_normalize=False, flatten_y=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.val_split = val_split
        self.shape = x_train[0].shape
        if img_normalize:
            self.image_normalize_data()
        if normalize:
            self.normalize_data()
        if flatten_y:
            self.y_train = self.y_train.flatten()
            self.y_test = self.y_test.flatten()

    def image_normalize_data(self):
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

    def normalize_data(self):
        mean = self.x_train.mean()
        std = self.x_train.std(ddof=1)
        self.x_train = (self.x_train - mean) / std
        self.x_test = (self.x_test - mean) / std

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
    LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, **kwargs)


class MNIST(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        super().__init__(x_train[..., None], y_train, x_test[..., None],  y_test, *args, **kwargs)


class FMNIST(Data):
    def __init__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        super().__init__(x_train[..., None], y_train, x_test[..., None],  y_test, *args, **kwargs)
