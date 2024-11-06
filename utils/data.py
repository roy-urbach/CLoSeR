import numpy as np

class Data:
    def __init__(self, x_train, y_train, x_test, y_test, x_val=None, y_val=None, val_split=None, normalize=False,
                 img_normalize=False, flatten_y=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split
        self.shape = x_train[0].shape
        if img_normalize:
            self.image_normalize_data()
        if normalize:
            self.normalize_data()
        if flatten_y:
            self.y_train = {k: v.flatten() for k, v in self.y_train.items()} if isinstance(self.y_train, dict) else self.y_train.flatten()
            self.y_test = {k: v.flatten() for k, v in self.y_test.items()} if isinstance(self.y_test, dict) else self.y_test.flatten()
            if self.y_val is not None:
                self.y_val = {k: v.flatten() for k, v in self.y_val.items()} if isinstance(self.y_val, dict) else self.y_val.flatten()

    @staticmethod
    def is_generator():
        return False

    def image_normalize_data(self):
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

    def normalize_data(self):
        mean = self.x_train.mean(axis=0, keepdims=True)
        std = self.x_train.std(ddof=1, axis=0, keepdims=True)
        self.x_train = np.true_divide(self.x_train - mean, std, where=(std > 0) & ~np.isnan(std), out=np.zeros(self.x_train.shape, dtype=std.dtype))
        self.x_test = np.true_divide(self.x_test - mean, std, where=(std > 0) & ~np.isnan(std), out=np.zeros(self.x_test.shape, dtype=std.dtype))
        if self.x_val is not None:
            self.x_val = np.true_divide(self.x_val - mean, std, where=(std > 0) & ~np.isnan(std),
                                        out=np.zeros(self.x_val.shape, dtype=std.dtype))

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

    def get_x_val(self):
        return self.x_val

    def get_y_val(self):
        return self.y_val

    def get_val_split(self):
        return self.val_split

    def get_shape(self):
        return self.shape


def gen_to_tf_dataset(gen, batch_size, buffer_size):
    import tensorflow as tf

    def generator_func():
        while True:
            yield gen[0]

    example = gen[0]

    dataset = tf.data.Dataset.from_generator(
        generator_func,
        output_types=(example[0].dtype, {k: v.dtype for k, v in example[1].items()}),
        output_shapes=(example[0].shape, {k: v.shape for k, v in example[1].items()})
    )
    dataset = dataset.map(lambda x, y: (tf.convert_to_tensor(x), {k: tf.convert_to_tensor(v) for k, v in y.items()}))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size)
    dataset.get_validation = lambda *args, **kwargs: gen_to_tf_dataset(gen.get_validation(), batch_size, buffer_size)
    dataset.get_train = lambda *args, **kwargs: gen_to_tf_dataset(gen.get_train(), batch_size, buffer_size)
    dataset.get_test = lambda *args, **kwargs: gen_to_tf_dataset(gen.get_test(), batch_size, buffer_size)
    for attr in dir(gen):
        if attr not in dir(dataset):
            setattr(dataset, attr, getattr(gen, attr))
    dataset.__len__ = gen.__len__
    return dataset
