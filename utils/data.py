import numpy as np
import abc


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, x_val=None, y_val=None, val_split=None, normalize=False,
                 img_normalize=False, flatten_y=False, split=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split

        if val_split and self.x_val is None and split:
            print("splitting randomly")
            perm = np.random.permutation(len(self.x_train))
            split_idx = int(len(self.x_train) * self.val_split)
            val_idx = perm[:split_idx]
            train_idx = perm[split_idx:]
            self.x_val = self.x_train[val_idx]
            self.y_val = {k: v[val_idx] for k, v in self.y_train.items()} if isinstance(self.y_train, dict) else self.y_train[val_idx]

            self.x_train = self.x_train[train_idx]
            self.y_train = {k: v[train_idx] for k, v in self.y_train.items()} if isinstance(self.y_train, dict) else self.y_train[train_idx]

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


class Label:
    def __init__(self, name, kind, dimension, meaning=None):
        self.name = name
        self.kind = kind
        self.dimension = dimension
        self.meaning = meaning


class ComplicatedData:
    __metaclass__ = abc.ABCMeta

    def __init__(self, train=True, val=False, test=False):
        assert train or val or test
        self.train = train
        self.train_ds = self if train else None
        self.test = test
        self.test_ds = self if test else None
        self.val = val
        self.val_ds = self if val else None
        self.x = None
        self.y = None

        self.name_to_label = {}

    def get_x(self, *args, **kwargs):
        if self.x is None:
            self._set_x(*args, **kwargs)
        return self.x

    def get_y(self, *args, labels=None, **kwargs):
        if self.y is None:
            self._set_y(*args, **kwargs)
        if labels is None and self.name_to_label is None:
            return self.y
        else:
            actual_y = {}
            for name, label in self.name_to_label.items() if labels is None else {(label.value if hasattr(label, 'value') else label).name: label
                                                                                  for label in labels}.items():
                actual_y[name] = np.array(self.y[(label.value if hasattr(label, 'value') else label).name] if label.value.name else np.zeros(self.__len__()))
            return actual_y

    @abc.abstractmethod
    def _set_x(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set_y(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _things_to_inherit(self):
        raise NotImplementedError()

    def clone(self, **kwargs):
        self_kwargs = self._things_to_inherit()
        self_kwargs.update(**kwargs)
        clone = self.__class__(**self_kwargs)
        clone.name_to_label = {k: v for k, v in self.name_to_label.items()}
        return clone

    def get_train(self):
        if self.train_ds is None:
            self.train_ds = self.clone(train=True, val=False, test=False)
        return self.train_ds

    def get_validation(self):
        if self.val_ds is None:
            self.val_ds = self.clone(train=False, val=True, test=False)
        return self.val_ds

    def get_test(self):
        if self.test_ds is None:
            self.test_ds = self.clone(train=False, val=False, test=True)
        return self.test_ds

    def get_shape(self):
        return self.get_x().shape[1:]

    def __len__(self):
        return self.get_x().shape[0]

    def get_x_train(self):
        return self.get_train().get_x()

    def get_y_train(self, *args, **kwargs):
        return self.get_train().get_y(*args, **kwargs)

    def get_x_val(self):
        return self.get_validation().get_x()

    def get_y_val(self, *args, **kwargs):
        return self.get_validation().get_y(*args, **kwargs)

    def get_x_test(self):
        return self.get_test().get_x()

    def get_y_test(self, *args, **kwargs):
        return self.get_test().get_y(*args, **kwargs)

    def update_name_to_label(self, name, label):
        self.name_to_label[name] = label
