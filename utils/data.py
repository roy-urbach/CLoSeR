import numpy as np
import abc

from utils.modules import Modules

CATEGORICAL = "categorical"
CONTINUOUS = 'continuous'


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, x_val=None, y_val=None, val_split=None, normalize=False,
                 img_normalize=False, flatten_y=False, split=False, simple_norm=False, module:Modules=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split
        self.module = module

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
            self.normalize_data(simple=simple_norm)
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

    def normalize_data(self, simple=False):
        mean = self.x_train.mean(axis=None if simple else 0, keepdims=True)
        std = self.x_train.std(ddof=1, axis=None if simple else 0, keepdims=True)
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

    def is_categorical(self):
        return self.kind == CATEGORICAL


class ComplicatedData:
    __metaclass__ = abc.ABCMeta

    def __init__(self, module: Modules, train=True, val=False, test=False, train_ds=None, test_ds=None, val_ds=None):
        assert train or val or test
        self.module = module
        self.train = train
        self.train_ds = self if train else train_ds
        self.test = test
        self.test_ds = self if test else test_ds
        self.val = val
        self.val_ds = self if val else val_ds
        self.x = None
        self.y = None

        self.name_to_label = {}
        self.label_to_dim = None

    def get_x(self, *args, **kwargs):
        if self.x is None:
            self._set_x(*args, **kwargs)
        return self.x

    def get_y(self, *args, labels=None, **kwargs):
        if self.y is None:
            self._set_x(*args, **kwargs)
        if labels is None and self.name_to_label is None:
            return self.y
        elif isinstance(labels, str):
            return np.array(self.y[labels])
        elif isinstance(labels, Label):
            return np.array(self.y[labels.name])
        else:
            actual_y = {}
            for name, label in self.name_to_label.items() if labels is None else {(label.value if hasattr(label, 'value') else label).name: label
                                                                                      for label in labels}.items():
                    actual_y[name] = np.array(self.y[(label.value if hasattr(label, 'value') else label).name] if label.value.name else np.zeros(len(self)))
            return actual_y

    @abc.abstractmethod
    def _set_x(self, *args, **kwargs):
        raise NotImplementedError()

    def _set_label_to_dim(self, *args, **kwargs):
        if self.y is None:
            self._set_x(*args, **kwargs)
        self.label_to_dim = {self.module.get_label(name).value.name: self.module.get_label(name).value.dimension for name in self.y}

    def get_label_to_dim(self):
        if self.label_to_dim is None:
            self._set_label_to_dim()
        return self.label_to_dim

    def get_config(self):
        return dict(train=self.train, val=self.val, test=self.test,
                    train_ds=self.train_ds, val_ds=self.val_ds, test_ds=self.test_ds)

    def clone(self, **kwargs):
        self_kwargs = self.get_config()
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


class TemporalData(ComplicatedData):
    def __init__(self, module: Modules, samples_per_example=2, single_time_label=True, x_samples=None, y_samples=None, **kwargs):
        super().__init__(module=module, **kwargs)
        self.samples_per_example = samples_per_example
        self.single_time_label = single_time_label
        self.x_samples = x_samples
        self.y_samples = y_samples
        if self.x_samples is None:
            self._load_data()   # (assume x_steps is (B, DIM), and y_steps is {label: y.label} or (B, DIM) or (B, ))

    def _load_data(self):
        if self.x_samples is None:
            raise NotImplementedError()

    def get_config(self):
        return dict(**super().get_config(),
                    samples_per_example=self.samples_per_example,
                    single_time_label=self.single_time_label)

    def _set_x(self):
        val_start = int(len(self.x_samples) * 0.6)
        test_start = int(len(self.x_samples) * 0.8)
        trans = lambda arr: np.transpose(arr, [0, 2, 1])

        if self.train:
            inds = np.arange(self.samples_per_example)[None] + np.arange(val_start - self.samples_per_example)[:, None]
        elif self.val:
            inds = val_start + np.arange(self.samples_per_example)[None] + np.arange(
                test_start - val_start - self.samples_per_example)[:, None]
        else:
            inds = test_start + np.arange(self.samples_per_example)[None] + np.arange(
                len(self.x_samples) - test_start - self.samples_per_example)[:, None]

        inds_y = inds[:, -1] if self.single_time_label else inds

        self.x = trans(self.x_samples[inds])
        self.y = {k: val[inds_y] for k, val in self.y_samples.items()} if isinstance(self.y_samples, dict) else self.y_samples[inds_y]


class GeneratorDataset:
    def __init__(self, module: Modules, train=True, val=False, test=False, batch_size=32, name_to_label=None):
        self.module = module
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size

        assert (train + val + test) == 1
        self.train = train
        self.val = val
        self.test = test
        self.name_to_label = {} if name_to_label is None else name_to_label
        self.label_to_dim = None
        self._set()
        self._set_label_to_dim()

    def get_label_to_dim(self):
        return self.label_to_dim

    @abc.abstractmethod
    def _set_label_to_dim(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_shape(self):
        raise NotImplementedError()

    def get_config(self):
        return dict(train=self.train, val=self.val, test=self.test,
                    module=self.module, batch_size=self.batch_size,
                    name_to_label={k: v for k, v in self.name_to_label.items()})

    def get_train(self):
        return self.__class__(**self.get_config(), train=True, val=False, test=False)

    def get_val(self):
        return self.__class__(**self.get_config(), train=False, val=True, test=False)

    def get_test(self):
        return self.__class__(**self.get_config(), train=False, val=False, test=True)

    def update_name_to_label(self, name, label):
        self.name_to_label[name] = label
