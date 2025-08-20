import numpy as np
import abc

from utils.modules import Modules
from utils.utils import run_on_dict

CATEGORICAL = "categorical"
CONTINUOUS = 'continuous'


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, x_val=None, y_val=None, val_split=None, normalize=False,
                 img_normalize=False, flatten_y=False, split=False, simple_norm=False, module:Modules=None, seed=1929,
                 test_only=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.test_only = test_only
        self.val_split = val_split
        self.module = module

        if not test_only and (val_split and self.x_val is None and split):
            print("splitting randomly")
            np.random.seed(seed)
            perm = np.random.permutation(len(self.x_train if not isinstance(self.x_train, dict) else list(self.x_train.value())[0]))
            split_idx = int(len(self.x_train) * self.val_split)
            val_idx = perm[:split_idx]
            train_idx = perm[split_idx:]
            self.x_val = run_on_dict(self.x_train, lambda arr: arr[val_idx])
            self.y_val = run_on_dict(self.y_train, lambda arr: arr[val_idx])

            self.x_train = run_on_dict(self.x_train, lambda arr: arr[train_idx])
            self.y_train = run_on_dict(self.y_train, lambda arr: arr[train_idx])

        self.shape = run_on_dict(x_test, lambda x: x[0].shape)
        if img_normalize:
            self.image_normalize_data()
        if normalize:
            self.normalize_data(simple=simple_norm)
        if flatten_y:
            self.y_test = run_on_dict(self.y_test, lambda y: y.flatten())
            if not test_only:
                self.y_train = run_on_dict(self.y_train, lambda y: y.flatten())
                if self.y_val is not None:
                    self.y_val = run_on_dict(self.y_val, lambda y: y.flatten())

    @staticmethod
    def is_generator():
        return False

    def image_normalize_data(self):
        self.x_test = run_on_dict(self.x_test, lambda a: a / 255)
        if not self.test_only:
            self.x_train = run_on_dict(self.x_train, lambda a: a / 255)
            if self.x_val is not None:
                self.x_val = run_on_dict(self.x_val, lambda a: a / 255)

    def normalize_data(self, simple=False):
        def normalize(arr_train, arr_test, arr_val=None):
            m = arr_train.mean(None if simple else 0, keepdims=True)
            std = arr_train.std(ddof=1, axis=None if simple else 0, keepdims=True)
            mask = (std > 0) & ~np.isnan(std)
            def to_z(arr):
                return np.true_divide(arr - m, std, where=mask, out=np.zeros(arr.shape, dtype=std.dtype))
            if arr_val is None:
                return to_z(arr_train), to_z(arr_test)
            else:
                return to_z(arr_train), to_z(arr_test), to_z(arr_val)

        if isinstance(self.x_train, dict):
            if self.x_val is None:
                new_x_train, new_x_test = {}, {}
                for k in self.x_train:
                    new_x_train[k], new_x_test[k] = normalize(self.x_train[k], self.x_test[k])
                self.x_train = new_x_train
                self.x_test = new_x_test
            else:
                new_x_train, new_x_test, new_x_val = {}, {}, {}
                for k in self.x_train:
                    new_x_train[k], new_x_test[k], new_x_val[k] = normalize(self.x_train[k], self.x_test[k], self.x_val[k])
                self.x_train = new_x_train
                self.x_test = new_x_test
                self.x_val = new_x_val

        else:

            if self.x_val is None:
                self.x_train, self.x_test = normalize(self.x_train, self.x_test)
            else:
                self.x_train, self.x_test, self.x_val = normalize(self.x_train, self.x_test, self.x_val)

    def get_all(self):
        return self.get_train(), self.get_test()

    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_x_train(self):
        return self.x_train

    def _get_y_labels(self, y, labels=None):
        if labels is None:
            return y
        else:
            assert isinstance(y, dict)
            if isinstance(labels, str):
                return y[labels]
            else:
                return {label: y[label] for label in labels}

    def get_y_train(self, labels=None):
        return self._get_y_labels(self.y_train, labels=labels)

    def get_x_test(self):
        return self.x_test

    def get_y_test(self, labels=None):
        return self._get_y_labels(self.y_test, labels=labels)

    def get_x_val(self):
        return self.x_val

    def get_y_val(self, labels=None):
        return self._get_y_labels(self.y_val, labels=labels)

    def get_val_split(self):
        return self.val_split

    def get_shape(self):
        return self.shape


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
        """
        This has to be implemented by any ComplicatedData
        """
        raise NotImplementedError()

    def _set_label_to_dim(self, *args, **kwargs):
        if self.y is None:
            self._set_x(*args, **kwargs)
        self.label_to_dim = {self.module.get_label(name).value.name: self.module.get_label(name).value.dimension
                             for name in self.y}

    def get_label_to_dim(self):
        if self.label_to_dim is None:
            self._set_label_to_dim()
        return self.label_to_dim

    def get_config(self):
        return dict(module=self.module, train=self.train, val=self.val, test=self.test,
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
