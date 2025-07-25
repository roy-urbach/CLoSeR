import os
from enum import Enum

import tensorflow as tf
import numpy as np
from utils.data import Data, CATEGORICAL, Label
from utils.modules import Modules


class Cifar10(Data):
    LABELS = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

    def __init__(self, *args, val_split=0.1, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, val_split=val_split, flatten_y=True, **kwargs)


class Cifar100(Data):
    """
    Not used in the paper, but preliminary results shows promising generalization
    """
    def __init__(self, *args, val_split=0.1, **kwargs):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        super().__init__(x_train, y_train, x_test,  y_test, *args, val_split=val_split, flatten_y=True, **kwargs)


class Labels(Enum):
    CLASS = Label("class", CATEGORICAL, len(Cifar10.LABELS))

    @staticmethod
    def get(name):
        if isinstance(name, Labels):
            return name
        else:
            relevant_labels = [label for label in Labels if name in (label.name, label.value, label.value.name)]
            if not len(relevant_labels):
                raise ValueError(f"No label named {name}")
            else:
                return relevant_labels[0]


class TinyImageNet200(Data):
    DIRECTORY = os.path.join(Modules.VISION.value, "tiny_imagenet", "tiny-imagenet-200")
    NUM_CLASSES = 200

    def __init__(self, *args, val_split=0.1, test_only=False, **kwargs):
        (x_train, y_train), (x_test, y_test) = self.load(test_only)
        super().__init__(x_train, y_train, x_test, y_test, *args, test_only=test_only,
                         val_split=val_split, flatten_y=True, **kwargs)

    @staticmethod
    def load(test_only=False):
        import numpy as np
        from PIL import Image
        from tqdm import tqdm

        image_size = (64, 64)

        # Get class ids from wnids.txt
        with open(os.path.join(TinyImageNet200.DIRECTORY, 'wnids.txt')) as f:
            wnids = [line.strip() for line in f.readlines()]
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # --- Load training data ---
        if not test_only:
            X_train, y_train = [], []
            train_dir = os.path.join(TinyImageNet200.DIRECTORY, 'train')
            for wnid in tqdm(wnids, desc="Loading train"):
                class_dir = os.path.join(train_dir, wnid, 'images')
                for filename in os.listdir(class_dir):
                    if filename.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, filename)
                        img = Image.open(img_path).convert("RGB").resize(image_size)
                        X_train.append(np.array(img))
                        y_train.append(wnid_to_label[wnid])
        else:
            X_train, y_train = None, None

        # --- Load validation data ---
        val_annotations_file = os.path.join(TinyImageNet200.DIRECTORY, 'val', 'val_annotations.txt')
        val_img_dir = os.path.join(TinyImageNet200.DIRECTORY, 'val', 'images')

        val_img_to_label = {}
        with open(val_annotations_file) as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                val_img_to_label[parts[0]] = wnid_to_label[parts[1]]

        X_val, y_val = [], []
        for filename in tqdm(os.listdir(val_img_dir), desc="Loading val"):
            if filename.endswith('.JPEG'):
                img_path = os.path.join(val_img_dir, filename)
                img = Image.open(img_path).convert("RGB").resize(image_size)
                X_val.append(np.array(img))
                y_val.append(val_img_to_label[filename])

        # Convert to arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        return (X_train, y_train), (X_val, y_val)

