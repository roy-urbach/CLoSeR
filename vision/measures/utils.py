from model.model import load_model_from_json
from utils.io_utils import load_json, save_json, get_file_time
from utils.utils import get_class, counter
import numpy as np
from enum import Enum, auto

MEASURES_FILE_NAME = 'measures'


class CrossPathMeasures(Enum):
    Acc = auto()
    LikeSelf = auto()
    Entropy = auto()
    MaxLikeNoSelf = auto()
    EntropyNoSelf = auto()
    Distance = auto()
    Correlation = auto()
    DKL = auto()


def load_measures_json(model_name, tolist=False):
    dct_with_list = load_json(MEASURES_FILE_NAME, base_path=f'models/{model_name}')
    if not tolist and dct_with_list is not None:
        dct = {k: np.array(v) for k, v in dct_with_list.items()}
    else:
        dct = dct_with_list
    return dct


def save_measures_json(model_name, dct):
    dct = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in dct.items()}
    save_json(MEASURES_FILE_NAME, dct, base_path=f'models/{model_name}')


def get_measuring_time(model_name, raw=True):
    return get_file_time(f'models/{model_name}/{MEASURES_FILE_NAME}.json', raw=raw)


def measure_model(model, iterations=50, b=128):
    import tensorflow as tf

    if isinstance(model, str):
        model = load_model_from_json(model)

    import utils.data
    kwargs = load_json(model.name)
    dataset = get_class(kwargs.get('dataset', 'Cifar10'), utils.data)(**kwargs.get("data_kwargs", {}))
    test_embd = model.predict(dataset.get_x_test())[0]

    n = test_embd.shape[2]

    loss = model.loss[model.name + "_embedding"]

    res_dct = {k: [] for k in CrossPathMeasures}

    triu_inds = np.triu_indices(b, 1)

    for i in counter(range(iterations)):
        cur_samples = test_embd[np.random.permutation(test_embd.shape[0])[:b]]
        cur_dists = np.sqrt(loss.calculate_dists(cur_samples).numpy())
        logits = loss.calculate_logits(None, dist=cur_dists)
        mrdev = loss.map_rep_dev(exp_logits=None, logits=tf.linalg.diag_part(logits))   # (b, n)

        likelihood = model.loss[model.name + "_embedding"].calculate_likelihood(logits=logits)
        res_dct[CrossPathMeasures.Acc].append(tf.reduce_mean(tf.cast((likelihood >= tf.reduce_max(likelihood, axis=0, keepdims=True))[
                                                     tf.eye(tf.shape(likelihood)[0], dtype=tf.bool)], dtype=tf.float32),
                                         axis=0))
        res_dct[CrossPathMeasures.LikeSelf].append(tf.reduce_mean(likelihood[tf.eye(tf.shape(likelihood)[0], dtype=tf.bool)], axis=0))
        res_dct[CrossPathMeasures.Entropy].append(-tf.reduce_mean(tf.einsum('bBnN,bBnN->BnN', likelihood, tf.math.log(likelihood)), axis=0))

        likelihood_without_self = tf.reshape(
            likelihood[tf.tile(~tf.eye(b, dtype=tf.bool)[..., None, None], [1, 1, n, n])],
            (b - 1, b, n, n))
        likelihood_without_self = likelihood_without_self / tf.reduce_sum(likelihood_without_self, axis=0,
                                                                          keepdims=True)

        res_dct[CrossPathMeasures.MaxLikeNoSelf].append(tf.reduce_mean(tf.reduce_max(likelihood, axis=0), axis=0))
        res_dct[CrossPathMeasures.EntropyNoSelf].append(
            -tf.reduce_mean(tf.einsum('bBnN,bBnN->BnN', likelihood_without_self, tf.math.log(likelihood_without_self)),
                            axis=0))

        res_dct[CrossPathMeasures.Distance].append(cur_dists[np.arange(b), np.arange(b)].mean(axis=0))
        res_dct[CrossPathMeasures.Correlation].append(np.corrcoef(cur_dists[triu_inds][..., np.arange(n), np.arange(n)].T))
        res_dct[CrossPathMeasures.DKL].append(tf.reduce_mean(mrdev, axis=0))

    res_dct = {k.name: np.mean(v, axis=0) for k, v in res_dct.items()}
    return res_dct
