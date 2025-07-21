from utils.data import GeneratorDataset
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.io_utils import load_json, save_json, get_file_time
from utils.utils import get_class, counter
import numpy as np
from enum import Enum, auto
import os

MEASURES_FILE_NAME = 'measures'


class CrossPathMeasures(Enum):
    # different measures about similarities between encoders
    Acc = "Agreement score"
    LikeSelf = auto()
    Entropy = auto()
    MaxLikeNoSelf = auto()
    EntropyNoSelf = auto()
    Distance = auto()
    Correlation = "Alignment of embedding maps"


@Modules.add_method
def load_measures_json(model_name, tolist=False):
    dct_with_list = load_json(os.path.join(model_name, MEASURES_FILE_NAME))
    if not tolist and dct_with_list is not None:
        dct = {k: np.array(v) for k, v in dct_with_list.items()}
    else:
        dct = dct_with_list
    return dct


@Modules.add_method
def save_measures_json(model_name, dct):
    dct = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in dct.items()}
    save_json(os.path.join(model_name, MEASURES_FILE_NAME), dct)


@Modules.add_method
def get_measuring_time(model_name, raw=True):
    return get_file_time(os.path.join(model_name, MEASURES_FILE_NAME) + '.json', raw=raw)


def measure_model(model, module:Modules, iterations=50, b=128):
    """
    Calculates a mean measure between pairs of encoders, over many samples of batches
    :param model: the model
    :param module: the module
    :param iterations: number of iterations to average over
    :param b: number of samples in batch
    :return: a dictionary where the keys correspond to CrossPathMeasures and the values are the measures
    """
    import tensorflow as tf

    if isinstance(model, str):
        model = load_model_from_json(model, module)
    kwargs = module.load_json(model.name, config=True)
    dataset = module.get_class_from_data(kwargs.get('dataset', 'Cifar10'))(module=module, **kwargs.get("data_kwargs", {}))
    bins_per_frame = dataset.bins_per_frame if hasattr(dataset, 'bins_per_frame') else (
        dataset.bins_per_sample if hasattr(dataset, 'bins_per_sample') else None)
    if issubclass(dataset.__class__, GeneratorDataset):
        dataset = dataset.to_regular_dataset()
    test_embd = model.predict(dataset.get_x_test())[0]

    if module in (Modules.NEURONAL, ):
        encoder_removed_bins = model.layers[0].output_shape[-1] not in (model.get_layer("embedding").output_shape[1],
                                                                        model.get_layer("embedding").output_shape[1] // 2)

        if encoder_removed_bins:
            last_step_embedding = test_embd[:, -1]
        else:
            last_step_embedding = test_embd[:, -bins_per_frame:]  # (B, bins_per_frame, DIM, P)
            last_step_embedding = last_step_embedding.reshape(last_step_embedding.shape[0],
                                                              last_step_embedding.shape[-2] * bins_per_frame,
                                                              last_step_embedding.shape[-1])  # (B, DIMS*bins_per_frame, P)
        test_embd = last_step_embedding

    n = test_embd.shape[2]

    from vision.model.losses import CLoSeRLoss
    loss_obj = CLoSeRLoss()

    res_dct = {k: [] for k in CrossPathMeasures}

    triu_inds = np.triu_indices(b, 1)

    for _ in counter(range(iterations)):
        cur_samples = test_embd[np.random.permutation(test_embd.shape[0])[:b]]
        cur_dists = np.sqrt(loss_obj.calculate_dists(cur_samples).numpy())
        logits = loss_obj.calculate_logits(cur_dists**2)
        likelihood = loss_obj.calculate_conditional_pseudo_likelihood(loss_obj.calculate_exp_logits(logits))
        res_dct[CrossPathMeasures.Acc].append(tf.reduce_mean(tf.cast((likelihood >= tf.reduce_max(likelihood, axis=0, keepdims=True))[
                                                     tf.eye(tf.shape(likelihood)[0], dtype=tf.bool)], dtype=tf.float32),
                                         axis=0))
        res_dct[CrossPathMeasures.LikeSelf].append(tf.reduce_mean(likelihood[tf.eye(tf.shape(likelihood)[0], dtype=tf.bool)], axis=0))
        res_dct[CrossPathMeasures.Entropy].append(-tf.reduce_mean(tf.einsum('bBnN,bBnN->BnN', likelihood, tf.math.log(likelihood)), axis=0))

        likelihood_without_self = tf.transpose(tf.reshape(
            likelihood[tf.tile(~tf.eye(b, dtype=tf.bool)[..., None, None], [1, 1, n, n])],
            (b, b-1, n, n)), [1,0,2,3])
        likelihood_without_self = likelihood_without_self / tf.reduce_sum(likelihood_without_self, axis=0,
                                                                          keepdims=True)

        res_dct[CrossPathMeasures.MaxLikeNoSelf].append(tf.reduce_mean(tf.reduce_max(likelihood, axis=0), axis=0))
        res_dct[CrossPathMeasures.EntropyNoSelf].append(
            -tf.reduce_mean(tf.einsum('bBnN,bBnN->BnN', likelihood_without_self, tf.math.log(likelihood_without_self)),
                            axis=0))

        res_dct[CrossPathMeasures.Distance].append(cur_dists[np.arange(b), np.arange(b)].mean(axis=0))
        res_dct[CrossPathMeasures.Correlation].append(np.corrcoef(cur_dists[triu_inds][..., np.arange(n), np.arange(n)].T))

    res_dct = {k.name: np.mean(v, axis=0) for k, v in res_dct.items()}
    return res_dct
