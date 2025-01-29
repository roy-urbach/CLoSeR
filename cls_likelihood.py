from utils.model.model import load_model_from_json
import numpy as np
import os

from vision.utils.data import Cifar10
from utils.utils import flatten_but_batch, unknown_args_to_dict, printd


def calculate_class_mean_likelihood(model, module, pred=None, save=False, examples_per_class=50, repeats=100, temp=10, **kwargs):
    ds = Cifar10()
    C = len(ds.LABELS)

    from tqdm import tqdm as counter
    path_to_save = f"{module.value}/models/{model.name}/cls_mean_likelihood.npy"
    if os.path.exists(path_to_save):
        with open(path_to_save, 'rb') as f:
            mean_dists = np.load(path_to_save)
    else:
        pred_x = model.predict(ds.get_x_test())[0] if pred is None else pred
        inds_by_class = [np.where(ds.get_y_test() == i)[0] for i in range(C)]
        mean_cls_cls_likelihood = []
        for _ in counter(range(repeats)):
            mean_cls_cls_likelihood_p = []
            cur_examples_inds = np.concatenate([inds[np.random.permutation(len(inds))[:examples_per_class]] for inds in inds_by_class])
            cur_examples = pred_x[cur_examples_inds]    # (B, DIM, P)

            for p in range(pred_x.shape[-1]):
                sim = np.exp(-(np.linalg.norm(cur_examples[:, None, ..., p] - cur_examples[None,..., p], axis=-1)**2)/temp)
                likelihood = sim / sim.sum(axis=1)  # (B, B)
                mean_example_cls_likelihood = likelihood.reshape(len(likelihood), C, examples_per_class).mean(axis=-1)   # (B, C)
                mean_cls_cls_likelihood_p.append(mean_example_cls_likelihood.reshape(C, examples_per_class, C).mean(axis=1))
            mean_cls_cls_likelihood.append(np.stack(mean_cls_cls_likelihood_p, axis=-1))
        mean_cls_cls_likelihood = np.mean(mean_cls_cls_likelihood, axis=0)
        if save:
            with open(path_to_save, 'wb') as f:
                np.save(f, mean_cls_cls_likelihood)
    return mean_dists


def calculate_class_mean_likelihood_ens(model, module, pred=None, save=False, examples_per_class=50, repeats=100, temp=10, **kwargs):
    ds = Cifar10()
    C = len(ds.LABELS)

    from tqdm import tqdm as counter
    path_to_save = f"{module.value}/models/{model.name}/cls_mean_likelihood_ens.npy"
    if os.path.exists(path_to_save):
        with open(path_to_save, 'rb') as f:
            mean_dists = np.load(path_to_save)
    else:
        pred_x = model.predict(ds.get_x_test())[0] if pred is None else pred
        inds_by_class = [np.where(ds.get_y_test() == i)[0] for i in range(C)]
        mean_cls_cls_likelihood = []
        for _ in counter(range(repeats)):
            cur_examples_inds = np.concatenate(
                [inds[np.random.permutation(len(inds))[:examples_per_class]] for inds in inds_by_class])
            cur_examples = flatten_but_batch(pred_x[cur_examples_inds])  # (B, DIM*P)

            sim = np.exp(-(np.linalg.norm(cur_examples[:, None] - cur_examples[None], axis=-1) ** 2) / temp)
            likelihood = sim / sim.sum(axis=1)  # (B, B)
            mean_example_cls_likelihood = likelihood.reshape(len(likelihood), C, examples_per_class).mean(axis=-2)  # (B, C)
            mean_cls_cls_likelihood.append(mean_example_cls_likelihood.reshape(C, examples_per_class, C).mean(axis=1))
        mean_cls_cls_likelihood = np.mean(mean_cls_cls_likelihood, axis=0)
        if save:
            with open(path_to_save, 'wb') as f:
                np.save(f, mean_cls_cls_likelihood)
    return mean_dists


import argparse


def parse():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')
    parser.add_argument('-r', '--repeats', type=int, default=250, help='number of repeats')
    parser.add_argument('-e', '--examples', type=int, default=50, help='examples per label')
    parser.add_argument('-t', '--temp', type=float, default=10., help='temperature')
    return parser.parse_known_args()


if __name__ == "__main__":
    from utils.modules import Modules
    from run_before_script import run
    run()

    args, unknown_args = parse()

    kwargs = unknown_args_to_dict(unknown_args, warning=True)

    model_name = args.json
    module = Modules.VISION
    printd(f"running {model_name}. Loading model...")
    model = load_model_from_json(model_name, module=module)
    printd("done")
    ds = Cifar10()
    printd("predicting test")
    pred = model.predict(ds.get_x_test())[0]
    printd("done")

    printd("running individual pathways...")
    calculate_class_mean_likelihood(model, module, pred=pred, save=True,
                                    repeats=args.repeats, examples_per_class=args.examples, temp=args.temp,
                                    **kwargs)
    printd("done")
    printd("running ensemble...")
    calculate_class_mean_likelihood_ens(model, module, pred=pred, save=True,
                                        repeats=args.repeats, examples_per_class=args.examples, temp=args.temp,
                                        **kwargs)
    printd("done")


