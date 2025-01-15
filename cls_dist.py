from utils.model.model import load_model_from_json
import numpy as np
import os

from vision.utils.data import Cifar10
from utils.utils import flatten_but_batch, unknown_args_to_dict


def calculate_class_mean_dists(model, module, pred=None, save=False, **kwargs):
    ds = Cifar10()

    from tqdm import tqdm as counter
    path_to_save = f"{module.value}/models/{model.name}/cls_mean_dists.npy"
    if os.path.exists(path_to_save):
        with open(path_to_save, 'rb') as f:
            mean_dists = np.load(path_to_save)
    else:
        pred_x = model.predict(ds.get_x_test())[0] if pred is None else pred
        mean_dists = np.full([len(ds.LABELS)] * 2 + [pred_x.shape[-1]], np.nan)
        for p in counter(range(pred_x.shape[-1])):
            for cls1 in range(mean_dists.shape[0]):
                where1 = ds.get_y_test() == cls1
                for cls2 in range(mean_dists.shape[1]):
                    where2 = ds.get_y_test() == cls2
                    dist = np.linalg.norm(pred_x[..., p][where1][:, None] - pred_x[..., p][where2][None], axis=-1)
                    if cls1 == cls2:
                        mean_dist = dist[np.triu_indices(dist.shape[0])].mean()
                    else:
                        mean_dist = dist.mean()
                    mean_dists[cls1, cls2, p] = mean_dist
        if save:
            with open(path_to_save, 'wb') as f:
                np.save(f, mean_dists)
    return mean_dists


def calculate_class_mean_dists_ens(model, module, pred=None, save=False, **kwargs):
    ds = Cifar10()

    from tqdm import tqdm as counter
    path_to_save = f"{module.value}/models/{model.name}/cls_mean_dists_ens.npy"
    if os.path.exists(path_to_save):
        with open(path_to_save, 'rb') as f:
            mean_dists = np.load(path_to_save)
    else:
        pred_x = model.predict(ds.get_x_test())[0] if pred is None else pred
        mean_dists = np.full([len(ds.LABELS)] * 2, np.nan)
        for cls1 in range(mean_dists.shape[0]):
            where1 = ds.get_y_test() == cls1
            for cls2 in range(mean_dists.shape[1]):
                where2 = ds.get_y_test() == cls2
                dist = np.linalg.norm(
                    flatten_but_batch(pred_x)[where1][:, None] - flatten_but_batch(pred_x)[where2][None], axis=-1)
                if cls1 == cls2:
                    mean_dist = dist[np.triu_indices(dist.shape[0])].mean()
                else:
                    mean_dist = dist.mean()
                mean_dists[cls1, cls2] = mean_dist
        if save:
            with open(path_to_save, 'wb') as f:
                np.save(f, mean_dists)
    return mean_dists


import argparse


def parse():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')
    return parser.parse_known_args()


if __name__ == "__main__":
    from utils.modules import Modules
    from run_before_script import run
    run()

    args, unknown_args = parse()

    kwargs = unknown_args_to_dict(unknown_args, warning=True)

    model_name = args.json
    module = Modules.VISION
    print(f"running {model_name}. Loading model...")
    model = load_model_from_json(model_name, module=module)
    print("done")
    ds = Cifar10()
    print("predicting test")
    pred = model.predict(ds.get_x_test())[0]
    print("done")

    print("running individual pathways...")
    calculate_class_mean_dists(model, module, pred=pred, save=True, **kwargs)
    print("done")
    print("running ensemble...")
    calculate_class_mean_dists_ens(model, module, pred=pred, save=True, **kwargs)
    print("done")


