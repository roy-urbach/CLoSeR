from utils.model.model import load_model_from_json
import numpy as np
import os
import argparse
from vision.utils.data import Cifar10
from utils.utils import unknown_args_to_dict, printd


def calculate_class_mean_likelihood(model, module, ds=Cifar10(), pred=None, save=False, mse=False,
                                    examples_per_class=50, repeats=100, temp=10, inp=False, **kwargs):
    """
    Calculate class mean conditional pseudo-likelihood
    """
    from tqdm import tqdm as counter
    path_to_save = f"{module.value}/models/{model.name}/cls_mean_likelihood{'_inp' if inp else ''}.npy"
    if os.path.exists(path_to_save):
        with open(path_to_save, 'rb') as f:
            mean_cls_cls_likelihood = np.load(path_to_save)
    else:
        pred_x = model.predict(ds.get_x_test())[0] if pred is None else pred         # embedding
        inds_by_class = [np.where(ds.get_y_test() == i)[0] for i in range(ds.NUM_CLASSES)]
        mean_cls_cls_likelihood = []
        for _ in counter(range(repeats)):   # calculate the average class-class conditional pseudo-likelihood over <repeats> image samples
            mean_cls_cls_likelihood_p = []
            cur_examples_inds = np.concatenate([inds[np.random.permutation(len(inds))[:examples_per_class]] for inds in inds_by_class])
            cur_examples = pred_x[cur_examples_inds]    # (B, DIM, P)

            for p in range(pred_x.shape[-1]):
                # psi
                reduce_f = np.mean if mse else np.sum
                sim = np.exp(-(reduce_f((cur_examples[:, None, ..., p] - cur_examples[None,..., p])**2, axis=-1)/temp))
                np.fill_diagonal(sim, np.nan)

                # conditional pseudo-likelihood
                likelihood = sim / np.nansum(sim, axis=1, keepdims=True)  # (B, B)
                mean_example_cls_likelihood = np.nanmean(likelihood.reshape(len(likelihood), ds.NUM_CLASSES, examples_per_class), axis=-1)   # (B, C)
                mean_cls_cls_likelihood_p.append(np.nanmean(mean_example_cls_likelihood.reshape(ds.NUM_CLASSES, examples_per_class, ds.NUM_CLASSES), axis=1))
            mean_cls_cls_likelihood.append(np.stack(mean_cls_cls_likelihood_p, axis=-1))
        mean_cls_cls_likelihood = np.mean(mean_cls_cls_likelihood, axis=0)
        if save:
            with open(path_to_save, 'wb') as f:
                np.save(f, mean_cls_cls_likelihood)
    return mean_cls_cls_likelihood




def parse():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')
    parser.add_argument('-r', '--repeats', type=int, default=250, help='number of repeats')
    parser.add_argument('-e', '--examples', type=int, default=50, help='examples per label')
    parser.add_argument('-t', '--temp', type=float, default=10., help='temperature')
    parser.add_argument('--mse', action=argparse.BooleanOptionalAction, default=False, help='use mse instead of Euclidian distance')
    parser.add_argument('--inp', action=argparse.BooleanOptionalAction, default=False, help='use input')
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
    if args.mse:
        printd("using mse")
    printd(f"using temp={args.temp}")
    model = load_model_from_json(model_name, module=module)
    dct = module.load_json(model_name, config=True)
    dataset_name = dct.pop("dataset")
    data_kwargs = dct.pop('data_kwargs', {})
    dataset = Modules.VISION.get_class_from_data(dataset_name)(**data_kwargs, test_only=True)
    printd("done")

    printd("predicting test")
    embd = model.predict(dataset.get_x_test())[0]
    printd("done")

    if args.inp:
        printd("using input")
        from vision.evaluate import get_masked_ds
        masked_inp = get_masked_ds(model, dataset=dataset).get_x_test()

        printd("running masked inputs...")
        calculate_class_mean_likelihood(model, module, pred=masked_inp, save=True, mse=args.mse,
                                        repeats=args.repeats, examples_per_class=args.examples, temp=args.temp,
                                        inp=True, **kwargs)
        printd("done")

    printd("running individual pathways...")
    calculate_class_mean_likelihood(model, module, ds=dataset, pred=embd, save=True, mse=args.mse,
                                    repeats=args.repeats, examples_per_class=args.examples, temp=args.temp,
                                    inp=False, **kwargs)
    printd("done")


