import os

from utils.data import Data
from utils.io_utils import save_json, load_json
from utils.model.callbacks import StopIfNaN
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import unknown_args_to_dict, printd
from tqdm import tqdm as counter
import numpy as np


RESULTS_FILE_NAME_K = 'classification_eval_{k}'


@Modules.add_method
def save_evaluation_k(model_name, dct, k):
    save_json(os.path.join(model_name, RESULTS_FILE_NAME_K.format(k=k)), dct)


@Modules.add_method
def load_evaluation_k(model_name, k):
    results = load_json(os.path.join(model_name, RESULTS_FILE_NAME_K.format(k=k)))
    return results


def evaluate_k(model, module, k, repeats, **kwargs):
    model_kwargs = module.load_json(model, config=True)
    assert model_kwargs is not None
    model = load_model_from_json(model, module)
    dataset = module.get_class_from_data(model_kwargs.get('dataset', 'Cifar10'))(module=module,
                                                                                 **model_kwargs.get('data_kwargs', {}),
                                                                                 split=True)

    printd("getting embedding...")
    printd("train...", end='\t')
    x_train_embd = model.predict(dataset.get_x_train())[0]
    printd("done!")

    printd("test...", end='\t')
    x_test_embd = model.predict(dataset.get_x_test())[0]
    printd("done!")

    printd("validation...", end='\t')
    x_val_embd = model.predict(dataset.get_x_val())[0]
    printd("done!")

    P = x_train_embd.shape[-1]

    embd_dataset = Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test(),
                        x_val=x_val_embd, y_val=dataset.get_y_val(), normalize=True)
    save_res = lambda *inputs: module.save_evaluation_json_k(model.name, results, k=k)

    from utils.evaluation.evaluation import classify_head_eval
    results = {}
    for rep in counter(range(repeats)):
        chosen_ps = np.random.permutation(P)[:k]
        results.setdefault('k', []).append(list(chosen_ps))
        cur_ds = Data(embd_dataset.get_x_train()[..., chosen_ps], dataset.get_y_train(),
                      embd_dataset.get_x_test()[..., chosen_ps], dataset.get_y_test(),
                      x_val=embd_dataset.get_x_val()[..., chosen_ps], y_val=dataset.get_y_val(), normalize=False)
        results.setdefault('res', []).append(classify_head_eval(cur_ds, linear=True, svm=False, categorical=True, **kwargs))
        save_res()

    return results


def evaluate_subset_main():
    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Evaluate a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('-k', '--k_pathways', type=int, help='num pathways to use for decoding')
        parser.add_argument('-r', '--repeats', type=int, help='num samples from P choose k', default=10)
        parser.add_argument('-m', '--module', type=str, default=Modules.VISION,
                            choices=Modules.get_cmd_module_options())
        return parser.parse_known_args()

    args, unknown_args = parse()
    kwargs = unknown_args_to_dict(unknown_args, warning=True)

    import warnings
    warnings.warn(f"unrecognized args: {kwargs}")
    args.json = ".".join(args.json.split(".")[:-1]) if args.json.endswith(".json") else args.json
    module = Modules.get_module(args.module)
    if module != Modules.VISION:
        raise NotImplementedError("didn't implement it for non-vision tasks yet")
    evaluating_fn = os.path.join(module.get_models_path(), args.json, 'is_evaluating_k')

    if os.path.exists(os.path.join(module.get_models_path(), args.json, StopIfNaN.FILENAME)):
        print(f"NaN issue, not evaluating")
        return

    if os.path.exists(evaluating_fn):
        print("already evaluating!")
        return
    else:
        with open(evaluating_fn, 'w') as f:
            f.write("Yes!")

    try:
        evaluate_k(args.json, module=module, k=args.k, repeats=args.repeats)
    finally:
        os.remove(evaluating_fn)


if __name__ == '__main__':
    import run_before_script
    run_before_script.run()

    evaluate_subset_main()