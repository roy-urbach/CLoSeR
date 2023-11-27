from evaluation.ensemble import EnsembleVotingMethods
from evaluation.evaluation import classify_head_eval_ensemble
from evaluation.utils import save_evaluation_json, load_evaluation_json, get_evaluation_time
from model.model import load_model_from_json
from utils.data import Cifar10, Data
from utils.io_utils import load_json
from utils.utils import get_class
from utils import data
import numpy as np


def get_masked_ds(model, dataset=Cifar10()):
    if isinstance(model, str):
        model = load_model_from_json(model)
    patch_layer = model.get_layer(model.name + '_patch')
    pathway_indices = model.get_layer(model.name + '_pathways').indices.numpy()
    setup_func = lambda x: np.transpose(patch_layer(x).numpy()[:, pathway_indices - 1], [0, 1, 3, 2]).reshape(
        x.shape[0], -1, pathway_indices.shape[-1])
    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test())
    return ds


def main():
    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Evaluate a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('--knn', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--linear', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--ensemble', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--ensemble_knn', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--override', action=argparse.BooleanOptionalAction, default=False)
        return parser.parse_args()

    args = parse()
    return evaluate(args.json, knn=args.knn, linear=args.linear, ensemble=args.ensemble, ensemble_knn=args.ensemble_knn,
                    save_results=True, dataset=None, override=args.override)


def evaluate(model, knn=False, linear=True, ensemble=True, ensemble_knn=False, save_results=False, override=False, dataset=Cifar10(), **kwargs):

    if not override:
        from utils.io_utils import get_output_time
        output_time = get_output_time(model)
        evaluation_time = get_evaluation_time(model)

        if output_time and evaluation_time and output_time < evaluation_time:
            print(f"Tried to evaluate, but output time is {output_time} and evaluation time is {evaluation_time} and override is False")
            return load_evaluation_json(model)

    if isinstance(model, str):
        model_kwargs = load_json(model)
        assert model_kwargs is not None
        model = load_model_from_json(model)
        if dataset is None:
            dataset = get_class(model_kwargs.get('dataset', 'Cifar10'), data)()

    x_train_embd = model.predict(dataset.get_x_train())[0]
    x_test_embd = model.predict(dataset.get_x_test())[0]
    embd_dataset = data.Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test())

    from evaluation.evaluation import classify_head_eval

    results = load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: save_evaluation_json(model.name, results) if save_results else None

    if knn:
        for k in [1] + list(range(5, 50, 5)):
            print(f"k={k}:", end='\t')
            results[f"k={k}"] = classify_head_eval(embd_dataset, linear=False, k=k, **kwargs)
            save_res()

    if linear:
        results['logistic'] = classify_head_eval(embd_dataset, linear=True, svm=False, **kwargs)
        save_res()

    if ensemble:
        results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                   voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]), **kwargs)
        save_res()

        if not any([k.startswith("image_pathway") for k in results.keys()]):
            masked_ds = get_masked_ds(model, dataset=dataset)
            results.update(classify_head_eval_ensemble(masked_ds, base_name='image_', svm=False,
                                                       voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]), **kwargs)
            save_res()

    if ensemble_knn:
        results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=15,
                                                   voting_methods=EnsembleVotingMethods), **kwargs)
        save_res()
    return results


if __name__ == '__main__':
    import run_before_script

    run_before_script.run()

    main()
