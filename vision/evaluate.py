from utils.io_utils import load_json, save_json
from utils.tf_utils import load_model_from_json
from utils.utils import get_class
from model.layers import *
from model.losses import *

RESULTS_FILE_NAME = 'classification_eval'


def main():
    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Evaluate a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('--knn', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--linear', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--ensemble', action=argparse.BooleanOptionalAction, default=True)
        return parser.parse_args()

    args = parse()
    return evaluate(args.json, knn=args.knn, linear=args.linear, ensemble=args.ensemble, save_results=True)


def evaluate(model, knn=False, linear=True, ensemble=True, save_results=False):
    if isinstance(model, str):
        kwargs = load_json(model)
        model = load_model_from_json(model)

    from utils import data
    dataset = get_class(kwargs.get('dataset', 'Cifar10'), data)()

    x_train_embd = model.predict(dataset.get_x_train())
    x_test_embd = model.predict(dataset.get_x_test())
    embd_dataset = data.Data(x_train_embd[0], dataset.get_y_train(), x_test_embd[0], dataset.get_y_test())

    from evaluation.evaluation import classify_head_eval

    base_path = f'models/{model.name}'

    results = load_json(RESULTS_FILE_NAME, base_path=base_path)
    if results is None:
        results = {}

    save_res = lambda *inputs: save_json(RESULTS_FILE_NAME, results, base_path=base_path) if save_results else None

    if knn:
        for k in [1] + list(range(5, 50, 5)):
            print(f"k={k}:", end='\t')
            results[f"k={k}"] = classify_head_eval(embd_dataset, linear=False, k=k)
            save_res()
    if linear:
        results['logistic'] = classify_head_eval(embd_dataset, linear=True, svm=False)
        save_res()
    if ensemble:
        ds_ens = data.Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test())
        results['ensemble_logistic'] = classify_head_eval(ds_ens, linear=True, svm=False, ensemble=True)
        save_res()
    return results


if __name__ == '__main__':
    main()
