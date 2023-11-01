from evaluation.ensemble import EnsembleVotingMethods
from evaluation.evaluation import classify_head_eval_ensemble
from evaluation.utils import save_evaluation_json, load_evaluation_json
from utils.data import Cifar10
from utils.io_utils import load_json
from utils.tf_utils import load_model_from_json
from utils.utils import get_class


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
    return evaluate(args.json, knn=args.knn, linear=args.linear,
                    ensemble=args.ensemble, save_results=True, dataset=None)


def evaluate(model, knn=False, linear=True, ensemble=True, save_results=False, dataset=Cifar10()):
    from utils import data

    if isinstance(model, str):
        kwargs = load_json(model)
        assert kwargs is not None
        model = load_model_from_json(model)
        if dataset is None:
            dataset = get_class(kwargs.get('dataset', 'Cifar10'), data)()

    x_train_embd = model.predict(dataset.get_x_train())[0]
    x_test_embd = model.predict(dataset.get_x_test())[0]
    embd_dataset = data.Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test())

    from evaluation.evaluation import classify_head_eval

    results = load_evaluation_json(model.name)

    if results is None:
        results = {}

    save_res = lambda *inputs: save_evaluation_json(model.name, results) if save_results else None

    if knn:
        for k in [1] + list(range(5, 50, 5)):
            print(f"k={k}:", end='\t')
            results[f"k={k}"] = classify_head_eval(embd_dataset, linear=False, k=k)
            save_res()
    if linear:
        results['logistic'] = classify_head_eval(embd_dataset, linear=True, svm=False)
        save_res()

    if ensemble:
        results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False, ensemble=True,
                                                   voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]))
        save_res()
    return results


if __name__ == '__main__':
    import run_before_script
    run_before_script.run()

    main()
