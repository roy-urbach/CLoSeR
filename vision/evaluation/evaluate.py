def eval():
    import sys
    print(sys.path)

    from utils.io_utils import load_json, save_json
    from utils.tf_utils import load_model_from_json
    from utils.utils import get_class

    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Evaluate a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('--knn', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--linear', action=argparse.BooleanOptionalAction, default=True)
        args = parser.parse_known_args()
        return args

    args = parse()

    kwargs = load_json(args.json)

    from utils import data
    dataset = get_class(kwargs.get('dataset', 'Cifar10'), data)()

    model = load_model_from_json(args.json)

    x_train_embd = model.predict(dataset.get_x_train())[0]
    x_test_embd = model.predict(dataset.get_x_test())[0]
    embd_dataset = data.Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test())
    from evaluation.evaluation import classify_head_eval
    results = {}
    if args.knn:
        for k in [1] + list(range(5, 50, 5)):
            print(f"k={k}:", end='\t')
            results[f"k={k}"] = classify_head_eval(embd_dataset, linear=False, k=k)
    if args.linear:
        results['logistic'] = classify_head_eval(embd_dataset, linear=True, svm=False)

    save_json('classification_eval', results, base_path=f'models/{model.name}')


if __name__ == '__main__':
    eval()
