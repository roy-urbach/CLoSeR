import argparse
import os

from utils.model.model import Modules


def parse():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')
    parser.add_argument('-m', '--module', type=str, default=Modules.VISION.name,
                        choices=[Modules.VISION.name, Modules.NEURONAL.name])

    args = parser.parse_known_args()
    return args


def run():
    args = parse()[0]
    model_name = args.json.split('.json')[0]
    module = [module for module in Modules if module.name == argparse.module][0]
    kwargs = module.load_json(args.json)

    txt = '\n'.join([str(s) for s in [model_name, args.__dict__, kwargs]])

    print(txt, flush=True)

    import sys
    sys.stdout.flush()

    training_fn = os.path.join(module.get_models_path(), model_name, "is_training")
    if os.path.exists(training_fn):
        print("already training")
        return
    else:
        with open(training_fn, 'w') as f:
            f.write("Yes!")

    try:
        from utils.model.model import train
        train(model_name, module, **kwargs, batch_size=args.batch, num_epochs=args.epochs)
    finally:
        os.remove(training_fn)


if __name__ == '__main__':
    import run_before_script
    run_before_script.run()

    run()
