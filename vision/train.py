import argparse
import os

from vision.utils.consts import VISION_MODELS_DIR
from vision.utils.io_utils import load_json


def parse():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')

    args = parser.parse_known_args()
    return args


def run():
    args = parse()[0]
    model_name = args.json.split('.json')[0]
    kwargs = load_json(args.json)

    txt = '\n'.join([str(s) for s in [model_name, args.__dict__, kwargs]])

    print(txt, flush=True)

    import sys
    sys.stdout.flush()

    training_fn = f"{VISION_MODELS_DIR}/{model_name}/is_training"
    if os.path.exists(training_fn):
        print("already training")
        return
    else:
        with open(training_fn, 'w') as f:
            f.write("Yes!")

    try:
        from model.model import train
        train(model_name, **kwargs, batch_size=args.batch, num_epochs=args.epochs)
    finally:
        os.remove(training_fn)


if __name__ == '__main__':
    import run_before_script
    run_before_script.run()

    run()
