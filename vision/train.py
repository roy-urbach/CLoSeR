import argparse
from utils.io_utils import *


def parse():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')

    args = parser.parse_args()
    return args


def run():
    args = parse()
    model_name = args.json.split('.json')[0]
    kwargs = load_json(args.json)

    print(model_name, flush=True)
    print(args.__dict__, flush=True)
    print(kwargs, flush=True)

    import sys
    sys.stdout.flush()

    import time
    time.sleep(3)
    print("DONE!")

    dummy = f"models/{model_name}/dummy.o"
    with open(dummy, 'w') as f:
        f.write("This is a dummy")
    # os.remove(dummy)

    # TODO: uncomment to actually run
    # from utils.model import train
    # train(model_name, **kwargs, batch_size=args.batch, num_epochs=args.epochs)


if __name__ == '__main__':
    run()
