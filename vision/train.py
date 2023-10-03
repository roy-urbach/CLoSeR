import argparse
from utils.io_utils import *


def parse():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')

    args = parser.parse_args()
    return args


def run():
    args = parse()
    model_name = args.json.split('.json')[0]
    kwargs = load_json(args.json)

    print(model_name)
    print(args.__dict__)
    print(kwargs)

    # TODO: uncomment to actually run
    # from utils.model import train
    # train(model_name, **kwargs, batch_size=args.batch, num_epochs=args.epochs)


if __name__ == '__main__':
    run()
