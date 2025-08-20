import argparse
import os

from utils.model.callbacks import StopIfNaN
from utils.modules import Modules
from utils.utils import printd


def parse():
    """
    Parse the arguments for the script
    """
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')
    parser.add_argument('-m', '--module', type=str, default=Modules.VISION.name,
                        choices=Modules.get_cmd_module_options())

    args = parser.parse_known_args()
    return args


def run():
    """
    A module-general training script
    """
    args = parse()[0]
    model_name = args.json.split('.json')[0]
    module = Modules.get_module(args.module)
    kwargs = module.load_json(args.json, config=True)

    txt = '\n'.join([str(s) for s in [model_name, args.__dict__, kwargs]])

    printd(txt)

    # if model got to NaNs in the past, don't continue training
    if os.path.exists(os.path.join(module.get_models_path(), model_name, StopIfNaN.FILENAME)):
        print(f"NaN issue, not training")
        return

    # if the model is already training in another job, don't train in parallel
    training_fn = os.path.join(module.get_models_path(), model_name, "is_training")
    if os.path.exists(training_fn):
        print("already training")
        return
    else:
        os.makedirs(os.path.dirname(training_fn), exist_ok=True)
        # mark that you are training the model, so that other jobs won't override you
        with open(training_fn, 'w') as f:
            f.write("Yes!")

    try:
        # train the model!
        from utils.model.model import train
        train(model_name, module, **kwargs, batch_size=args.batch, num_epochs=args.epochs)
    finally:
        # mark that you are no longer training the model
        os.remove(training_fn)


if __name__ == '__main__':
    # this is the script to run to train the model. See train_cmd_format for the command format for running jobs

    import run_before_script
    run_before_script.run()

    run()
