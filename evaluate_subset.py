import os

from utils.model.callbacks import StopIfNaN
from utils.modules import Modules
from utils.utils import unknown_args_to_dict


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
        module.evaluate_k(args.json, module=module, k=args.k_pathways, repeats=args.repeats)
    finally:
        os.remove(evaluating_fn)


if __name__ == '__main__':
    import run_before_script
    run_before_script.run()

    evaluate_subset_main()