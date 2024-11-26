from utils.model.callbacks import StopIfNaN
from utils.modules import Modules
import os

from utils.utils import streval, unknown_args_to_dict


def check_evaluation_time(model, module: Modules):
    from utils.io_utils import get_output_time
    output_time = get_output_time(model, module)
    evaluation_time = module.get_evaluation_time(model)

    if output_time and evaluation_time and output_time < evaluation_time:
        print(
            f"Tried to evaluate, but output time is {output_time} and evaluation time is {evaluation_time} and override is False")
        return False
    else:
        return True


def main():
    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Evaluate a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('-m', '--module', type=str, default=Modules.VISION, choices=Modules.get_cmd_module_options())
        parser.add_argument('--knn', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--linear', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--inp', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--ensemble', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--override', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--fill', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--override_linear', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--ks', type=str, default=[1] + list(range(5, 21, 5)), help='what ks to use')
        return parser.parse_known_args()

    args, unknown_args = parse()
    kwargs = unknown_args_to_dict(unknown_args, warning=True)
    import warnings
    warnings.warn(f"unrecognized args: {kwargs}")
    args.json = ".".join(args.json.split(".")[:-1]) if args.json.endswith(".json") else args.json
    module = Modules.get_module(args.module)
    evaluating_fn = os.path.join(module.get_models_path(), args.json, 'is_evaluating')

    if os.path.exists(os.path.join(module.get_models_path(), args.json, StopIfNaN.FILENAME)):
        print(f"NaN issue, not evaluating")
        return

    if os.path.exists(evaluating_fn):
        print("already evaluating!")
        return
    else:
        with open(evaluating_fn, 'w') as f:
            f.write("Yes!")

    if not args.override and not args.override_linear and not args.fill:
        if not check_evaluation_time(args.json, module):
            return module.load_evaluation_json(args.json)

    try:
        res = module.evaluate(args.json, knn=args.knn, linear=args.linear, ensemble=args.ensemble,
                              save_results=True, dataset=None, override=args.override, override_linear=args.override_linear,
                              inp=args.inp, ks=streval(args.ks), **kwargs)
    finally:
        os.remove(evaluating_fn)

    return res


if __name__ == '__main__':
    import run_before_script

    run_before_script.run()

    main()
