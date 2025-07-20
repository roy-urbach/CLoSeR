from utils.model.callbacks import StopIfNaN
from utils.modules import Modules
from utils.measures.utils import measure_model
import os


def main():
    """
    A module-general cross-path measures script
    """
    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Measure some metrics about the pathways of a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('-m', '--module', type=str, default=Modules.VISION, choices=Modules.get_cmd_module_options())
        parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
        parser.add_argument('-i', '--iterations', type=int, default=50, help='number of repeats')
        parser.add_argument('--override', action=argparse.BooleanOptionalAction, default=False)
        return parser.parse_args()

    args = parse()
    args.json = ".".join(args.json.split(".")[:-1]) if args.json.endswith(".json") else args.json
    module = Modules.get_module(args.module)
    measuring_fn = os.path.join(module.get_models_path(), args.json, 'is_measuring')

    # if nan, don't evaluate
    if os.path.exists(os.path.join(module.get_models_path(), args.json, StopIfNaN.FILENAME)):
        print(f"NaN issue, not measuring")

    if os.path.exists(measuring_fn):
        # if already measuring, don't measure too. Useful for parallelization
        print("already measuring!")
        return
    else:
        # mark that you are evaluating, so that parallel jobs won't remeasure
        with open(measuring_fn, 'w') as f:
            f.write("Yes!")

    try:
        res = measure(args.json, module, b=args.batch, iterations=args.iterations, save_results=True, override=args.override)
    finally:
        # mark that you are no longer evaluating
        os.remove(measuring_fn)
    return res


def measure(model, module:Modules, save_results=False, override=False, **kwargs):
    model_name = model if isinstance(model, str) else model.name

    # if override or the model changed since the last time it was measured, measure

    if not override:
        output_time = module.get_output_time(model)
        measuring_time = module.get_measuring_time(model)

        if output_time and measuring_time and output_time < measuring_time:
            print(f"Tried to measure, but output time is {output_time} and measuring time is {measuring_time} and override is False")
            return module.load_measures_json(model_name)

    results = module.load_measures_json(model_name) if not override else {}

    if results is None:
        results = {}
    res_dct = measure_model(model, module, **kwargs)
    if save_results:
        module.save_measures_json(model_name, res_dct)

    return results


if __name__ == '__main__':
    import run_before_script

    run_before_script.run()

    main()
