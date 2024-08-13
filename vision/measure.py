from vision.measures.utils import save_measures_json, measure_model, get_measuring_time, load_measures_json
from vision.utils.consts import VISION_MODELS_DIR


def main():
    import argparse

    def parse():
        parser = argparse.ArgumentParser(description='Measure some metrics about the pathways of a model')
        parser.add_argument('-j', '--json', type=str, help='name of the config json')
        parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
        parser.add_argument('-i', '--iterations', type=int, default=50, help='number of repeats')
        parser.add_argument('--override', action=argparse.BooleanOptionalAction, default=False)
        return parser.parse_args()

    args = parse()
    args.json = ".".join(args.json.split(".")[:-1]) if args.json.endswith(".json") else args.json
    measuring_fn = f'{VISION_MODELS_DIR}/{args.json}/is_measuring'

    import os
    if os.path.exists(measuring_fn):
        print("already measuring!")
        return
    else:
        with open(measuring_fn, 'w') as f:
            f.write("Yes!")

    try:
        res = measure(args.json, b=args.batch, iterations=args.iterations, save_results=True, override=args.override)
    finally:
        os.remove(measuring_fn)
    return res


def measure(model, save_results=False, override=False, **kwargs):
    model_name = model if isinstance(model, str) else model.name
    if not override:
        from utils.io_utils import get_output_time
        output_time = get_output_time(model)
        measuring_time = get_measuring_time(model)

        if output_time and measuring_time and output_time < measuring_time:
            print(f"Tried to measure, but output time is {output_time} and measuring time is {measuring_time} and override is False")
            return load_measures_json(model_name)

    results = load_measures_json(model_name) if not override else {}

    if results is None:
        results = {}
    res_dct = measure_model(model, **kwargs)
    if save_results:
        save_measures_json(model_name, res_dct)

    return results


if __name__ == '__main__':
    import run_before_script

    run_before_script.run()

    main()
