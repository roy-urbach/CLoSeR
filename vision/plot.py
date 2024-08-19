from utils.figure_utils import plot_history
from vision.utils.figures_utils import *
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Plot a figure')
    parser.add_argument('-r', '--regex', type=str, help='a regex to filter models', required=True)
    parser.add_argument('--action', choices=['h', 'p', 'd', 'm'], default=True)
    parser.add_argument('--kwargs', type=str, default="{}")
    parser.add_argument('-m', '--module', type=str, default=Modules.VISION.name,
                        choices=[Modules.VISION.name, Modules.NEURONAL.name])
    parsed_args = parser.parse_known_args()
    return parsed_args


if __name__ == "__main__":
    parsed_args, bsub_args = parse()
    import matplotlib
    matplotlib.use('Agg')

    module = [module for module in Modules if module.name == argparse.module][0]

    kwargs = eval(parsed_args.kwargs)
    if parsed_args.action == 'h':
        plot_history(parsed_args.regex, module=module, **kwargs, save=True)
    elif parsed_args.action == 'p':
        for model in regex_models(parsed_args.regex, module=module):
            plot_positional_encoding(model, **kwargs, save=True)
    elif parsed_args.action == 'd':
        n = len(parsed_args.regex.split(','))
        fig = None
        for i, model_format in enumerate(parsed_args.regex.split(',')):
            fig = plot_lines_different_along_d(model_format, **kwargs, save=(i+1)==n, c_shift=i,
                                               legend=model_format, fig=fig)
    elif parsed_args.action == 'm':
        plot_measures(parsed_args.regex, **kwargs, save=True)
