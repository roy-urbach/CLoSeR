from utils.figures_utils import *
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Plot a figure')
    parser.add_argument('-r', '--regex', type=str, help='a regex to filter models', required=True)
    parser.add_argument('--action', choices=['h', 'p', 'd'], default=True)
    parser.add_argument('--kwargs', type=str, default="{}")
    args = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args, bsub_args = parse()
    import matplotlib
    matplotlib.use('Agg')

    kwargs = eval(args.kwargs)
    if args.action == 'h':
        plot_history(args.regex, **kwargs, save=True)
    elif args.action == 'p':
        for model in regex_models(args.regex):
            plot_positional_encoding(model, **kwargs, save=True)
    elif args.action == 'd':
        n = len(args.regex.split(','))
        fig = None
        for i, model_format in enumerate(args.regex.split(',')):
            fig = plot_lines_different_along_d(model_format, **kwargs, save=(i+1)==n, c_shift=i, legend=model_format, fig=fig)

        plt.subplot(211)
        plt.legend()
