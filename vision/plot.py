from utils.figures_utils import *
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Plot a figure')
    parser.add_argument('-r', '--regex', type=str, help='a regex to filter models', required=True)
    parser.add_argument('-k', '--keys', type=str, help='an iterable array of keys to plot', required=False, default=None)
    parser.add_argument('--history', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--history_kwargs', type=str, default="{}")
    args = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args, bsub_args = parse()
    import matplotlib
    matplotlib.use('Agg')
    if args.history:
        plot_history(args.regex, keys=eval(args.keys) if args.keys else None, **eval(args.history_kwargs), save=True)