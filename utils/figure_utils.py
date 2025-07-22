from itertools import product
from json import JSONDecodeError

from matplotlib import pyplot as plt
from scipy import stats

from utils import plot_utils
from utils.modules import Modules
import numpy as np
import os
import re

from utils.plot_utils import savefig, calculate_square_rows_cols, simpleaxis, NameAndColor, YLABEL_CLASS_F


def regex_models(regex, module:Modules):
    return list(filter(lambda m: re.match(regex, m), os.listdir(module.get_models_path())))


def name_to_q(name, module=Modules.VISION, naive=True, mapping={}, frac=True, patches=None):
    if naive:
        return eval(re.findall("0\.\d+", name)[-1])
    else:
        if name in mapping:
            return mapping[name]
        else:
            if patches:
                d = f"{int(module.load_json(name, config=True)['model_kwargs']['pathways_kwargs']['d'] * patches)}/{patches}"
            else:
                model = module.load_model_from_json(name, load=False)
                pathways_layer = model.get_layer(model.name + '_pathways')
                if frac:
                    d = f"{pathways_layer.num_patches_per_path}/{pathways_layer.num_patches}"
                else:
                    d = pathways_layer.num_patches_per_path / pathways_layer.num_patches
            mapping[name] = d
            return d


def load_classfications_by_regex(model_regex, module, name_to_q_naive=False, convert_name=True, negative_regex=None, print_err=True):
    base_path = module.get_models_path()
    archive = {}
    for m in os.listdir(base_path):
        if re.match(model_regex, m) and (negative_regex is None or not re.match(negative_regex, m)):
            try:
                eval_json = module.load_evaluation_json(m)
            except Exception as err:
                if print_err: print(f"Couldn't load {m} because of {err.__str__()}")
                continue
            if eval_json is not None:
                if convert_name:
                    m = name_to_q(m, module=module, naive=name_to_q_naive)
                archive[m] = eval_json
    return archive


def plot_history(model_regex, module: Modules, window=10, name_to_name=lambda m: m, keys=None, keys_f=None,
                 log_keys={'embedding'}, plot_train=True, plot_val=True, name_to_c=None, save=None, legend=True, title=None,
                 save_f=None):
    """
    Plots the training history of models
    :param model_regex: regex for all models that should be plotted
    :param module: the module
    :param window: smoothing window (convolution with average)
    :param name_to_name: a function that receives a model name and outputs the label for the plot
    :param keys: a container with names of metric keys to plot
    :param keys_f: if keys is None and this is not None, chooses keys based on this boolean function which receieves
                    a string returns a boolean value
    :param log_keys: Same as keys_f, but whether to change the y axis to log scale
    :param plot_train: whether to plot the train metrics or not
    :param plot_val: whether to plot the validation metrics or not
    :param name_to_c: a function that converts a string to a color
    :param save: whether to save this figure in "figures/{k}.png" where k is the metric name
    :param legend: whether to call plt.legend
    :param title: title for the figure
    :param save_f: if given, calls this function with the metric name
    :return: None
    """
    models = sorted(regex_models(model_regex, module))
    models_names = []
    orig_names = []
    histories = {}
    for model in models:
        history = module.load_history(model)
        if history:
            orig_names.append(model)
            models_names.append(name_to_name(model))
            for k, v in history.items():
                k = k.replace(model+'_', "")
                histories.setdefault(k, {})[name_to_name(model)] = v
    if keys is None and keys_f is not None:
        keys = {key for key in histories.keys() if keys_f(key)}

    smooth = lambda arr, window: np.convolve(arr, [1/window]*window, mode='valid')
    for k, v in histories.items():
        if k.startswith("val") or k == "loss": continue
        if keys is not None and k not in keys: continue
        plt.figure()
        plt.title((model_regex if title is None else title) + " " + k)
        for i, (model_name, value) in enumerate(v.items()):
            if plot_train:
                if len(value) > window:
                    plt.plot(smooth(value, window), label=model_name,
                             c=f"C{i}" if name_to_c is None else name_to_c(orig_names[i]))
            if plot_val:
                val = histories["val_"+k][model_name]
                if len(val) > window:
                    plt.plot(smooth(val, window), label=model_name if not plot_train else None,
                             c=f"C{i}" if name_to_c is None else name_to_c(orig_names[i]), linestyle=':' if plot_train else '-')
        if any([key in k for key in log_keys]):
            plt.yscale("log")
        if legend:
            plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(k)
        if save:
            savefig(f"figures/{k}.png")
        if save_f:
            save_f(k)


def plot_metrics_along_q(model_regex, module: Modules, metric_regex=("logistic", '.*linear_.*', '.*knn.*'),
                         name_to_q_naive=False, baseline_name=None, metric_to_label=None, xticks=False):
    archive = load_classfications_by_regex(model_regex, module, name_to_q_naive=name_to_q_naive)
    best = max([max([v[-1] for v in val.values()]) for val in archive.values()])
    print(best)
    sorted_keys = np.array([k for k in sorted(archive.keys(), key=lambda k: eval(k))])
    eval_keys = np.array([eval(k) for k in sorted_keys])
    keys_labels = [r'$\frac{{{0}}}{{{1}}}$'.format(*k.split('/')) for k in sorted_keys]

    ax = None
    relevant_metrics = set(classi for dct in archive.values() for classi in dct if
                           any([re.match(regex, classi) for regex in metric_regex]))
    rows, cols = calculate_square_rows_cols(len(relevant_metrics))
    plt.figure(figsize=(4 * cols, 3 * rows))
    plt.suptitle(model_regex)
    subplot = 1
    for classi in relevant_metrics:
        ax = plt.subplot(rows, cols, subplot, sharey=ax)
        plt.title(classi)
        for i in range(2):
            plt.scatter(eval_keys, [archive[k].get(classi, {i: np.nan})[i] for k in sorted_keys],
                        label=(metric_to_label(classi) if metric_to_label is not None else classi) + [' train', ' test'][i])
        if baseline_name:
            plt.axhline(0.4, linestyle=":", c='g', label=baseline_name)
        if not classi.startswith('k'): plot_utils.legend(loc='upper right' if best > 0.5 else "lower right")
        plt.xlabel("d")
        plt.ylabel("accuracy")
        if xticks:
            plt.grid(alpha=0.2)
            plt.xlim(0, 1)
        subplot += 1
        plt.xticks(eval_keys, keys_labels)
        simpleaxis(ax)
    plt.tight_layout()
    plt.show()
    return archive


def gather_results_over_all_args(model_format, args, module=Modules.VISION, name='logistic', seeds=[1],
                                 measure=False, print_missing=True):
    """
    loads evaluations for many models with the same format and returns as a tensor, whose shape is a cartesian product of args
    :param model_format: a format model name
    :param args: a dictionary where the keys are argument names and the values are arguments (for examples {"q": [1,2,3,4,5]})
    :param module: Module
    :param name: what metric to load
    :param seeds: seeds to iterate over
    :param measure: whether it is a measure (see utils/measures)
    :param print_missing: whether to print the names of models that couldn't be loaded
    :return: a numpy tensor with shape [arg1, ..., argK, seed, metric_dim1, ..., metric_dimD]
    """
    names = list(args.keys())
    args = [args[n] for n in names]
    shape = [len(arg) for arg in args]
    res = None

    for i, inds in enumerate(product(*[range(s) for s in shape])):
        for s, seed in enumerate(seeds):
            model_name = model_format.format(
                **{k: v for k, v in zip(names, [args[arg_ind][cur_ind] for arg_ind, cur_ind in enumerate(inds)])},
                seed=seed)
            try:
                dct = module.load_measures_json(model_name) if measure else module.load_evaluation_json(model_name)
            except JSONDecodeError as err:
                dct = None
                if print_missing:
                    print(model_name)
            if dct is None:
                val = np.nan
            else:
                val = dct.get(name, np.nan)
            if val is np.nan and print_missing:
                print(f"val is nan: {model_name}")
            if not isinstance(val, float) and not np.isnan(val).all():
                if res is None:
                    res = np.full(list(shape) + [len(seeds)] + ([len(val)] if not measure else (list(val.shape) if isinstance(val, np.ndarray) else [len(val)])), np.nan)
                cur = res
                for cur_ind in inds:
                    cur = cur[cur_ind]
                if cur[s].shape != np.array(val).shape and print_missing:
                    print(f"val is shaped 2 instead of 3 for {model_name}")
                cur[s] = val if np.array(val).shape == cur[s].shape else np.array([val[0], np.nan, val[1]])
    return res


def plot_lines_different_along_q(model_format, module:Modules, seeds, args, qs, name="logistic", save=False, measure=False, mask=None,
                                 arg=None, legend=True, fig=None, c_shift=0, train=False, baseline=None, sem=False, c=None,
                                 xticks=False, marker=None, ax=None, suptitle=False, title=False, **kwargs):
    if isinstance(args, str):
        args = eval(args)
    if isinstance(qs, str):
        qs = eval(qs)
    res = gather_results_over_all_args(model_format, module=module, name=name,
                                       seeds=seeds,
                                       args={arg: args, 'd': qs} if arg else {'d': qs},
                                       measure=measure,
                                       **kwargs)
    if res is None or np.isnan(res).all():
        print(f"for {model_format} and {name}, everything is nan, so not plotting")
        return
    fig = plt.figure(figsize=(8, 4+8*train)) if fig is None else fig
    if suptitle:
        plt.suptitle(model_format + " " + name + f" different {arg}")
    only_test = measure or not train
    given_ax = ax is not None
    for i in range(res.shape[-1] if not measure else 1):
        if not measure and only_test and i != (res.shape[-1] - 1): continue
        if not given_ax and not only_test:
            ax = plt.subplot(1 if measure or only_test else res.shape[-1], 1, 1 if measure or only_test else i+1, sharey=ax)
        if title:
            (ax.set_title if ax is not None else plt.title)((["Train", "Test"] if res.shape[-1] == 2 else ['Train', 'Val', 'Test'][i]) if not measure else 'Test')
        if arg:
            for ind, identity in enumerate(args):
                relevant_part = res[ind, ..., i] if not measure else np.stack([np.stack([res[ind, i_d, s][mask if mask is not None else ~np.eye(res.shape[-1], dtype=bool)]
                                                                                         for s in range(res.shape[2])], axis=0)
                                                                               for i_d in range(len(res[ind]))], axis=0)
                mean = np.nanmean(relevant_part, axis=(-2, -1))
                std_err_mean = np.nanstd(relevant_part, ddof=1, axis=(-2, -1)) / np.sqrt(np.sum(~np.isnan(relevant_part), axis=(-2, -1)))
                if sem:
                    CI = mean + std_err_mean[None] * np.array([-1, 1])[..., None]
                else:
                    CI = stats.norm.interval(0.975, loc=mean, scale=std_err_mean)

                color = f"C{ind+c_shift}" if c is None else (c.get_color() if isinstance(c, NameAndColor) else c)

                (ax if ax is not None else plt).plot(qs, mean, label=(legend + ' ' if isinstance(legend, str) else "") + str(identity), c=color, marker=marker)
                if len(seeds) > 1:
                    (ax if ax is not None else plt).fill_between(qs, CI[0], CI[1][ind, ..., i], color=color, alpha=0.3)
        else:
            color = f"C{c_shift}" if c is None else c
            relevant_part = res[..., i] if not measure else np.stack([np.stack([res[i_d, s][mask if mask is not None else ~np.eye(res.shape[-1], dtype=bool)]
                                                                                for s in range(res.shape[1])], axis=0)
                                                                      for i_d in range(len(res))], axis=0)
            over_axes = (-2, -1) if measure else (-1, )
            mean = np.nanmean(relevant_part, axis=over_axes)
            std_err_mean = np.nanstd(relevant_part, ddof=1, axis=over_axes) / np.sqrt(np.sum(~np.isnan(relevant_part), axis=over_axes))
            if sem:
                CI = mean + std_err_mean[None] * np.array([-1, 1])[..., None]
            else:
                CI = stats.norm.interval(0.975, loc=mean, scale=std_err_mean)

            (ax if ax is not None else plt).plot(qs, mean, label=(legend + ' ') if isinstance(legend, str) else "", c=color)
            if len(seeds) > 1:
                (ax if ax is not None else plt).fill_between(qs, CI[0], CI[1], color=color, alpha=0.3)
        if i:
            (ax.set_xlabel if ax is not None else plt.xlabel)('d')
        else:
            if legend:
                plt.legend()
        (ax.set_xlabel if ax is not None else plt.xlabel)('d')
        YLABEL_CLASS_F() if not measure else (ax.set_ylabel if ax is not None else plt.ylabel)(name)
        if xticks and module is Modules.VISION:
            from vision.utils.figures_utils import qs_to_labels
            (ax.set_xticks if ax is not None else plt.xticks)(qs, qs_to_labels(qs))
        if xticks:
            (ax if ax is not None else plt).grid(alpha=0.3)
        if not measure and baseline:
            (ax if ax is not None else plt).axhline(baseline, linestyle=':', c='k')
    plt.tight_layout()
    if save:
        savefig(f"figures/{model_format}_along_d_{arg}")
    return fig
