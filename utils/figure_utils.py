from matplotlib import pyplot as plt

from utils import plot_utils
from utils.modules import Modules
import numpy as np
import os
import re

from utils.plot_utils import savefig, calculate_square_rows_cols, simpleaxis
from utils.utils import smooth


def regex_models(regex, module:Modules):
    return list(filter(lambda m: re.match(regex, m), os.listdir(module.get_models_path())))


def name_to_d(name, module=Modules.VISION, naive=True, mapping={}, frac=True, patches=None):
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


def load_classfications_by_regex(model_regex, module, name_to_d_naive=False, convert_name=True, negative_regex=None, print_err=True):
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
                    m = name_to_d(m, module=module, naive=name_to_d_naive)
                archive[m] = eval_json
    return archive


def plot_history(model_regex, module, window=10, name_to_name=lambda m: m, log=True, keys=None,
                 log_keys={'embedding'}, plot_train=True, plot_val=True, name_to_c=None, save=None):
    models = regex_models(model_regex, module)
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
    for k, v in histories.items():
        if k.startswith("val") or k == "loss": continue
        if keys is not None and k not in keys: continue
        plt.figure()
        plt.title(model_regex + " " + k)
        for i, (model_name, value) in enumerate(v.items()):
            if plot_train:
                plt.plot(*smooth(value, window), label=model_name,
                         c=f"C{i}" if name_to_c is None else name_to_c(orig_names[i]))
            if plot_val:
                plt.plot(*smooth(histories["val_"+k][model_name], window), label=model_name if not plot_train else None,
                         c=f"C{i}" if name_to_c is None else name_to_c(orig_names[i]), linestyle=':' if plot_train else '-')
        if log and any([key in k for key in log_keys]):
            plt.yscale("log")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(k)
        if save:
            savefig(f"figures/{k}.png")


def sorted_barplot(model_regex, metric_regex, module=Modules.VISION, sort_by_train=True, show_top=None, regex_suptitle=True,
                   negative_regex=None, print_err=False, baseline=None, plot_train=True):
    archive = load_classfications_by_regex(model_regex, module, convert_name=False, negative_regex=negative_regex,
                                           print_err=print_err)
    metrics = set(metric for dct in archive.values() for metric in dct.keys() if re.match(metric_regex, metric))
    rows, cols = calculate_square_rows_cols(len(metrics))
    plt.figure(figsize=(cols * 5, rows * 3))
    if regex_suptitle:
        plt.suptitle(model_regex)
    for subplot, metric in enumerate(sorted(metrics)):
        plt.subplot(rows, cols, subplot + 1)
        plt.title(metric)
        values = {k: archive[k].get(metric, [np.nan] * 2) for k in archive}

        keys = np.array(sorted(archive.keys()))
        values_to_sort_by = np.array([values[k][0 if sort_by_train else 1] for k in keys])
        nan_mask = ~np.isnan(values_to_sort_by)
        sorted_k = keys[nan_mask][np.argsort(values_to_sort_by[nan_mask])]
        if show_top is not None:
            sorted_k = sorted_k[-show_top:]
        if plot_train:
            plt.barh(np.arange(len(sorted_k)), [values[k][0] for k in sorted_k], label='train', zorder=1)
            plt.scatter([values[k][1] for k in sorted_k], np.arange(len(sorted_k)), label='test', zorder=2)
        else:
            plt.barh(np.arange(len(sorted_k)), [values[k][1] for k in sorted_k], label='test', zorder=2)
        plt.yticks(np.arange(len(sorted_k)), sorted_k)
        if baseline:
            plt.axvline(module.load_evaluation_json(baseline)[metric][1], c='k', linestyle=':')
        plt.legend()
        if baseline:
            plt.gca().set_xlim(left=module.load_evaluation_json(baseline)[metric][1])
            # plt.xticks(np.linspace(0, 1, 11))
            # plt.xlim(load_evaluation_json(BASELINE_UNTRAINED)[metric][1], 1)
        plt.grid(axis='x', zorder=0, alpha=0.2, linewidth=2)
    if len(metrics) > 1: plt.tight_layout()


def plot_metrics_along_d(model_regex, module: Modules, metric_regex=("logistic", '.*linear_.*', '.*knn.*'),
                         name_to_d_naive=False, baseline_name=None, metric_to_label=None):
    archive = load_classfications_by_regex(model_regex, module, name_to_d_naive=name_to_d_naive)
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
        plt.grid(alpha=0.2)
        subplot += 1
        plt.xlim(0, 1)
        plt.xticks(eval_keys, keys_labels)
        simpleaxis(ax)
    plt.tight_layout()
    plt.show()
    return archive
