from measures.utils import load_measures_json
from utils.plot_utils import *
from utils import plot_utils
import re
from utils.io_utils import load_json
import os
from scipy import stats

BASELINE_NAME = r'$f^{image}_{logistic}$'
DS = (0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9)
PATCHES = 64
EXTENDED_DS = sorted([0.025, 0.04, 0.05, 0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.45, 0.4, 0.55, 0.5, 0.65, 0.6, 0.75, 0.7, 0.85, 0.8, 0.95, 0.9,])
PS = (2,3,5,10,15)
PS = np.array(PS)
SEEDS = list(range(1,5))
BASELINE_UNTRAINED = "naive_paths_10_d0.1_imgsize_32_patch_4_nobug_seed1_outdim_128_untrained"


def ds_to_labels(ds):
    ds_labels = [f"{int(d*PATCHES)}/{PATCHES}" for d in ds]
    ds_labels = [r'$\frac{{{0}}}{{{1}}}$'.format(*k.split('/')) for k in ds_labels]
    return ds_labels


EXTENDED_DS_LABELS = ds_to_labels(EXTENDED_DS)

from enum import Enum

class ClassifierLatex(Enum):
    INPUT = "image" #r"$f_s$"
    MASKED_INPUT = "masked image" # r'$f_{s_p}$'
    EMBEDDING = "individual pathway embedding" # r'$f_{\mu_p}$'
    ENSEMBLE = "full embedding" # r'$f_\mu$'
    MEAN_ENSEMBLE = r'$\bar{f}_{\mu_p}$'

def Acc(string=r'$f$'):
    return r"$Acc($" + string + r"$)$"


def name_to_d(name, naive=True, mapping={}, frac=True):
    if naive:
        return eval(re.findall("0\.\d+", name)[-1])
    else:
        if name in mapping:
            return mapping[name]
        else:
            if PATCHES:
                d = f"{int(load_json(name)['model_kwargs']['pathways_kwargs']['d'] * PATCHES)}/{PATCHES}"
            else:
                from model.model import load_model_from_json
                model = load_model_from_json(name, load=False)
                pathways_layer = model.get_layer(model.name + '_pathways')
                if frac:
                    d = f"{pathways_layer.num_patches_per_path}/{pathways_layer.num_patches}"
                else:
                    d = pathways_layer.num_patches_per_path / pathways_layer.num_patches
            mapping[name] = d
            return d


from evaluation.utils import save_evaluation_json, load_evaluation_json, get_evaluation_time

def load_classfications_by_regex(model_regex, name_to_d_naive=False, convert_name=True, negative_regex=None, print_err=True):
    base_path = 'models'
    archive = {}
    for m in os.listdir(base_path):
        if re.match(model_regex, m) and (negative_regex is None or not re.match(negative_regex, m)):
            try:
                eval_json = load_evaluation_json(m)
            except Exception as err:
                if print_err: print(f"Couldn't load {m} because of {err.__str__()}")
                continue
            if eval_json is not None:
                if convert_name:
                    m = name_to_d(m, naive=name_to_d_naive)
                archive[m] = eval_json
    return archive


def metric_to_label(metric):
    return {"logistic": r"$f^{ensemble}_{logistic}$",
            "ensemble_linear_ArgmaxMeanProb": r"$f^{ensemble}_{mean prob}$"}.get(metric, metric)


def plot_metrics_along_d(model_regex, metric_regex=("logistic", '.*linear_.*', '.*knn.*'), name_to_d_naive=False, measures=False):
    archive = load_classfications_by_regex(model_regex, name_to_d_naive=name_to_d_naive)
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
                        label=metric_to_label(classi) + [' train', ' test'][i])
        plt.axhline(0.4, linestyle=":", c='g', label=BASELINE_NAME)
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


def plot_pathways_distribution_over_d(model_regex, name_to_d_naive=False, plot_train=False):
    archive = load_classfications_by_regex(model_regex, name_to_d_naive=name_to_d_naive)
    plt.figure(figsize=(8, 3))
    ax = None
    for subplot, name in enumerate(("linear", "knn")):
        best = max([max([v[-1] for v in val.values()]) for val in archive.values()])
        pathways_res = {
            d: np.stack([archive[d].get(k, [np.nan] * 2) for k in archive[d] if k.startswith('pathway') and name in k],
                        axis=-1)
            for d in archive if any([k.startswith('pathway') and name in k for k in archive[d]])}

        image_pathways_res = {d: np.stack(
            [archive[d].get(k, [np.nan] * 2) for k in archive[d] if k.startswith('image_pathway') and name in k],
            axis=-1)
                              for d in archive if
                              any([k.startswith('image_pathway') and name in k for k in archive[d]])}

        sorted_keys = np.array([k for k in sorted(pathways_res.keys(), key=lambda k: eval(k))])
        eval_keys = np.array([eval(k) for k in sorted_keys])
        keys_labels = [r'$\frac{{{0}}}{{{1}}}$'.format(*k.split('/')) for k in sorted_keys]

        ax = plt.subplot(1, 2, subplot + 1, sharey=ax)
        plt.title(
            f"{model_regex} Pathways accuracy with {name}")  # + f", d and accuracy pearson's r={corr_test.statistic:.2f} (p={corr_test.pvalue:.2e})")
        for i, (d, k) in enumerate(zip(eval_keys, sorted_keys)):
            for ind in range(2):
                if not ind and not plot_train: continue
                plt.scatter(np.full_like(pathways_res[k][ind], d), pathways_res[k][ind], c=f'C{ind}',
                            alpha=0.5, label=r'$f^{pathway_i}_{logistic}$ ' + ['train', 'test'][ind] if not i else None)
                plt.plot(d + np.array([-0.025, 0.025]), [pathways_res[k][ind].mean()] * 2, c=f'C{ind}')
                if k in image_pathways_res:
                    plt.scatter(np.full_like(image_pathways_res[k][ind], d), image_pathways_res[k][ind],
                                c=f'C{ind + 2}',
                                alpha=0.5,
                                label=r'$f^{mask_i}_{logistic}$ ' + ['train', 'test'][ind] if not i else None)
                    plt.plot(d + np.array([-0.025, 0.025]), [image_pathways_res[k][ind].mean()] * 2, c=f'C{ind + 2}')
        plt.axhline(0.4, c='gray', linestyle=':', label=BASELINE_NAME)
        plt.grid(alpha=0.2)
        plt.xlabel("d")
        plt.ylabel("accuracy")
        plt.xticks(eval_keys, keys_labels)
        simpleaxis(ax)
        plot_utils.legend(facecolor='white', framealpha=0)
        plt.xlim(0, 1)

    plt.tight_layout()


def gather_train_test_res(model_regex, convert_name_to_d=False):
    import re
    base_path = 'models'
    archive_loss = {}
    archive_acc = {}
    for m in os.listdir(base_path):
        if not re.match(model_regex, m):
            continue
        try:
            if convert_name_to_d:
                key = name_to_d(m, naive=False)
            else:
                key = m
        except ValueError:
            continue
        except AttributeError:
            continue
        from utils.io_utils import load_output
        output = load_output(m)
        if output:
            res = {'train': [], 'val': []}
            res_acc = {'train': [], 'val': []}
            for l in output:
                for val in (True, False):
                    if '[==============================]' in l and 'ETA' not in l:
                        matches = re.search(f'{"val_" * val}naive_[\w\d_\.]+embedding_loss: (\d+\.\d+)', l)
                        if matches:
                            res['val' if val else 'train'].append(float(matches.group(1)))

                        matches = re.search(f'{"val_" * val}naive_[\w\d_\.]+accuracy: (\d+\.\d+)', l)
                        if matches:
                            res_acc['val' if val else 'train'].append(float(matches.group(1)))
            if res['train']:
                archive_loss[key] = {k: v for k, v in res.items()}
            if res_acc['train']:
                archive_acc[key] = {k: v for k, v in res_acc.items()}
        else:
            print(m)

    return archive_loss, archive_acc


def plot_train_test(model_regex, convert_name_to_d=False, conversion_f=lambda x: x):
    archive_loss, archive_acc = gather_train_test_res(model_regex, convert_name_to_d)
    for name, dct in zip(('loss', 'accuracy'), (archive_loss, archive_acc)):
        plt.figure(figsize=(8, 5))
        plt.title(model_regex + f" {name}")
        if convert_name_to_d:
            sorted_keys = np.array([k for k in sorted(dct.keys(), key=lambda k: eval(k))])
            eval_keys = np.array([eval(k) for k in sorted_keys])
            keys_labels = [r'$\frac{{{0}}}{{{1}}}$'.format(*k.split('/')) for k in sorted_keys]
        else:
            sorted_keys = sorted(dct.keys())
        colors = [f"C{i}" for i in np.arange(len(dct))]  # plt.cm.magma(eval_keys) if convert_name_to_d else
        for i, d in enumerate(sorted_keys):
            v = dct[d]
            label = keys_labels[i] if convert_name_to_d else conversion_f(d)
            modulu = 10

            def get_x_y(arr):
                arr = np.array(arr)
                n = len(arr)
                sub_arr = arr[:n - (n % modulu)]
                y = sub_arr.reshape((-1, modulu)).mean(axis=-1)
                x = np.arange(len(y)) * modulu + modulu / 2
                if n % modulu:
                    x = np.concatenate([x, [n - modulu / 2]])
                    y = np.concatenate([y, [arr[-modulu:].mean()]])
                return x, y

            plt.plot(*get_x_y(v['train']), c=colors[i], label=label)
            plt.plot(*get_x_y(v['val']), linestyle=':', c=colors[i])
        plt.xlabel("epoch")
        plt.ylabel("embedding " + name)
        if name == 'loss': plt.yscale("log")
        plt.legend(loc=('lower' if name == 'accuracy' else 'upper') + ' right',
                   ncol=len(sorted_keys) // 2 + len(sorted_keys) % 2)
        plt.grid()


def check_correlation_regex(model_regex, metric_regex, cutoff=6/64, name_to_d_naive=False):
    archive = load_classfications_by_regex(model_regex, name_to_d_naive=name_to_d_naive)
    sorted_keys = np.array([k for k in sorted(archive.keys(), key=lambda k: eval(k))])
    eval_keys = np.array([eval(k) for k in sorted_keys])
    mask_before = eval_keys <= cutoff
    mask_after = eval_keys >= cutoff
    for metric in set(classi for dct in archive.values() for classi in dct if any([re.match(regex, classi) for regex in metric_regex])):
        for mask, name in [(mask_before, "before"), (mask_after, "after")]:
            r_test = scipy.stats.pearsonr(eval_keys[mask], [archive[key].get(metric, [np.nan]*2)[-1] for key in sorted_keys[mask]])
            print(metric,  name + f" cutoff {cutoff}, ", f", r={r_test.statistic:.3f}", "p=", f"{r_test.pvalue:.2e}, ", f"n={mask.sum()}")


def check_correlation_regex_dist(model_regex, cutoff=6 / 64, name_to_d_naive=False):
    archive = load_classfications_by_regex(model_regex, name_to_d_naive=name_to_d_naive)
    pathways_res = {
        d: np.stack([archive[d].get(k, [np.nan] * 2)[-1] for k in archive[d] if k.startswith('pathway')], axis=-1)
        for d in archive if any([k.startswith('pathway') for k in archive[d]])}
    x = np.concatenate([[eval(d)] * len(pathways_res[d]) for d in pathways_res])
    y = np.concatenate([pathways_res[d] for d in pathways_res])

    mask_before = x <= cutoff
    mask_after = x >= cutoff
    for mask, name in [(mask_before, "before"), (mask_after, "after")]:
        r_test = scipy.stats.pearsonr(x[mask], y[mask])
        print(name + f" cutoff {cutoff}, ", f", r={r_test.statistic:.3f}", "p=", f"{r_test.pvalue:.2e}, ",
              f"n={mask.sum()}")


def all_plot_regex(model_regex, ensemble_types=("logistic", '.*linear_.*', '.*knn.*'), plot_pathway_train=False):
    archive = plot_metrics_along_d(model_regex, metric_regex=ensemble_types)
    plot_pathways_distribution_over_d(model_regex, plot_train=plot_pathway_train)


from utils.data import Data
from evaluation.evaluation import classify_head_eval
from evaluation.utils import load_evaluation_json, save_evaluation_json


def plot_pathways_vs_masked_images(model_name, num_pathways=10):
    eval_dct = load_evaluation_json(model_name)
    reg_image_res = []
    reg_pathway_res = []

    for pathway_num in range(num_pathways):
        cur_name = f"image_pathway{pathway_num}_linear"
        reg_image_res.append(eval_dct[cur_name])
        reg_pathway_res.append(eval_dct[f"pathway{pathway_num}_linear"])

    from utils.plot_utils import basic_scatterplot, set_ticks_style, simpleaxis, legend
    fig = plt.figure(figsize=(4, 2.5))
    plt.suptitle(model_name + " pathways accuracy\n\n")
    ax = plt.subplot(1, 1, 1)
    for i in range(2):
        basic_scatterplot(np.array(reg_image_res)[..., i], np.array(reg_pathway_res)[..., i],
                          t=not bool(i), fig=fig, label=["train", "test"][i], c=f"C{i}", identity=i)
    full_image_baseline = 0.4
    plt.axvline(full_image_baseline, color='gray', alpha=0.2, label=BASELINE_NAME)
    plt.axhline(full_image_baseline, color='gray', alpha=0.2)
    plt.xlabel("accuracy of " + r"$f^{mask_i}_{logistic}$")
    plt.ylabel("accuracy of " + r"$f^{pathway_i}_{logistic}$")
    #     plt.xlabel("Acc. from masked image")
    #     plt.ylabel("Acc. from pathway embedding")
    legend(loc='lower right')
    simpleaxis(ax)
    plt.show()


def smooth(arr, window=10):
    arr = np.array(arr)
    n = len(arr)
    sub_arr = arr[:n-(n % window)]
    y = sub_arr.reshape((-1, window)).mean(axis=-1)
    x = np.arange(len(y)) * window + (window-1)/2
    if n % window:
        x = np.concatenate([x, [n - window/2]])
        y = np.concatenate([y, [arr[-window:].mean()]])
    return x, y


def regex_models(regex):
    return list(filter(lambda m: re.match(regex, m), os.listdir("models")))


def plot_history(model_regex, window=10, name_to_name=lambda m: m, log=True, keys=None,
                 log_keys={'embedding'}, plot_train=True, plot_val=True, name_to_c=None, save=None):
    from utils.tf_utils import load_history
    models = regex_models(model_regex)
    models_names = []
    orig_names = []
    histories = {}
    for model in models:
        history = load_history(model)
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


def sorted_barplot(model_regex, metric_regex, sort_by_train=True, show_top=None, regex_suptitle=True,
                   negative_regex=None, print_err=False, baseline=None, plot_train=True):
    archive = load_classfications_by_regex(model_regex, convert_name=False, negative_regex=negative_regex,
                                           print_err=print_err)
    n_models = len(archive)
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
        plt.yticks(np.arange(len(sorted_k)), sorted_k)
        if baseline:
            plt.axvline(load_evaluation_json(BASELINE_UNTRAINED)[metric][1], c='k', linestyle=':')
        plt.legend()
        if baseline:
            plt.xticks(np.linspace(0, 1, 11))
            plt.xlim(load_evaluation_json(BASELINE_UNTRAINED)[metric][1], 1)
        plt.grid(axis='x', zorder=0, alpha=0.2, linewidth=2)
    if len(metrics) > 1: plt.tight_layout()


from json import JSONDecodeError
from itertools import product


def gather_results_over_all_args(model_format, name='logistic', seeds=[1], args=dict(P=PS, d=EXTENDED_DS), measure=False):
    names = list(args.keys())
    args = [args[n] for n in names]
    shape = [len(arg) for arg in args]
    res = None # np.full(list(shape) + [len(seeds), 2 - measure], np.nan)

    for i, inds in enumerate(product(*[range(s) for s in shape])):
        for s, seed in enumerate(seeds):
            model_name = model_format.format(
                **{k: v for k, v in zip(names, [args[arg_ind][cur_ind] for arg_ind, cur_ind in enumerate(inds)])},
                seed=seed)
            try:
                dct = load_measures_json(model_name) if measure else load_evaluation_json(model_name)
            except JSONDecodeError as err:
                print(model_name)
            if dct is None:
                val = np.nan
            #                     print(f"missing {model_format.format(P=P, d=d, seed=seed)}")
            else:
                val = dct.get(name, np.nan)
            if val is np.nan:
                print(f"val is nan: {model_name}")
            if res is None:
                res = np.full(list(shape) + [len(seeds)] + ([2] if not measure else (list(val.shape) if isinstance(val, np.ndarray) else [2])), np.nan)
            cur = res
            for cur_ind in inds:
                cur = cur[cur_ind]
            cur[s] = val
    return res


def gather_results_over_all_args_pathways_mean(model_format, name_format='pathway{}_linear', seeds=[1],
                                               args=dict(P=PS, d=EXTENDED_DS), P=10, measure=False):
    names = list(args.keys())
    args = [args[n] for n in names]
    shape = [len(arg) for arg in args]
    res = None # np.full(list(shape) + [len(seeds), 2], np.nan)

    for i, inds in enumerate(product(*[range(s) for s in shape])):
        for s, seed in enumerate(seeds):
            model_name = model_format.format(
                **{k: v for k, v in zip(names, [args[arg_ind][cur_ind] for arg_ind, cur_ind in enumerate(inds)])},
                seed=seed)
            dct = load_measures_json(model_name) if measure else load_evaluation_json(model_name)
            val = np.mean(
                [dct[name_format.format(path)] for path in range(args[names.index("P")] if "P" in names else P)],
                axis=0)
            if res is None:
                res = np.full(list(shape) + [len(seeds)] + ([2] if not measure else list(val.shape)), np.nan)
            cur = res
            for cur_ind in inds:
                cur = cur[cur_ind]
            cur[s] = val
    return res


def plot_mesh_accuracy(res, name=r'$f^{ensemble}_{logistic}$', only_mesh=True, xlabel=r"$d$", ylabel=r"$P$",
                       xticks=EXTENDED_DS, xticks_names=EXTENDED_DS_LABELS, yticks=PS, yticks_names=None):
    suptitle = name + f" as a function of {xlabel} and {ylabel}"

    for which_kind in range(2-only_mesh):
        plt.figure()
        plt.suptitle(suptitle)
        for i in range(2):
            plt.subplot(2,1,i+1)
            plt.title(["train", "test"][i])
            if which_kind:
                plt.scatter(*[arr.flatten() for arr in np.meshgrid(xticks, yticks)], c=res[..., i].flatten(), vmin=res.min(), vmax=res.max())
            else:
                plt.pcolormesh(xticks, yticks, res[..., i], shading='gouraud', vmin=res.min(), vmax=res.max())
            plt.colorbar().set_label("acc")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(xticks, xticks_names)
            plt.yticks(yticks, yticks_names)
        plt.tight_layout()


def plot_lines_different_along_d(model_format, seeds=SEEDS, name="logistic", save=False, measure=False, mask=None,
                                 args=PS, arg=None, mean=False, legend=True, fig=None, c_shift=0, ds=EXTENDED_DS, **kwargs):
    if isinstance(args, str):
        args = eval(args)
    if isinstance(ds, str):
        ds = eval(ds)
    res = (gather_results_over_all_args if not mean else gather_results_over_all_args_pathways_mean)(model_format, name, seeds=seeds,
                                                                                                     args={arg: args, 'd': ds} if arg else {'d': ds},
                                                                                                     measure=measure,
                                                                                                     **kwargs)
    if np.isnan(res).all():
        print(f"for {model_format} and {name}, everything is nan, so not plotting")
        return
    # means = np.nanmean(res, axis=2)
    # stds = np.nanstd(res, axis=2, ddof=1)
    # CI = stats.norm.interval(0.975, loc=means, scale=stds / np.sqrt(np.sum(~np.isnan(res), axis=2)))
    ax = None
    fig = plt.figure() if fig is None else fig
    plt.suptitle(model_format + " " + name + f" different {arg}")
    for i in range(2):
        if measure and not i: continue
        ax = plt.subplot(2-measure,1,i+1-measure, sharey=ax)
        plt.title(["Train", "Test"][i])
        if arg:
            for ind, identity in enumerate(args):
                relevant_part = res[ind, ..., i] if not measure else np.stack([np.stack([res[ind, i_d, s][mask if mask is not None else ~np.eye(res.shape[-1], dtype=bool)]
                                                                                         for s in range(res.shape[2])], axis=0)
                                                                               for i_d in range(len(res[ind]))], axis=0)
                mean = np.nanmean(relevant_part, axis=(-2, -1))
                CI = stats.norm.interval(0.975, loc=mean, scale=np.nanstd(relevant_part, ddof=1, axis=(-2, -1)) / np.sqrt(np.sum(~np.isnan(relevant_part), axis=(-2, -1))))
                plt.plot(ds, mean, label=(legend + ' ' if isinstance(legend, str) else "") + str(identity), c=f"C{ind+c_shift}")
                if len(seeds) > 1:
                    plt.fill_between(ds, CI[0], CI[1][ind, ..., i], color=f"C{ind+c_shift}", alpha=0.3)
        else:
            relevant_part = res[..., i] if not measure else np.stack([np.stack([res[i_d, s][mask if mask is not None else ~np.eye(res.shape[-1], dtype=bool)]
                                                                                for s in range(res.shape[1])], axis=0)
                                                                      for i_d in range(len(res))], axis=0)
            over_axes = (-2, -1) if measure else (-1, )
            mean = np.nanmean(relevant_part, axis=over_axes)
            CI = stats.norm.interval(0.975, loc=mean, scale=np.nanstd(relevant_part, ddof=1, axis=over_axes) / np.sqrt(
                np.sum(~np.isnan(relevant_part), axis=over_axes)))

            plt.plot(ds, mean, label=(legend + ' ') if isinstance(legend, str) else "",
                     c=f"C{c_shift}")
            if len(seeds) > 1:
                plt.fill_between(ds, CI[0], CI[1], color=f"C{c_shift}", alpha=0.3)
        if i:
            plt.xlabel('d')
        else:
            if legend:
                plt.legend()
        plt.ylabel("Accuracy") if not measure else plt.ylabel(name)
        plt.xticks(ds, ds_to_labels(ds))
        plt.grid(alpha=0.3)
        if not measure:
            plt.axhline(0.4, linestyle=':', c='k')
    plt.tight_layout()
    if save:
        savefig(f"figures/{model_format}_along_d_{arg}")
    return fig


def plot_positional_encoding(model, cosine=True, save=False):
    from model.model import load_model_from_json
    if isinstance(model, str):
        model = load_model_from_json(model, optimizer_state=False, skip_mismatch=True)
    embd = model.get_layer(model.name + '_patchenc').position_embedding
    class_token = model.get_layer(model.name + '_patchenc').class_token.numpy()
    num_patches = model.get_layer(model.name + '_pathways').num_patches
    n_pathways = model.get_layer(model.name + "_pathways").n
    indices_shape = [int(np.sqrt(num_patches))] * 2
    patches_enc = embd(np.arange(num_patches)).numpy()
    plt.figure(figsize=(6, 6))
    plt.suptitle(f"Model {model.name} Cosine similarity of positional patch encoding")

    masks = np.full([indices_shape[0] * indices_shape[1], n_pathways], False)
    for pathway in range(n_pathways):
        inds = model.get_layer(model.name + '_pathways').indices.numpy()[:, pathway] - model.get_layer(
            model.name + '_pathways').shift
        masks[inds, pathway] = True
    full_mask = masks.any(axis=-1)

    for i in range(num_patches):
        if full_mask[i]:
            plt.subplot(indices_shape[0], indices_shape[1] + 1, i + 1 + i // indices_shape[1])
            if cosine:
                cos_sim = cosine_sim(patches_enc[i], patches_enc, axis=-1).reshape(indices_shape)
                plt.imshow(np.where(~full_mask.reshape(indices_shape), np.nan, cos_sim), vmin=-1, vmax=1,
                           cmap='seismic')
            else:
                dist = np.einsum('j,ij->i', patches_enc[i], patches_enc).reshape(indices_shape)
                plt.imshow(np.where(~full_mask.reshape(indices_shape), np.nan, dist), cmap='magma')

            #                 dist = np.sqrt(np.power(patches_enc[i, None] - patches_enc, 2).sum(axis=-1)).reshape(indices_shape)
            #                 plt.imshow(np.where(~full_mask.reshape(indices_shape), np.nan, dist), cmap='magma')

            plt.xticks([])
            plt.yticks([])
    if cosine:
        ax = plt.subplot(1, indices_shape[1] + 1, indices_shape[1] + 1)
        plt.imshow(np.tile(np.linspace(-1, 1, 1001)[..., None], [1, 50]), cmap='seismic', origin='lower')
        plt.yticks([0, 1000], [-1, 1])
        plt.xticks([])
        plt.ylabel("cosine similarity")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    plt.tight_layout()
    if save:
        savefig(f"figures/positional_encoding_{model.name}")


def plot_measures(model_regex, mask=None, save=False):
    from measures.utils import CrossPathMeasures, load_measures_json
    models = regex_models(model_regex)
    dcts = {model: load_measures_json(model) for model in models}
    for measure in CrossPathMeasures:
        plt.figure()
        plt.title(measure)
        dct_to_multiviolin({k: v[measure.name][mask if mask is not None else ~np.eye(len(v), dtype=bool)]
                            for k, v in dcts.items()})
        if save:
            savefig(f"figures/{measure}")


def compare_measures(*models, names=None, log=False, mask=None, grid=True, **kwargs):
    from measures.utils import CrossPathMeasures, load_measures_json
    if names is None:
        names = models
    fig = plt.figure(figsize=(len(models) * 3, 5))
    for i, k in enumerate(CrossPathMeasures):
        plt.subplot(3, 3, i+1)
        plt.title("log "*log +  k.name)
        remove = lambda arr: np.where(np.eye(len(arr)) > 0, np.nan, arr) if mask is None else np.where(mask, np.nan, arr)
        f_log = lambda arr: np.log(arr) if log else arr
        dct_to_multiviolin({name: f_log(remove(load_measures_json(model)[k.name]))
                            for name, model in zip(names, models)}, fig=fig, **kwargs)
        if grid:
            plt.grid()
    plt.tight_layout()
