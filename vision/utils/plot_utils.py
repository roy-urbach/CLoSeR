from matplotlib import pyplot as plt
from utils.utils import *


def basic_scatterplot(x, y, identity=True, fig=None, c='k', corr=False, t=False, regress=False, mean_diff=False):
    x = np.array(x)
    y = np.array(y)
    if fig is None: fig = plt.figure()
    plt.scatter(x, y, c=c, alpha=0.6)
    if corr:
        plt.title(f"r={correlation(x,y):.3f}")
    if t:
        plt.title(f"paired ttest p = {paired_t_test(x,y):.2e}" + f', mean diff = {np.abs(np.nanmean(x-y)):.3f}'*mean_diff)
    if identity:
        min_, max_ = get_min_max(x, y)
        plt.plot([min_, max_], [min_, max_], c='k', linestyle=':')
    if regress:
        reg = lambda p: correlation(x,y) * np.std(y) / np.std(x) * (p - np.mean(x)) + np.mean(y)
        min_ = np.min(x)
        max_ = np.max(x)
        min_, max_ = min_ - 0.05 * (max_ - min_), max_ + 0.05 * (max_ - min_)
        plt.plot([min_, max_], [reg(min_), reg(max_)], c='r')
    return fig


def savefig(fn):
    if not fn.endswith('.png'):
        fn = fn + '.png'
    plt.savefig(fn)
    print(f"saved figure as {fn}")


def violinplot_with_CI(arr, x, c='C0'):
    vi = plt.violinplot(arr, [x], showextrema=False, showmeans=False)
    for pc in vi['bodies']:
        pc.set_facecolor(c)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = arr.size
    CI = scipy.stats.t.ppf(0.025, n - 1) * std / np.sqrt(n)
    plt.errorbar([x], mean, yerr=CI, marker='o', capsize=10, c=c)


def multiviolin(arr, xshift=0, fig=None):
    if fig is None:
        fig = plt.figure()
    for i in range(len(arr)):
        if (isinstance(arr[i], np.ndarray) and arr[i].size) or (not isinstance(arr[i], np.ndarray) and arr):
            violinplot_with_CI(arr[i][~np.isnan(arr[i])], [i + xshift], c=f"C{i}")


def dct_to_multiviolin(dct, rotation=0, **kwargs):
    keys = list(dct.keys())
    multiviolin([np.array(dct[k]) for k in keys], **kwargs)
    plt.xticks(np.arange(len(keys)), keys, rotation=rotation)


def get_CI(arr, conf=0.975, of_the_mean=True):
    m = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    n = (~np.isnan(arr)).sum(axis=0)
    t = scipy.stats.t.ppf(conf, df=n - 1)
    ci = std * t / (np.sqrt(n) if of_the_mean else 1)
    return ci


def plot_with_CI(vals, x=None, label=None, fill_between=True, err=True, conf=0.975, **kwargs):
    m = np.nanmean(vals, axis=0)
    ci = get_CI(vals, conf=conf)
    if x is None:
        x = np.arange(m.size)
    mask = ~np.isnan(m)
    if fill_between:
        plt.fill_between(x[mask], m[mask] - ci[mask], m[mask] + ci[mask], alpha=0.3, **kwargs)
    if err:
        plt.errorbar(x[mask], m[mask], yerr=ci[mask], label=label, alpha=0.8, capsize=10 if not fill_between else 0,
                     **kwargs)
    else:
        plt.plot(x[mask], m[mask], alpha=0.8, label=label)
