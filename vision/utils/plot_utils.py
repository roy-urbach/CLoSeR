from matplotlib import pyplot as plt
from utils.utils import *
import matplotlib as mpl
mpl.rc('image', cmap='gray')


def basic_scatterplot(x, y, identity=True, fig=None, c='k', corr=False, t=False,
                      regress=False, mean_diff=False, label=None, regress_color='r',
                      regress_label=None, add_r_squared_to_regress_label=True, log_x=False, log_y=False):
    x = np.array(x)
    y = np.array(y)

    x_ = np.log(x) if log_x else x
    y_ = np.log(y) if log_y else y

    if fig is None: fig = plt.figure()
    plt.scatter(x, y, c=c, alpha=0.6, label=label)
    if corr:
        plt.title(f"r={correlation(x_, y_):.3f}")
    if t:
        plt.title(f"paired ttest p = {paired_t_test(x_,y_):.2e}" + f', mean diff = {np.abs(np.nanmean(x_-y_)):.3f}'*mean_diff)
    if identity:
        min_, max_ = get_min_max(x, y)
        plt.plot([min_, max_], [min_, max_], c='k', linestyle=':')
    if regress:
        corr = correlation(x_,y_)
        slope = corr * np.std(y_) / np.std(x_)
        reg = lambda p: slope * ((np.log(p) if log_x else p) - np.mean(x_)) + np.mean(y_)
        if log_y:
            reg = lambda p: np.exp(reg(p))
        min_ = np.min(x)
        max_ = np.max(x)
        min_, max_ = min_ - 0.05 * (max_ - min_), max_ + 0.05 * (max_ - min_)
        plt.plot([min_, max_], [reg(min_), reg(max_)], c=regress_color,
                 label=(regress_label + (r"$R^2=$" + f"{corr**2:.2f}") if add_r_squared_to_regress_label else "") if regress_label else None)
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
    return fig


def savefig(fn):
    if not fn.endswith('.png'):
        fn = fn + '.png'
    plt.savefig(fn)
    print(f"saved figure as {fn}")


def violinplot_with_CI(arr, x, c='C0', widths=0.5):
    vi = plt.violinplot(arr, [x], showextrema=False, showmeans=False, widths=widths)
    for pc in vi['bodies']:
        pc.set_facecolor(c)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = arr.size
    CI = np.abs(scipy.stats.t.ppf(0.025, n - 1) * std / np.sqrt(n))
    plt.errorbar([x], mean, yerr=CI, marker='o', capsize=10, c=c)


def multiviolin(arr, xshift=0, xs=None, fig=None, **kwargs):
    if fig is None:
        fig = plt.figure()
    for i in range(len(arr)):
        if (isinstance(arr[i], np.ndarray) and arr[i].size) or (not isinstance(arr[i], np.ndarray) and arr):
            if np.isnan(arr[i]).all(): continue
            violinplot_with_CI(arr[i][~np.isnan(arr[i])], [i + xshift] if xs is None else xs[i], c=f"C{i}", **kwargs)


def dct_to_multiviolin(dct, rotation=0, xs=None, **kwargs):
    keys = list(dct.keys())
    multiviolin([np.array(dct[k]) for k in keys], xs=xs, **kwargs)
    plt.xticks(np.arange(len(keys)) if xs is None else xs, keys, rotation=rotation)


def get_CI(arr, conf=0.975, of_the_mean=True):
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


def simpleaxis(ax, remove_left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if remove_left:
        ax.spines['left'].set_visible(False)
    else:
        ax.get_yaxis().tick_left()
    ax.get_xaxis().tick_bottom()


def set_ticks_style(fig, remove_left=False):
    for ax in fig.axes:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        simpleaxis(ax, remove_left=remove_left)


def legend(*args, facecolor='w', framealpha=0, **kwargs):
    plt.legend(*args, **kwargs, facecolor=facecolor, framealpha=framealpha)


def noticks():
    plt.xticks([])
    plt.yticks([])


def calculate_square_rows_cols(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n/cols))
    return rows, cols
