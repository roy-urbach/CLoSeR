from matplotlib import pyplot as plt
from utils.utils import *
import numpy as np
import matplotlib as mpl
from enum import Enum
from utils.utils import paired_t_test
mpl.rc('image', cmap='gray')

YLABEL_CLASS = "Classification accuracy"
XLABEL_PERC = "fraction of {inp} seen by encoder"
YLABEL_CLASS_F = lambda ax=None: (ax.set_ylabel if ax else plt.ylabel)(YLABEL_CLASS)
XLABEL_FRAC_F = lambda ax=None, inp_name='image': (ax.set_xlabel if ax else plt.xlabel)(XLABEL_PERC.format(inp=inp_name))
XTICKS_FRAC_F = lambda ax=None, ticksize=10: (ax.set_xticks if ax else plt.xticks)(np.linspace(0, 1, 11),
                                                                                   [f"{d:.1f}" if d not in (0, 1) else d
                                                                                    for d in np.linspace(0, 1, 11)],
                                                                                   size=ticksize) and (ax.set_xlim if ax else plt.xlim)(0, 1)


def D_OVER_ACC_F(ax=None, inp_name='image', ticksize=10):
    # XTICKS_FRAC_F(ax=ax, ticksize=ticksize)
    XLABEL_FRAC_F(ax=ax, inp_name=inp_name)
    YLABEL_CLASS_F(ax=ax)
    (ax.set_ylim if ax else plt.ylim)(0, 1)


class NameAndColor:
    def __init__(self, name, c=None, color=None, i=None, **kwargs):
        assert c is not None or color is not None or i is not None
        self.name = name
        self.color = (color or c) if (color or c) is not None else f"C{i}"
        self.c = self.color
        self.i = i
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_name(self):
        return self.name

    def get_color(self):
        return self.color

    def get_c(self):
        return self.get_color()

    def get_i(self):
        return self.i

    def get_kwargs(self):
        return self.kwargs

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class NamesAndColors(Enum):
    CHANCE = NameAndColor("chance level", color='k', linestyle=':')
    MASKED = NameAndColor("masked raw input", i=0, ending='_untrained')
    UNTRAINED = NameAndColor("untrained model", color='chocolate', ending='_untrained')
    SHARED = NameAndColor("shared weights CLoSeR", i=5)
    FULL_INPUT = NameAndColor("full raw input", i=0)
    UNSUPERVISED = NameAndColor("CLoSeR", color='g')
    SUPERVISED = NameAndColor("supervised", color='hotpink', ending='_pathsup')


def basic_scatterplot(x, y, identity=True, fig=None, c='k', corr=False, t=False,
                      regress=False, mean_diff=False, label=None, regress_color='r',
                      regress_label=None, add_r_squared_to_regress_label=True,
                      log_x=False, log_y=False, min_=None, max_=None, alpha=0.6, **kwargs):
    x = np.array(x)
    y = np.array(y)

    mask = np.isnan(x) | np.isnan(y)
    x, y = x[~mask], y[~mask]
    if isinstance(c, np.ndarray) and len(c) == mask.size:
        c = c[~mask]

    x_ = np.log(x) if log_x else x
    y_ = np.log(y) if log_y else y

    if fig is None: fig = plt.figure()
    plt.scatter(x, y, c=c, alpha=alpha, label=label, **kwargs)
    if corr:
        plt.title(f"r={correlation(x_, y_):.3f}")
    if t:
        plt.title(f"paired ttest p = {paired_t_test(x_,y_):.2e}" + f', mean diff = {np.abs(np.nanmean(x_-y_)):.3f}'*mean_diff)
    if identity:
        if min_ is None or max_ is None:
            min_, max_ = get_min_max(x, y)
        plt.plot([min_, max_], [min_, max_], c='k', linestyle=':')
    if regress:
        corr = correlation(x_,y_)
        slope = corr * np.std(y_) / np.std(x_)
        reg = lambda p: slope * ((np.log(p) if log_x else p) - np.mean(x_)) + np.mean(y_)
        if log_y:
            reg = lambda p: np.exp(reg(p))

        if min_ is None or max_ is None:
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


def violinplot_with_CI(arr, x, c='C0', widths=0.5, bar=False, scatter=False, sem=False, horizontal=False, hatch=None,
                       box=False, plot_CI=True, ax=None, **kwargs):
    if bar:
       (plt if ax is None else ax).bar(x, arr.mean(), color=c, **kwargs)
    else:
        if horizontal:
            kwargs['orientation'] = 'horizontal'
        if box:
            bplot = plt.boxplot([arr], positions=x, patch_artist=True, medianprops=dict(color='k', linewidth=2.5), **kwargs)
            for pc in bplot['boxes']:
                pc.set_facecolor(c)
                if hatch:
                    pc.set_hatch(hatch)
        else:
            vi = (plt if ax is None else ax).violinplot(arr, [x], showextrema=False, showmeans=False, widths=widths,
                                **kwargs)
            for pc in vi['bodies']:
                pc.set_facecolor(c)
                if hatch:
                    pc.set_hatch(hatch)
    if plot_CI:
        mean = arr.mean()
        std = arr.std(ddof=1)
        n = arr.size
        SEM = std / np.sqrt(n)
        if sem:
            CI = SEM
        else:
            CI = np.abs(scipy.stats.t.ppf(0.025, n - 1) * SEM)
        err_kwargs = dict(marker='o', capsize=10, c=c)
        if horizontal:
            (plt if ax is None else ax).errorbar(mean, [x], xerr=CI, **err_kwargs)
        else:
            (plt if ax is None else ax).errorbar([x], mean, yerr=CI, **err_kwargs)
    if scatter:
        if horizontal:
            (plt if ax is None else ax).scatter(arr, np.full_like(arr, x), alpha=kwargs.get("alpha", 0.8), c=c)
        else:
            (plt if ax is None else ax).scatter(np.full_like(arr, x), arr, alpha=kwargs.get("alpha", 0.8), c=c)


def multiviolin(arr, xshift=0, xs=None, fig=None, c=None, hatch=None, ax=None, **kwargs):
    if fig is None and ax is None:
        fig = plt.figure()
    for i in range(len(arr)):
        if (isinstance(arr[i], np.ndarray) and arr[i].size) or (not isinstance(arr[i], np.ndarray) and arr):
            if np.isnan(arr[i]).all(): continue
            violinplot_with_CI(arr[i][~np.isnan(arr[i])], [i + xshift] if xs is None else xs[i],
                               c=f"C{i}" if c is None else c[i] if isinstance(c, list) else c,
                               hatch=hatch[i] if hatch else None, ax=ax,
                               **kwargs)


def dct_to_multiviolin(dct, rotation=0, xs=None, horizontal=False, keys=None, c=None, hatch=None, ax=None, **kwargs):
    keys = list(dct.keys()) if keys is None else keys
    if c is None and isinstance(keys[0], NameAndColor):
        c = [k.get_color() for k in keys]
    elif c and isinstance(c, dict):
        c = [c[k] for k in keys]
    if isinstance(hatch, dict):
        hatch = [hatch[k] for k in keys]
    multiviolin([np.array(dct[k]) for k in keys], xs=xs, c=c, horizontal=horizontal, hatch=hatch, ax=ax, **kwargs)
    ticks_to_change = ((plt.yticks if ax is None else ax.set_yticks) if horizontal else (plt.xticks if ax is None else ax.set_xticks))
    ticks_to_change(np.arange(len(keys)) if xs is None else xs, keys, rotation=rotation)


def get_CI(arr, conf=0.975, of_the_mean=True, axis=0, sem=False):
    std = np.nanstd(arr, axis=axis, ddof=1)
    n = (~np.isnan(arr)).sum(axis=axis)
    t = 1 if sem else scipy.stats.t.ppf(conf, df=n - 1)
    ci = std * t / (np.sqrt(n) if of_the_mean else 1)
    return ci


def plot_with_CI(vals, x=None, label=None, fill_between=True, err=True, conf=0.975, axis=0, sem=False, c=None, **kwargs):
    m = np.nanmean(vals, axis=axis)
    ci = get_CI(vals, conf=conf, axis=axis, sem=sem)
    if x is None:
        x = np.arange(m.size)
    mask = ~np.isnan(m)
    if fill_between:
        plt.fill_between(x[mask], m[mask] - ci[mask], m[mask] + ci[mask], alpha=0.3, color=c, **kwargs)
    if err:
        plt.errorbar(x[mask], m[mask], yerr=ci[mask], label=label, alpha=0.8, capsize=10 if not fill_between else 0,
                     c=c, **kwargs)
    else:
        plt.plot(x[mask], m[mask], alpha=0.8, label=label, c=c)


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


def colorbar(mappable, ax=None):
    # https://joseph-long.com/writing/colorbars/
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes if ax is None else ax
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_CDF(arr, xlabel=None, **kwargs):
    plt.plot(np.sort(arr), np.linspace(0, 1, arr.size), **kwargs)
    plt.ylabel("CDF")
    if xlabel:
        plt.xlabel(xlabel)


def pvalue_to_str(p):
    string = ''
    if p < 0.05:
        string += '∗'
    if p < 0.01:
        string += '∗'
    if p < 1e-3:
        string += '∗'
    return string


def plot_significance(x1, x2, y, p, dist=0.1, dist_from_edges=0, line_dist=None, ax=None, horizontal=False, size=None, **kwargs):
    string = pvalue_to_str(p)
    if ax is None:
        ax = plt
    if string:
        if not horizontal:
            ax.plot([x1 + (dist_from_edges if x1 < x2 else -dist_from_edges),
                     x2 + (-dist_from_edges if x1<x2 else dist_from_edges)],
                    [y]*2, c='k', **kwargs)
            ax.plot([x1 + (dist_from_edges if x1 < x2 else -dist_from_edges)]*2, [y-dist, y], c='k', **kwargs)
            ax.plot([x2 + (-dist_from_edges if x1<x2 else dist_from_edges)]*2, [y-dist, y], c='k', **kwargs)
            ax.text((x1+x2)/2, y + dist, string, ha='center', va='center', size=size)
        else:
            if line_dist is None:
                line_dist = dist
            ax.plot([y]*2, [x1 + (dist_from_edges if x1<x2 else -dist_from_edges),
                            x2 + (-dist_from_edges if x1<x2 else dist_from_edges)], c='k', **kwargs)
            ax.plot([y-dist, y], [x1 + (dist_from_edges if x1<x2 else -dist_from_edges)]*2, c='k', **kwargs)
            ax.plot([y-dist, y], [x2 + (-dist_from_edges if x1<x2 else dist_from_edges)]*2, c='k', **kwargs)
            y_text = np.arange(len(string))
            y_text = (y_text - y_text.mean()) * line_dist
            for s, y_text_letter in zip(string, y_text):
                ax.text(y + dist, (x1 + x2) / 2 + y_text_letter, s, ha='center', va='center', size=size)


def plot_significance_anchor(dct, k, keys=None, test=paired_t_test, significance_dist=0.0075, dist=0.002, horizontal=False,
                             x=None, q=None, **kwargs):
    if keys is None:
        keys = list(dct.keys())
    ind_k = keys.index(k)
    if x is not None and isinstance(x, dict):
        x = [x[k] for k in keys]
    if x is not None and not hasattr(x, "__iter__"):
        x = [x]*len(dct)
    for i, alter in enumerate(keys):
        if alter != k:
            p = test(dct[k], dct[alter], alternative='greater')
            if horizontal:
                plot_significance(np.nanmean(dct[k]), np.nanmean(dct[alter]),
                                  (i if x is None else x[i])+significance_dist*np.abs(ind_k- i),
                                  dist=dist, p=p, linewidth=1, horizontal=True, **kwargs)
            else:
                if significance_dist > 0:
                    edge = max(np.nanmax(dct[k]) if q is None else np.quantile(dct[k], q),
                               np.nanmax(dct[alter]) if q is None else np.quantile(dct[alter], q))
                else:
                    edge = min(np.nanmin(dct[k]), np.nanmin(dct[alter]))

                plot_significance(i if x is None else x[i], ind_k if x is None else x[ind_k],
                                  edge+significance_dist*np.abs(ind_k- i), dist=dist, p=p, linewidth=1, **kwargs)


def remove_spines(ax, top_right=False):
    for sp in ax.spines:
        if not top_right or sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)


def legend_linewidth(ax, linewidth=2, **kwargs):
    leg = ax.legend(**kwargs)
    for lobj in leg.get_lines():
        lobj.set_linewidth(linewidth)
    return leg
