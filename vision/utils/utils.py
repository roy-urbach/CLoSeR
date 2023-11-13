import sys
from tqdm import tqdm as counter
import numpy as np
import scipy


def printd(*args, **kwargs):
    print(*args, flush=True, **kwargs)
    sys.stdout.flush()


def get_class(cls, file):
    if isinstance(cls, str):
        classes = [getattr(file, c) for c in dir(file) if isinstance(getattr(file, c), type) and c==cls]
        if not classes:
            raise ValueError(f"{file.__name__}.{cls} doesn't exist")
        else:
            cls = classes[0]
    else:
        assert isinstance(cls, type)
    return cls


def cosine_sim(vec1, vec2, axis=-1):
    normalize = lambda vec: vec / np.linalg.norm(vec, axis=axis, keepdims=True)
    return (normalize(vec1) * normalize(vec2)).sum(axis=axis)


def correlation(arr1, arr2):
    return (np.nanmean(np.multiply(arr1, arr2)) - np.nanmean(arr1) * np.nanmean(arr2)) / (np.nanstd(arr1) * np.nanstd(arr2))


def correlation_t_test(corr, n, a=0.95):
    t_cutoff = scipy.stats.t.ppf(a, df=n-2)
    t = corr * np.sqrt(n-2) / (1 - corr**2)
    p = 1 - scipy.stats.t.cdf(t, df=n - 2)
    return t, t_cutoff, p


def paired_t_test(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    return scipy.stats.ttest_rel(x[mask], y[mask]).pvalue


def ind_t_test(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    return scipy.stats.ttest_ind(x[mask], y[mask]).pvalue


def get_min_max(*arrs):
    return min([np.nanmin(arr) for arr in arrs]), max([np.nanmax(arr) for arr in arrs])
