import sys
from tqdm import tqdm as counter
import numpy as np
import scipy
import os
import time
import errno


def printd(*args, **kwargs):
    """
    Print with flush
    """
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


def flatten_but_batch(arr):
    """
    Remove all dimensions but the first
    """
    return arr.reshape(len(arr), -1)


def cosine_sim(vec1, vec2, axis=-1):
    """
    Cosine similarity
    :param vec1:
    :param vec2:
    :param axis: axis to reduce over
    :return: the cosine similarity of vec1 and vec2
    """
    normalize = lambda vec: vec / np.linalg.norm(vec, axis=axis, keepdims=True)
    return (normalize(vec1) * normalize(vec2)).sum(axis=axis)


def correlation(arr1, arr2):
    """
    Pearson correlation
    :param arr1: a tensor
    :param arr2: a tensor with the same shape as arr1
    :return: a single float
    """
    return (np.nanmean(np.multiply(arr1, arr2)) - np.nanmean(arr1) * np.nanmean(arr2)) / (np.nanstd(arr1) * np.nanstd(arr2))


def paired_t_test(x, y, **kwargs):
    """
    Calculates the p-value of a paired student's t-test of the two distributions.
    Ignores NaNs
    :param x: distribution 1
    :param y: distribution 2
    :param kwargs: scipy.stats.ttest_rel kwargs
    :return: p-value
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    return scipy.stats.ttest_rel(x[mask], y[mask], **kwargs).pvalue


def ind_t_test(x, y, **kwargs):
    """
    Calculates the p-value of an independent student's t-test of the two distributions.
    Ignores NaNs
    :param x: distribution 1
    :param y: distribution 2
    :param kwargs: scipy.stats.ttest_rel kwargs
    :return: p-value
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    return scipy.stats.ttest_ind(x[mask], y[mask], **kwargs).pvalue


def ind_permutation_test(arr1, arr2, n_permutations=10000, **kwargs):
    """
    Calculates the p-value of an independent permutation test of the two distributions,
    where the statistic is the difference of the means.
    Ignores NaNs
    :param arr1: distribution 1
    :param arr2: distribution 2
    :param n_permutations: number of permutations. Defaults to 10000
    :param kwargs: scipy.stats.permutation_test kwargs
    :return: p-value
    """
    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    result = scipy.stats.permutation_test((arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)]), statistic,
                                          n_resamples=n_permutations, **kwargs)
    return result.pvalue


def paired_permutation_test(arr1, arr2, n_permutations=10000, **kwargs):
    """
    Calculates the p-value of a paired permutation test of the two distributions,
    where the statistic is the mean of the differences.
    Ignores NaNs
    :param arr1: distribution 1
    :param arr2: distribution 2
    :param n_permutations: number of permutations. Defaults to 10000
    :param kwargs: scipy.stats.permutation_test kwargs
    :return: p-value
    """
    def statistic(x, y):
        return np.mean(x - y)
    assert arr1.size == arr2.size, "sizes should be the same"
    mask = np.isnan(arr1) | np.isnan(arr2)

    result = scipy.stats.permutation_test((arr1[~mask], arr2[~mask]), statistic,
                                          n_resamples=n_permutations, **kwargs)
    return result.pvalue


def get_min_max(*arrs):
    """
    :param arrs: a general number of arrays
    :return: (min, max) over all the arrays
    """
    return min([np.nanmin(arr) for arr in arrs]), max([np.nanmax(arr) for arr in arrs])


def streval(w, warning=False):
    """
    Given an input, if it's a string, tries to eval. Otherwise, returns the input
    """
    if isinstance(w, str):
        try:
            return eval(w)
        except Exception as err:
            if warning:
                print(f"couldn't eval {w}")
                import warnings
                warnings.warn(f"couldn't eval {w}")
                return w
            else:
                raise err
    else:
        return w


def unknown_args_to_dict(args, warning=False):
    """
    Takes args from argparse, and turns them to dictionary, by removing "--" and streval the values
    """
    return {args[2*i].split("--")[-1]: streval(args[2*i+1], warning=warning) for i in range(len(args)//2)}


def run_on_dict(dct, f):
    """
    Runs a function on the values of a dict. If dct is not a dict, just returns f(dct).
    :param dct: a dictionary
    :param f: a function to run on the values of the dictionaries
    :return: {k: f(v)}
    """
    if isinstance(dct, dict):
        return {k: f(v) for k, v in dct.items()}
    else:
        return f(dct)
