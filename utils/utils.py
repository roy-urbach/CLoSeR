import sys
from tqdm import tqdm as counter
import numpy as np
import scipy
import os
import time
import errno


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


def flatten_but_batch(arr):
    return arr.reshape(len(arr), -1)


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


def paired_t_test(x, y, **kwargs):
    mask = ~(np.isnan(x) | np.isnan(y))
    return scipy.stats.ttest_rel(x[mask], y[mask], **kwargs).pvalue


def ind_t_test(x, y, **kwargs):
    mask = ~(np.isnan(x) | np.isnan(y))
    return scipy.stats.ttest_ind(x[mask], y[mask], **kwargs).pvalue


def get_min_max(*arrs):
    return min([np.nanmin(arr) for arr in arrs]), max([np.nanmax(arr) for arr in arrs])



class FileLockException(Exception):
    pass


class FileLock(object):

    """ A file locking mechanism that has context-manager support so
        you can use it in a with statement. This should be relatively cross
        compatible as it doesn't rely on msvcrt or fcntl for the locking.
        Taken from:
        https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py
    """

    def __init__(self, file_name, timeout=10, delay=.05):
        """ Prepare the file locker. Specify the file to lock and optionally
            the maximum timeout and the delay between each attempt to lock.
        """
        if timeout is not None and delay is None:
            raise ValueError("If timeout is not None, then delay must not be None.")
        self.is_locked = False
        self.lockfile = os.path.join(os.getcwd(), "%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay
        self.fd = None

    def acquire(self):
        """ Acquire the lock, if possible. If the lock is in use, it check again
            every `wait` seconds. It does this until it either gets the lock or
            exceeds `timeout` number of seconds, in which case it throws
            an exception.
        """
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                self.is_locked = True  # moved to ensure tag only when locked
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if self.timeout is None:
                    raise FileLockException("Could not acquire lock on {}".format(self.file_name))
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException("Timeout occured.")
                time.sleep(self.delay)

    #        self.is_locked = True

    def release(self):
        """ Get rid of the lock by deleting the lockfile.
            When working in a `with` statement, this gets automatically
            called at the end.
        """
        if self.is_locked:
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False

    def __enter__(self):
        """ Activated when used in the with statement.
            Should automatically acquire a lock to be used in the with block.
        """
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        """ Activated at the end of the with statement.
            It automatically releases the lock if it isn't locked.
        """
        if self.is_locked:
            self.release()

    def __del__(self):
        """ Make sure that the FileLock instance doesn't leave a lockfile
            lying around.
        """
        self.release()


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


def streval(w, warning=False):
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
    return {args[2*i].split("--")[-1]: streval(args[2*i+1], warning=warning) for i in range(len(args)//2)}


def run_on_dict(dct, f):
    if isinstance(dct, dict):
        return {k: f(v) for k, v in dct.items()}
    else:
        return f(dct)
