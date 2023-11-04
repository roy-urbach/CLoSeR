import sys
from tqdm import tqdm as counter
import numpy as np


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
    new_vec1 = vec1 / np.linalg.norm(vec1, axis=axis)
    new_vec2 = vec2 / np.linalg.norm(vec2, axis=axis)
    return (new_vec1 * new_vec2).sum(axis=axis)
