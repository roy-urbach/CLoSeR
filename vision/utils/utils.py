import sys


def printd(*args, **kwargs):
    print(*args, flush=True, **kwargs)
    sys.stdout.flush()
