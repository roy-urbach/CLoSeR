from utils.tf_utils import *
from utils.io_utils import *
import sys


def printd(*args, **kwargs):
    print(*args, flush=True, **kwargs)
    sys.stdout.flush()
