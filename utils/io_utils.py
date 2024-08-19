import json
import os

from utils.modules import Modules


@Modules.add_method
def load_json(fn):
    if not fn.endswith('json'):
        fn = fn + '.json'
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            dct = json.load(f)
        return dct
    else:
        return None


@Modules.add_method
def save_json(fn, dct, indent=4, **kwargs):
    if not fn.endswith('json'):
        fn = fn + '.json'
    with open(fn, 'w') as f:
        json.dump(dct, f, indent=indent, **kwargs)
    print(f"saved as {fn}")
    return fn


@Modules.add_method
def load_output(model_dir):
    fn = os.path.join(model_dir, 'output.o')
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        return lines
    else:
        return None


@Modules.add_method
def get_file_time(fn, raw=True):
    if os.path.exists(fn):
        mtime = os.path.getmtime(fn)
        if raw:
            return mtime
        else:
            from time import ctime
            return ctime(mtime)
    else:
        return None


@Modules.add_method
def get_output_time(model_dir, raw=True):
    fn = os.path.join(model_dir, 'output.o')
    return get_file_time(fn, raw=raw)
