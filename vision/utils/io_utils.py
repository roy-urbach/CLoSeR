import json
import os


def load_json(fn, base_path='config/'):
    if not fn.endswith('json'):
        fn = fn + '.json'
    fn = os.path.join(base_path, fn)
    assert os.path.exists(fn), f"{fn} doesn't exist"
    with open(fn, 'r') as f:
        dct = json.load(f)
    return dct


def save_json(fn, dct, base_path='config/', indent=4, **kwargs):
    if not fn.endswith('json'):
        fn = fn + '.json'
    fn = os.path.join(base_path, fn)
    with open(fn, 'w') as f:
        json.dump(dct, f, indent=indent, **kwargs)
    print(f"saved as {fn}")
    return fn
