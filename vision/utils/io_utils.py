import json
import os


def load_json(fn, base_path='config/'):
    if not fn.endswith('json'):
        fn = fn + '.json'
    assert os.path.exists(fn), f"{fn} doesn't exist"
    return json.loads(os.path.join(base_path, fn))
