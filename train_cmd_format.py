import argparse
import os
import re

from utils.modules import Modules

# QUEUE_GPU = "gsla-gpu"    # TODO: change back when it works
# QUEUE_GPU = 'sch-gpu'
# QUEUE_GPU = 'gpu-short'
QUEUE_GPU = 'short-gpu'
QUEUE_CPU = "new-short"
# QUEUE_CPU = 'gsla-cpu'
CONDA_PATH = '~/miniconda3/etc/profile.d/conda.sh'
RUSAGE = 6000


# def run_command(cmd):
#     if isinstance(cmd, str):
#         cmd = cmd.split()
#     print(f"Calling", ' '.join(cmd))
#     import subprocess
#     result = subprocess.run(cmd, stdout=subprocess.PIPE)
#     return result.stdout.decode('utf-8')


def parse():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-j', '--json', type=str, help='name of the config json', required=True)
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('-q', '--queue', type=str, default=QUEUE_GPU, help='name of the queue')
    parser.add_argument('-m', '--module', type=str, default=Modules.VISION.name,
                        choices=Modules.get_cmd_module_options())
    parser.add_argument('-s', '--seed', type=int, default=None, help='seed to change to')
    parser.add_argument('-d', '--masking_ratio', type=float, default=None, help='d to change to')
    # parser.add_argument('--kwargs', type=dict, default={}, help='kwargs to change and save config')
    parser.add_argument('--rusage', type=int, default=RUSAGE, help='CPU mem')
    parser.add_argument('--mem', type=int, default=4, help='GPU mem')

    args = parser.parse_known_args()
    return args


def get_cmd():
    args, bsub_args = parse()
    module = Modules.get_module(args.module)
    if args.json.endswith('.json'):
        model_name = args.json[:-len('.json')]
    else:
        model_name = args.json

    assert module.load_json(model_name, config=True) is not None

    new_config = False
    dct = module.load_json(model_name, config=True)

    pretrained_model_name = dct.get("pretrained_name", None)

    if args.seed is not None:
        new_config = True
        dct['model_kwargs']['pathways_kwargs']['seed'] = args.seed
        model_name = re.sub(r"seed\d+", f"seed{args.seed}", model_name)
        if pretrained_model_name:
            pretrained_model_name = re.sub(r"seed\d+", f"seed{args.seed}", pretrained_model_name)

    if args.masking_ratio is not None:
        new_config = True
        dct['model_kwargs']['pathways_kwargs']['d'] = args.masking_ratio
        model_name = re.sub(r"_d0\.\d+_", f"_d{args.masking_ratio}_", model_name)
        if pretrained_model_name:
            pretrained_model_name = re.sub(r"seed\d+", f"seed{args.seed}", pretrained_model_name)

    if pretrained_model_name:
        dct['pretrained_name'] = pretrained_model_name

    if new_config:
        module.save_json(model_name, dct, config=True)

    path = os.path.join(module.get_models_path(), model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, "checkpoints")):
        os.mkdir(os.path.join(path, "checkpoints"))
    output_name = os.path.join(path, 'output')
    error_name = os.path.join(path, 'error')

    training_fn = os.path.join(path, "is_training")
    if os.path.exists(training_fn):
        print("already training")
        return f'echo "{model_name} already trained"'

    bsub_call = f'bsub -q {args.queue} -J {model_name} -o {output_name}.o -e {error_name}.e -C 1'
    if 'gpu' in args.queue:
        bsub_call += f" -gpu num=1:j_exclusive=no:gmem={args.mem}GB"
    else:
        bsub_call += f' -R rusage[mem={args.rusage}]'
    train_call = f'python3 train.py -b {args.batch} -e {args.epochs} --json {model_name} -m {args.module}'
    cmd = [*bsub_call.split(), *bsub_args, f'"{train_call}"']
    return ' '.join(cmd)


if __name__ == '__main__':
    # Call this script using the bash function:
    # function train_vis() { python3 train_cmd_format.py $@; cmd=$(<'/tmp/cmd'); eval $cmd; rm /tmp/cmd;}
    from utils.io_utils import load_json, save_json

    with open('/tmp/cmd', 'w') as f:
        f.write(get_cmd())
