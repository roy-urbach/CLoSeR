import argparse
import os

from utils.modules import Modules

BASE_PATH = 'models'
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
    parser.add_argument('--rusage', type=int, default=RUSAGE, help='CPU mem')
    parser.add_argument('--mem', type=int, default=4, help='GPU mem')

    args = parser.parse_known_args()
    return args


def get_cmd():
    args, bsub_args = parse()
    if args.json.endswith('.json'):
        model_name = args.json[:-len('.json')]
    else:
        model_name = args.json
    path = os.path.join(BASE_PATH, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, "checkpoints")):
        os.mkdir(os.path.join(path, "checkpoints"))
    output_name = os.path.join(path, 'output')
    error_name = os.path.join(path, 'error')

    training_fn = f"{path}/is_training"
    if os.path.exists(training_fn):
        print("already training")
        return f'echo "{model_name} already trained"'

    bsub_call = f'bsub -q {args.queue} -J {model_name} -o {output_name}.o -e {error_name}.e -C 1'
    if 'gpu' in args.queue:
        bsub_call += f" -gpu num=1:j_exclusive=no:gmem={args.mem}GB"
    else:
        bsub_call += f' -R rusage[mem={RUSAGE}]'
    train_call = f'python3 {Modules.get_module(args.module).value}/train.py"'
    train_call += f' -b {args.batch} -e {args.epochs} --json {args.json} -m {args.module}'
    cmd = [*bsub_call.split(), *bsub_args, f'"{train_call}"']
    return ' '.join(cmd)


if __name__ == '__main__':
    # Call this script using the bash function:
    # function train_vis() { python3 train_cmd_format.py $@; cmd=$(<'/tmp/cmd'); eval $cmd; rm /tmp/cmd;}

    with open('/tmp/cmd', 'w') as f:
        f.write(get_cmd())
