import argparse
import os

BASE_PATH = 'models'
# QUEUE_GPU = "gsla-gpu"    # TODO: change back when it works
# QUEUE_GPU = 'sch-gpu'
QUEUE_GPU = 'gpu-short'
QUEUE_CPU = "new-short"
# QUEUE_CPU = 'gsla-cpu'
CONDA_PATH = '~/miniconda3/etc/profile.d/conda.sh'
RUSAGE = 75000


# def run_command(cmd):
#     if isinstance(cmd, str):
#         cmd = cmd.split()
#     print(f"Calling", ' '.join(cmd))
#     import subprocess
#     result = subprocess.run(cmd, stdout=subprocess.PIPE)
#     return result.stdout.decode('utf-8')


def parse():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-j', '--json', type=str, help='name of the config json')
    parser.add_argument('-q', '--queue', type=str, default=QUEUE_GPU, help='name of the queue')

    args = parser.parse_known_args()
    return args


def get_cmd():
    args, bsub_args = parse()
    model_name = args.json.split('.json')[0]
    path = os.path.join(BASE_PATH, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, "checkpoints")):
        os.mkdir(os.path.join(path, "checkpoints"))
    output_name = os.path.join(path, 'output')
    error_name = os.path.join(path, 'error')

    bsub_call = f'bsub -q {args.queue} -J {model_name} -o {output_name}.o -e {error_name}.e -C 1'
    if args.queue.startswith('gpu'):
        bsub_call += " -gpu num=1:j_exclusive=no:gmem=32GB"
    else:
        bsub_call += ' -R rusage[mem={RUSAGE}]'
    train_call = f'python3 train.py -b {args.batch} -e {args.epochs} --json {args.json}'
    cmd = [*bsub_call.split(), *bsub_args, f'"{train_call}"']
    return ' '.join(cmd)


if __name__ == '__main__':
    # Call this script using the bash function:
    # function train_vis() { python3 train_cmd_format.py $@; cmd=$(<'/tmp/cmd'); eval $cmd; rm /tmp/cmd;}

    with open('/tmp/cmd', 'w') as f:
        f.write(get_cmd())
