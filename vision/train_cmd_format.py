from train import parse
import os

BASE_PATH = 'models'
# QUEUE_GPU = "gsla-gpu"    # TODO: change back when it works
QUEUE_GPU = 'sch-gpu'
QUEUE_CPU = 'gsla-cpu'
CONDA_PATH = '~/miniconda3/etc/profile.d/conda.sh'
VENV_NAME = 'tf-gpu'


# def run_command(cmd):
#     if isinstance(cmd, str):
#         cmd = cmd.split()
#     print(f"Calling", ' '.join(cmd))
#     import subprocess
#     result = subprocess.run(cmd, stdout=subprocess.PIPE)
#     return result.stdout.decode('utf-8')


def get_cmd():
    args, bsub_args = parse()
    model_name = args.json.split('.json')[0]
    path = os.path.join(BASE_PATH, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    output_name = os.path.join(path, 'output')
    error_name = os.path.join(path, 'error')

    bsub_call = f'bsub -q {QUEUE_GPU} -J {model_name} -o {output_name}.o -e {error_name}.e -C 1'
    train_call = f'python3 train.py -b {args.batch} -e {args.epochs} --json {args.json}'
    cmd = [*bsub_call.split(), *bsub_args, f'"{train_call}"']
    return ' '.join(cmd)


if __name__ == '__main__':
    # Call this script using the bash function:
    # function train_vis() { python3 train_cmd_format.py $@; cmd=$(<'/tmp/cmd'); eval $cmd; rm /tmp/cmd;}

    with open('/tmp/cmd', 'w') as f:
        f.write(get_cmd())
