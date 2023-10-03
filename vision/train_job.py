from train import parse
import os

BASE_PATH = 'models'
QUEUE_GPU = "gsla-gpu"
QUEUE_CPU = 'gsla-cpu'

def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    import subprocess
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


def run():
    args = parse()
    model_name = args.json.split('.json')[0]
    path = os.path.join(BASE_PATH, model_name)
    os.mkdir(path)
    output_name = os.path.join(path, 'output')
    error_name = os.path.join(path, 'error')

    train_call = f"python3 train.py -b {args.batch} -e {args.epochs} --json {args.json}"

    cmd = f'bsub -q {QUEUE_GPU} -J {model_name} -o {output_name}-%J.o -e {error_name}-%J.e "{train_call}"'

    print(run_command(train_call))


if __name__ == '__main__':
    run()
