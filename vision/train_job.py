from train import parse
import os

BASE_PATH = 'models'
# QUEUE_GPU = "gsla-gpu"    # TODO: change back when it works
QUEUE_GPU = 'sch-gpu'
QUEUE_CPU = 'gsla-cpu'


def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"Calling", ' '.join(cmd))
    import subprocess
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


def run():
    args = parse()
    model_name = args.json.split('.json')[0]
    path = os.path.join(BASE_PATH, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    output_name = os.path.join(path, 'output')
    error_name = os.path.join(path, 'error')

    train_call = f'source ~/miniconda3/etc/profile.d/conda.sh; conda activate tf-gpu; python3 train.py -b {args.batch} -e {args.epochs} --json {args.json}'

    cmd = ['bsub', '-q', QUEUE_GPU, '-J', model_name, '-o', output_name + '.o',
           '-e', error_name+'.e', f'"{train_call}"']

    print(run_command(cmd), flush=True)


if __name__ == '__main__':
    run()
