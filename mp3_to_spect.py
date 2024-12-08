import argparse
from utils.modules import Modules
import librosa
import numpy as np
import os
import tqdm

from utils.utils import printd
BINS = 84


def parse():
    parser = argparse.ArgumentParser(description='convert a bird folder of mp3 to Morlet spectrogram')
    parser.add_argument('-b', '--bird', type=str, help='name of the bird', required=True)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_known_args()
    return args


def mp3_to_spect(fn, db=True):
    y, sr = librosa.load(fn)
    hop_length = 512

    spectrogram = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C2'), n_bins=BINS)
    spectrogram = np.abs(spectrogram)
    if db:
        spectrogram = librosa.amplitude_to_db(spectrogram)
    return spectrogram


def convert():
    args, kwargs = parse()
    bird = args.bird
    printd(f"running bird {bird}")
    module = Modules.AUDITORY
    train_test = 'test' if args.test else 'train'
    path = f"{module.value}/data/{train_test}_audio/{bird}"
    outpath = f"{module.value}/data/{train_test}_spect/{bird}.npz"
    fns = os.listdir(path)
    printd(f"converting {len(fns)} files")
    spects = []
    for fn in tqdm.tqdm(fns):
        spects.append(mp3_to_spect(os.path.join(path, fn)).flatten())
    printd("done converting")
    printd(f"saving compressed as {outpath}")
    np.savez_compressed(outpath, spectrograms=spects, files=fns)
    print("done")


if __name__ == '__main__':
    convert()
