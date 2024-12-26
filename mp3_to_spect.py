import argparse

from auditory.utils.consts import BINS
from utils.modules import Modules
import librosa
import numpy as np
import os
import tqdm

from utils.utils import printd

FMIN = 50
FMAX = 10000


def parse():
    parser = argparse.ArgumentParser(description='convert a bird folder of mp3 to log-melspectrogram')
    parser.add_argument('-b', '--bird', type=str, help='name of the bird', required=True)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_known_args()
    return args


def mp3_to_spect(fn,
                 window_size=1024,
                 hop_size=320,
                 mel_bins=256,
                 fmin=FMIN,
                 fmax=FMAX,
                 window='hann',
                 center=True,
                 pad_mode='reflect'):
    y, sr = librosa.load(fn)
    stft = librosa.core.stft(y,
                             n_fft=window_size,
                             hop_length=hop_size,
                             win_length=window_size,
                             window=window,
                             center=center,
                             pad_mode=pad_mode, )
    spectrogram = np.abs(stft) ** 2
    melw = librosa.filters.mel(sr=sr,
                               n_fft=window_size,
                               n_mels=mel_bins,
                               fmin=fmin, fmax=fmax)
    mel_spect = (melw @ spectrogram)
    log_mel_spect = librosa.core.power_to_db(mel_spect)
    # return log_mel_spect

    # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=BINS, **kwargs)
    # spectrogram = np.log10(spectrogram+eps)
    # if db:
    #     spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_mel_spect


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
