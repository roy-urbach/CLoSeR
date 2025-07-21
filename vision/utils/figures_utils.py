import numpy as np

BASELINE_NAME = r'$f^{image}_{logistic}$'
PATCHES = 64
EXTENDED_QS = sorted([0.025, 0.04, 0.05, 0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.45, 0.4, 0.55, 0.5, 0.65, 0.6, 0.75, 0.7, 0.85, 0.8, 0.95, 0.9,])
PS = (2,3,5,10,15)
PS = np.array(PS)
SEEDS = list(range(1,5))


def qs_to_labels(ds):
    qs_labels = [f"{int(d*PATCHES)}/{PATCHES}" for d in ds]
    qs_labels = [r'$\frac{{{0}}}{{{1}}}$'.format(*k.split('/')) for k in qs_labels]
    return qs_labels


def qs_to_perc(ds):
    return np.array([f'{int(d*PATCHES)/PATCHES:.1%}' for d in ds])

EXTENDED_QS_LABELS_FRAC = qs_to_labels(EXTENDED_QS)
EXTENDED_QS_LABELS = qs_to_perc(EXTENDED_QS)
