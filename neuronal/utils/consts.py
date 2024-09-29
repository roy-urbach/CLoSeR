from utils.consts import MODELS_DIR, CONFIG_DIR

NEURONAL_BASE_DIR = "neuronal"

NEURONAL_MODELS_DIR = f"{NEURONAL_BASE_DIR}/{MODELS_DIR}"
NEURONAL_CONFIG_DIR = f"{NEURONAL_BASE_DIR}/{CONFIG_DIR}"

MODULE_MODELS_DIR = NEURONAL_MODELS_DIR
MODULE_CONFIG_DIR = NEURONAL_CONFIG_DIR

AREAS_DICT = {'cortex': ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal', 'VISmma', 'VISmmp', 'VISli'],
              'thalamus': ['LGd', 'LD', 'LP', 'VPM', 'TH', 'MGm', 'MGv', 'MGd', 'PO', 'LGv', 'VL',
                           'VPL', 'POL', 'Eth', 'PoT', 'PP', 'PIL', 'IntG', 'IGL', 'SGN', 'VPL', 'PF', 'RT'],
              'hippocampus': ['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POST', 'PRE', 'ProS', 'HPF'],
              'midbrain': ['MB', 'SCig', 'SCiw', 'SCsg', 'SCzo', 'PPT', 'APN', 'NOT', 'MRN', 'OP', 'LT', 'RPF', 'CP']}

SESSIONS = (758798717,
            756029989,
            737581020,
            715093703,
            757216464,
            719161530,
            754312389,
            732592105,
            739448407,
            797828357,
            743475441,
            721123822,
            742951821)

INVALID_SESSIONS = (2, 6)
VALID_SESSIONS = set(range(len(SESSIONS))) - set(INVALID_SESSIONS)

NATURAL_MOVIES = ("natural_movie_one", "natural_movie_three")
NATURAL_MOVIES_FRAMES = {NATURAL_MOVIES[0]: 900,
                         NATURAL_MOVIES[1]: 3600}
NATURAL_MOVIES_TRIALS = {NATURAL_MOVIES[0]: 20,
                         NATURAL_MOVIES[1]: 10}
BLOCKS = 2

SESSIONS_NUM_UNITS = {758798717: 593,
                      756029989: 684,
                      737581020: 568,
                      715093703: 884,
                      757216464: 959,
                      719161530: 755,
                      754312389: 502,
                      732592105: 824,
                      739448407: 625,
                      797828357: 611,
                      743475441: 553,
                      721123822: 444,
                      742951821: 893}
