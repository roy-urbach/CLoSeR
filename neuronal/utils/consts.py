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
USED_AREAS = ["VISp", "VISam", "VISrl", "LP", "CA1"]
SESSIONS = (715093703,
            719161530,
            721123822,
            732592105,
            737581020,
            739448407,
            742951821,
            743475441,
            744228101,
            746083955,
            750332458,
            750749662,
            751348571,
            754312389,
            754829445,
            755434585,
            756029989,
            757216464,
            757970808,
            758798717,
            759883607,
            760345702,
            760693773,
            761418226,
            762120172,
            762602078,
            763673393,
            773418906,
            791319847,
            797828357,
            798911424,
            799864342)

PREVIOUS_SESSIONS = (758798717,
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

VALID_SESSIONS = set(range(len(SESSIONS))) - {13, 4}

INVALID_SESSIONS = (2, 6)
VALID_SESSIONS_PREV = set(range(len(PREVIOUS_SESSIONS))) - set([PREVIOUS_SESSIONS[ind] for ind in INVALID_SESSIONS])

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

PSEUDO_MOUSE_NUM_UNITS = {'grey': 1238,
                          'CA1': 644,
                          'VISp': 490,
                          'VISrl': 485,
                          'LP': 481,
                          'VISam': 467,
                          'VISl': 397,
                          'APN': 389,
                          'VISpm': 322,
                          'VISal': 274,
                          'LGd': 261,
                          'VIS': 229,
                          'DG': 217,
                          'SUB': 159,
                          'PO': 147,
                          'CA3': 88,
                          'POL': 73,
                          'SGN': 66,
                          'VPM': 63,
                          'MGd': 47,
                          'VISmma': 42,
                          'NOT': 37,
                          'MB': 29,
                          'TH': 25,
                          'Eth': 22,
                          'LGv': 22,
                          'MGv': 18,
                          'MGm': 16,
                          'SCig': 13,
                          'PIL': 13,
                          'PP': 12,
                          'PPT': 11,
                          'ProS': 8,
                          'VL': 7,
                          'OP': 6,
                          'MRN': 5,
                          'CA2': 1}
