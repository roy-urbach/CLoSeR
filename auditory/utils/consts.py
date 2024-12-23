import numpy as np

from utils.consts import MODELS_DIR, CONFIG_DIR

AUDITORY_BASE_DIR = "auditory"

AUDITORY_MODELS_DIR = f"{AUDITORY_BASE_DIR}/{MODELS_DIR}"
AUDITORY_CONFIG_DIR = f"{AUDITORY_BASE_DIR}/{CONFIG_DIR}"

MODULE_MODELS_DIR = AUDITORY_MODELS_DIR
MODULE_CONFIG_DIR = AUDITORY_CONFIG_DIR


# ALL_BIRDS = np.array(list(filter(lambda bird: os.path.exists(os.path.join(base_path, 'train_spect', f"{bird}.npz")), sorted(os.listdir(os.path.join(base_path, 'train_audio'))))))

ALL_BIRDS = np.array(['aldfly', 'ameavo', 'amebit', 'amecro', 'amegfi', 'amekes',
       'amered', 'amerob', 'amewig', 'amewoo', 'amtspa', 'annhum',
       'astfly', 'baisan', 'baleag', 'balori', 'banswa', 'barswa',
       'bawwar', 'belkin1', 'belspa2', 'bewwre', 'bkbcuc', 'bkbmag1',
       'bkbwar', 'bkcchi', 'bkchum', 'bkhgro', 'bkpwar', 'bktspa',
       'blkpho', 'blugrb1', 'blujay', 'bnhcow', 'boboli', 'bongul',
       'brdowl', 'brebla', 'brespa', 'brncre', 'brnthr', 'brthum',
       'brwhaw', 'btbwar', 'btnwar', 'btywar', 'buffle', 'buggna',
       'buhvir', 'bulori', 'bushti', 'buwtea', 'buwwar', 'cacwre',
       'calgul', 'calqua', 'camwar', 'cangoo', 'canwar', 'canwre',
       'carwre', 'casfin', 'caster1', 'casvir', 'cedwax', 'chispa',
       'chiswi', 'chswar', 'chukar', 'clanut', 'cliswa', 'comgol',
       'comgra', 'comloo', 'commer', 'comnig', 'comrav', 'comred',
       'comter', 'comyel', 'coohaw', 'coshum', 'daejun', 'doccor',
       'dowwoo', 'eargre', 'easblu', 'easkin', 'easmea', 'easpho',
       'eastow', 'eawpew', 'eucdov', 'eursta', 'evegro', 'fiespa',
       'fiscro', 'foxspa', 'gadwal', 'gcrfin', 'gnttow', 'gnwtea',
       'gockin', 'gocspa', 'goleag', 'grbher3', 'grcfly', 'greegr',
       'greroa', 'greyel', 'grhowl', 'grnher', 'grycat', 'gryfly',
       'haiwoo', 'hamfly', 'hergul', 'herthr', 'hoomer', 'hoowar',
       'horgre', 'horlar', 'houfin', 'houspa', 'houwre', 'indbun',
       'juntit1', 'killde', 'labwoo', 'larspa', 'lazbun', 'leabit',
       'leafly', 'leasan', 'lecthr', 'lesgol', 'lesyel', 'lewwoo',
       'linspa', 'lobcur', 'lobdow', 'logshr', 'lotduc', 'louwat',
       'macwar', 'magwar', 'mallar3', 'marwre', 'merlin', 'moublu',
       'mouchi', 'moudov', 'norcar', 'norfli', 'norhar2', 'normoc',
       'norpar', 'norpin', 'norsho', 'norwat', 'nrwswa', 'nutwoo',
       'olsfly', 'orcwar', 'osprey', 'ovenbi1', 'palwar', 'pasfly',
       'pecsan', 'perfal', 'phaino', 'pibgre', 'pilwoo', 'pingro',
       'pinjay', 'pinsis', 'pinwar', 'plsvir', 'prawar', 'purfin',
       'pygnut', 'rebmer', 'rebnut', 'rebsap', 'rebwoo', 'redcro',
       'redhea', 'reevir1', 'renpha', 'reshaw', 'rethaw', 'rewbla',
       'ribgul', 'rinduc', 'robgro', 'rocpig', 'rocwre', 'rthhum',
       'ruckin', 'rudduc', 'rufgro', 'rufhum', 'rusbla', 'sagspa1',
       'sagthr', 'savspa', 'saypho', 'scatan', 'scoori', 'semplo',
       'semsan', 'sheowl', 'shshaw', 'snobun', 'snogoo', 'solsan',
       'sonspa', 'sora', 'sposan', 'spotow', 'stejay', 'swahaw', 'swaspa',
       'swathr', 'treswa', 'truswa', 'tuftit', 'tunswa', 'veery',
       'vesspa', 'vigswa', 'warvir', 'wesblu', 'wesgre', 'weskin',
       'wesmea', 'wessan', 'westan', 'wewpew', 'whbnut', 'whcspa',
       'whfibi', 'whtspa', 'whtswi', 'wilfly', 'wilsni1', 'wiltur',
       'winwre3', 'wlswar', 'wooduc', 'wooscj2', 'woothr', 'y00475',
       'yebfly', 'yebsap', 'yehbla', 'yelwar', 'yerwar', 'yetvir'])

N_FREQS = 256
SR = 22050
BINS = 2018
