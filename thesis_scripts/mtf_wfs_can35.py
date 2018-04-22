from pathlib import Path

import pandas as pd

from prysm.io import read_trioptics_mtf
from prysm.mtf_utils import mtf_ts_to_dataframe
from prysm.macros import SimulationConfig

from iris.recipes import opt_routine_basinhopping, opt_routine_lbfgsb
from iris.rings import W3

#root = Path(__file__).parent / 'data'

# read data files and merge into a single dataframe to feed the algorithm
root = Path(r'C:\Users\brand\Desktop\bd-wfs-realdata\data\MTF')
chunks = root.glob('*.mht')
mtfs = [read_trioptics_mtf(chunk) for chunk in chunks]
dfs = []
for mtf in mtfs:
    t, s = mtf['tan'], mtf['sag']
    focus, freqs = mtf['focus'], mtf['freq']
    df = mtf_ts_to_dataframe(t, s, freqs, 0, focus)
    dfs.append(df)

df = pd.concat(dfs)

# adjust focus to zero mean and units of microns
df.Focus -= df.Focus.mean()  # zero out center value
df.Focus *= 1e3              # mm to um

# EFL and EPD manually entered here from alternative data sources
# produce a SimulationConfig; the focus_range_waves value will be ignored, but the focus zernike and focus_normed
# values will not be
efl = 34.449
epd = 16.9549  # 16.95489884
cfg = SimulationConfig(
    efl=efl,
    fno=efl/epd,
    wvl=0.546,
    samples=128,
    freqs=mtfs[0]['freq'],
    focus_range_waves=0,
    focus_zernike=True,
    focus_normed=True,
    focus_planes=21)

# fire in the hole
if __name__ == '__main__':
    guess = [0] * 18
    guess[5] = -0.075
    guess[0] = 0.05
    # result = opt_routine_lbfgsb(cfg, df, W3, guess, 1e-5, parallel=False)
    result = opt_routine_basinhopping(cfg, df, W3, guess=guess, parallel=False, nthreads=3, max_starts=5)

    import pickle
    with open('res.pkl', 'wb') as fid:
        pickle.dump(result, fid)
