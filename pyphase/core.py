from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd

from mystic.solvers import PowellDirectionalSolver
from mystic.termination import VTR

from prysm import FringeZernike, Seidel, PSF, MTF, config
from prysm.thinlens import image_displacement_to_defocus
config.set_precision(32)

from pyphase.util import mtf_ts_extractor, mtf_ts_to_dataframe

def thrufocus_mtf_from_wavefront(focused_wavefront, sim_params, focus_diversity, freqs):
    ''' Creates a thru-focus T/S MTF curve at each frequency requested from a
        focused wavefront.
    '''
    s = sim_params
    focusdiv_wvs = image_displacement_to_defocus(focus_diversity,s['fno'], s['wavelength'])
    dfs = []
    for focus, displacement in zip(focusdiv_wvs, focus_diversity):
        wvfront_defocus = Seidel(W020=focus, samples=s['samples'], epd=s['efl']/s['fno'], wavelength=s['wavelength'])
        psf = PSF.from_pupil(focused_wavefront.merge(wvfront_defocus), efl=s['efl'])
        mtf = MTF.from_psf(psf)
        tan, sag = mtf_ts_extractor(mtf, freqs)
        dfs.append(mtf_ts_to_dataframe(tan, sag, freqs, focus=displacement))

    return pd.concat(dfs)
