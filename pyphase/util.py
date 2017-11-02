''' Misc. utilities
'''

import pandas as pd

def mtf_cost_fcn(true_tan, true_sag, sim_tan, sim_sag):
    ''' A cost function that compares a measured or simulated T/S MTF to a
        simulated one.
    '''
    t = ((true_tan-sim_tan)**2).sum()
    s = ((true_sag-sim_sag)**2).sum()
    return t+s

def mtf_ts_extractor(mtf, freqs):
    ''' Extracts the T and S MTF from a PSF object.
    '''
    tan = mtf.exact_polar(freqs=freqs, azimuths=0)
    sag = mtf.exact_polar(freqs=freqs, azimuths=90)
    return tan, sag

def mtf_ts_to_dataframe(tan, sag, freqs, field=0, focus=0):
    ''' Creates a Pandas dataframe from tangential and sagittal MTF data.

    Args:
        tan (`numpy.ndarray`): vector of tangential MTF data.

        sag (`numpy.ndarray`): vector of sagittal MTF data.

        freqs (`iterable`): vector of spatial frequencies for the data.

        field (`float`): relative field associated with the data.

        focus (`float`): focus offset (um) associated with the data.

    Returns:
        pandas dataframe.

    '''
    rows = []
    for f, s, t in zip(freqs, tan, sag):
        base_dict = {
            'Field': field,
            'Focus': focus,
            'Freq': f,
        }
        rows.append({**base_dict, **{
            'Azimuth': 'Tan',
            'MTF': t,
            }})
        rows.append({**base_dict, **{
            'Azimuth': 'Sag',
            'MTF': s,
            }})
    return pd.DataFrame(data=rows)
