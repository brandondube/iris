''' Misc. utilities
'''
import numpy as np
import pandas as pd

from prysm.util import share_fig_ax, correct_gamma

parameters = {
    'efl': 50,
    'fno': 2,
    'wavelength': 0.5,
    'samples': 128
}

def mtf_cost_fcn(true_tan, true_sag, sim_tan, sim_sag):
    ''' A cost function that compares a measured or simulated T/S MTF to a
        simulated one.
    '''
    t = ((true_tan-sim_tan)**2).sum()
    s = ((true_sag-sim_sag)**2).sum()
    return t+s
