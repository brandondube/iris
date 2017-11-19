''' Misc. utilities
'''
import numpy as np
import pandas as pd

def mtf_cost_fcn(true_tan, true_sag, sim_tan, sim_sag):
    ''' A cost function that compares a measured or simulated T/S MTF to a
        simulated one.
    '''
    t = ((true_tan-sim_tan)**2).sum()
    s = ((true_sag-sim_sag)**2).sum()
    return t+s

def round_to_int(value, integer):
    return integer * round(float(value)/integer)
