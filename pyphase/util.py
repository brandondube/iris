''' Misc. utilities
'''
import re


def mtf_cost_fcn(true_tan, true_sag, sim_tan, sim_sag):
    ''' A cost function that compares a measured or simulated T/S MTF to a
        simulated one.
    '''
    t = ((true_tan - sim_tan) ** 2).sum()
    s = ((true_sag - sim_sag) ** 2).sum()
    return t + s


def round_to_int(value, integer):
    return integer * round(float(value) / integer)


def parse_cost_by_iter_lbfgsb(string):
    # use regex to find lines that include "at iteration" information.
    lines_with_cost_function_values = \
        re.findall(r'At iterate\s*\d*?\s*f=\s*-*?\d*.\d*D[+-]\d*', string)

    # grab the value from each line and convert to python floats
    fortran_values = [s.split()[-1] for s in lines_with_cost_function_values]
    # fortran uses "D" to denote double and "raw" exp notation,
    # fortran value 3.0000000D+02 is equivalent to
    # python value  3.0000000E+02 with double precision
    return [float(s.replace('D', 'E')) for s in fortran_values]
