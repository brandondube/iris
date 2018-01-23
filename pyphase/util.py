"""Misc. utilities."""
import re


def mtf_cost_fcn(true_tan, true_sag, sim_tan, sim_sag):
    """Cost function that compares a measured or simulated T/S MTF to a simulated one.

    Parameters
    ----------
    true_tan : `numpy.ndarray`
        true tangential MTF values
    true_sag : `numpy.ndarray`
        true sagittal MTF values
    sim_tan : `numpy.ndarray`
        simulated tangential MTF values
    sim_sag : `numpy.ndarray`
        simulated sagittal MTF values

    Returns
    -------
    `float`
        a scalar cost function

    Notes
    -----
    Algorithm is
    sum((M - S)^2) / numel(M)  ~= Mean Square Error (MSE)

    """
    t = ((true_tan - sim_tan) ** 2).sum() / true_tan.size
    s = ((true_sag - sim_sag) ** 2).sum() / true_sag.size
    return t + s


def net_costfcn_reducer(costfcn):
    """Reduces a vector cost function to a single scalar value.

    Parameters
    ----------
    costfcn : `iterable`
        an iterable containing cost functions for different focal planes

    Returns
    -------
    `float`
        dimensionally reduced cost function

    Notes
    -----
        Algorithm is simply a mean, multiplied by 100 to yield a percentage

    """
    return sum(costfcn) / len(costfcn) * 100


def round_to_int(value, integer):
    """Round a value to the nearest integer.

    Parameters
    ----------
    value : `float` or `int`
        a value to round
    integer : `int`
        what int to round to (closest 5, 10, etc)

    Returns
    -------
    `int`
        rounded value

    """
    return integer * round(float(value) / integer)


def parse_cost_by_iter_lbfgsb(string):
    """Parse the cost function history from L-BFGS-B optimizer print statements.

    Parameters
    ----------
    string : `str`
        A string containing the entire L-BFGS-B output

    Returns
    -------
    `list`
        a list of cost function values by iteration

    """
    # use regex to find lines that include "at iteration" information.
    lines_with_cost_function_values = \
        re.findall(r'At iterate\s*\d*?\s*f=\s*-*?\d*.\d*D[+-]\d*', string)

    # grab the value from each line and convert to python floats
    fortran_values = [s.split()[-1] for s in lines_with_cost_function_values]
    # fortran uses "D" to denote double and "raw" exp notation,
    # fortran value 3.0000000D+02 is equivalent to
    # python value  3.0000000E+02 with double precision
    return [float(s.replace('D', 'E')) for s in fortran_values]
