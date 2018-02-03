"""Misc. utilities."""
import re
import io
from operator import itemgetter

import numpy as np

from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement


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


def make_focus_range_realistic_number_of_microns(config, round_focus_to=5):
    """Create a modified config dictionary that has a focus range which is realistic for an MTF bench.

    Parameters
    ----------
    config : `dict`
        dict with keys focus_range_waves, fno, wavelength
    round_focus_to : `int`, optional
        integer number of microns to round focus to

    Returns
    -------
    `dict`
        modified dic that has same keys as input dict

    """
    d = config
    focusdiv_wvs = d['focus_range_waves']
    focusdiv_um = round_to_int(defocus_to_image_displacement(
        focusdiv_wvs,
        d['fno'],
        d['wavelength']), round_focus_to)  # round to nearest x microns
    # copy and update the focus diversity in the config dict
    cfg = config.copy()
    cfg['focus_range_waves'] = image_displacement_to_defocus(
        focusdiv_um,
        d['fno'],
        d['wavelength'])
    return cfg


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


def prepare_document(sim_params, codex, truth_params, truth_rmswfe, normed, optimization_result):
        """Prepare a document (dict) for insertion into the results database.

        Parameters
        ----------
        sim_params : `dict`
            dictionary with keys for efl, fno, wavelength, samples, focus_planes, focus_range_waves, freqs
        codex : `dict`
            dictionary with integer keys in [0, len(truth_params)] and values 'Z1'..'Z49'; each key
                is a position in the parameter vector and each value is the appropriate zernike.
        truth_parms: iterable
            truth parameters
        truth_rmswfe: `float`
            RMS WFE of the truth
        normed : `bool`
            if the coefficients should be made unit RMS value
        optimization_result : object
            object with keys x, x_iter, fun, fun_iter, time

        Returns
        -------
        `dict`
            dictionary with keys, types:
                - sim_params, dict
                - codex, dict
                - truth_params, tuple
                - truth_rmswfe, float
                - zernike_norm, bool
                - result_final, tuple
                - result_iter, list
                - cost_final, float
                - cost_iter, list
                - time, float

        """
        x, xiter, f, fiter, t = itemgetter('x', 'x_iter', 'fun', 'fun_iter', 'time')(optimization_result)
        return {
            'sim_params': sim_params,
            'codex': codex,
            'truth_params': truth_params,
            'truth_rmswfe': truth_rmswfe,
            'zernike_normed': normed,
            'result_final': x,
            'result_iter': xiter,
            'cost_final': f,
            'cost_iter': fiter,
            'time': t,
        }


def pgm_img_to_array(imgstr):
    """Convert a string version of an ASCII pgm file to a numpy array.

    Parameters
    ----------
    imgstr : `str`
        image as a string

    Returns
    -------
    `numpy.ndarray`
        2D ndarray of the image

    Notes
    -----
    Operates under the assumption that the image will be to the specification
    used by the trioptics imagemaster HR; that is, the header shall be 3 lines
    long.

    """
    string = io.StringIO(imgstr)
    return np.loadtxt(string, dtype=np.int16, skiprows=3)
