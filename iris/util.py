"""Misc. utilities."""
import re
import io

import numpy as np

from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement


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


def grab_axial_data(setup_parameters, truth_dataframe):
    """Pull axial through-focus MTF data from a pandas DataFrame.

    Parameters
    ----------
    setup_parameters : dict
        dictionary with keys `fno` and `wavelength`
    truth_dataframe : pandas.DataFrame
        Dataframe with columns `Field`, `Focus`, `Azimuth`, and `MTF`.

    Returns
    -------
    wvfront_defocus : numpy.ndarray
        array of defocus values in waves zero to peak.
    ax_t : numpy.ndarray
        array of tangential MTF values.
    ax_s : numpy.ndarray
        array of sagittal MTF values.

    """
    s = setup_parameters
    axial_mtf_data = truth_dataframe[truth_dataframe.Field == 0]
    focuspos = np.unique(axial_mtf_data.Focus.as_matrix())
    wvfront_defocus = image_displacement_to_defocus(focuspos, s['fno'], s['wavelength'])
    ax_t = []
    ax_s = []
    for pos in focuspos:
        fd = axial_mtf_data[axial_mtf_data.Focus == pos]
        ax_t.append(fd[fd.Azimuth == 'Tan']['MTF'].as_matrix())
        ax_s.append(fd[fd.Azimuth == 'Sag']['MTF'].as_matrix())

    wvfront_defocus = np.asarray(wvfront_defocus)
    ax_t = np.asarray(ax_t)
    ax_s = np.asarray(ax_s)
    return wvfront_defocus, ax_t, ax_s


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
