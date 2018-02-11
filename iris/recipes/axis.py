"""Solve for aberrations on the optical axis given some truth MTF values and lens parameters."""
from itertools import product

import numpy as np

from prysm.thinlens import image_displacement_to_defocus


def generate_axial_truth_coefs(max_val, num_steps, symmetric=True):
    """Generate a cartesian product of W040, W060, and W080.

    Subject to a best focus constraint for minimum RMS wavefront error. Note that for a large
    num_steps, the output will be large; num_steps=10 will produce 1,000 items.

    Parameters
    ----------
    max_val : `int` or `float`
        maximum value of each coefficient
    num_steps : `int`
        number of points between bounds
    symmetric : bool, optional
        whether the bounds are symmetric (negative to positive) or not (0 to positive)

    Returns
    -------
    list
        list of arrays that are truth values for each of Z4, Z9, Z16, Z25

    """
    if symmetric is True:
        lower = -max_val
    else:
        lower = 0

    w040 = np.linspace(lower, max_val, num_steps)
    w060 = np.linspace(lower, max_val, num_steps)
    w080 = np.linspace(lower, max_val, num_steps)

    # take the product
    coefs = list(product(w040, w060, w080))

    # here, figure out what w020 is for best focus and add it to the
    # coefficients
    return coefs


def generate_random_axial_truth_coefs(peak, ncoefs, symmetric=True):
    """Generate random axial truth coefficients.

    Parameters
    ----------
    peak : `float`
        peak value, in waves RMS
    ncoefs : `int`
        number of truths to generate
    symmetric : `bool`, optional
        whether the distribution is symmetric and ranges from (-peak, peak)

    Returns
    -------
    `list`
        a list of coefficient arrays of length ncoefs

    """
    dat = np.random.random((ncoefs, 4))
    if symmetric is True:
        dat -= 0.5
        dat *= (2 * peak)
    else:
        dat *= peak

    return list(dat)


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
    wvfront_defocus = image_displacement_to_defocus(focuspos, s.fno, s.wvl, s.focus_zernike, s.focus_normed)
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
