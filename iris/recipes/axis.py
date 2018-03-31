"""Solve for aberrations on the optical axis given some truth MTF values and lens parameters."""

import numpy as np

from prysm.thinlens import image_displacement_to_defocus


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

    dat[:, 0] = 0  # set defocus to zero

    return list(dat)


def grab_axial_data(setup_parameters, df):
    """Pull axial through-focus MTF data from a pandas DataFrame.

    Parameters
    ----------
    setup_parameters : dict
        dictionary with keys `fno` and `wavelength`
    df : pandas.DataFrame
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
    df = df.sort_values(by=['Field', 'Focus', 'Azimuth', 'Freq'])
    axial_mtf_data = df[df.Field == 0]
    focuspos = axial_mtf_data.Focus.unique()
    wvfront_defocus = image_displacement_to_defocus(focuspos, s.fno, s.wvl, s.focus_zernike, s.focus_normed)
    ax_t = []
    ax_s = []
    for pos in focuspos:
        fd = axial_mtf_data[axial_mtf_data.Focus == pos]
        ax_t.append(fd[fd.Azimuth == 'Tan']['MTF'].values)
        ax_s.append(fd[fd.Azimuth == 'Sag']['MTF'].values)

    wvfront_defocus = np.asarray(wvfront_defocus)
    ax_t = np.asarray(ax_t)
    ax_s = np.asarray(ax_s)
    return wvfront_defocus, ax_t, ax_s
