"""Solve for aberrations on the optical axis given some truth MTF values and lens parameters."""
import os
import time
from multiprocessing import Pool
from itertools import product

import numpy as np

from scipy.optimize import minimize

from prysm.thinlens import image_displacement_to_defocus

from iris.core import optfcn, prepare_globals
from iris.utilities import parse_cost_by_iter_lbfgsb
from iris.forcefully_redirect_stdout import forcefully_redirect_stdout


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
        list of arrays that are truth values for w040, w060, w080

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


def sph_from_focusdiverse_axial_mtf(sys_parameters, truth_dataframe, codex, guess=(0, 0, 0, 0), parallel=False):
    """Retrieve spherical aberration-related coefficients from axial MTF data.

    Parameters
    ----------
    sys_parameters : `dict`
        dictionary with keys efl, fno, wavelength, samples, focus_planes, focus_range_waves, freqs, freq_step
    truth_dataframe : `pandas.DataFrame`
        a dataframe containing truth values
    codex : dict
        dictionary of key, value pairs where keys are ints and values are strings.  Maps parameter
        numbers to zernike numbers, e.g. {0: 'Z1', 1: 'Z9'} maps (10, 11) to {'Z1': 10, 'Z9': 11}
    guess : iterable, optional
        guess coefficients for the wavefront
    parallel : `bool`, optional
        whether to run optimization in parallel.  Defaults to true

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
    # declare some state for this run as global variables to speed up access in multiprocess pool
    setup_parameters = sys_parameters
    (focus_diversity,
     ax_t, ax_s) = grab_axial_data(setup_parameters, truth_dataframe)

    # casting ndarray to list makes it a list of arrays where the first index
    # is the focal plane and the second frequency.
    t_true, s_true = list(ax_t), list(ax_s)

    _globals = {
        't_true': t_true,
        's_true': s_true,
        'setup_parameters': setup_parameters,
        'decoder_ring': codex,
    }
    if parallel is True:
        pool = Pool(processes=os.cpu_count() - 1, initializer=prepare_globals, initargs=[_globals])
    else:
        pool = None
    prepare_globals({**_globals, 'pool': pool})
    optimizer_function = optfcn
    parameter_vectors = []

    def callback(x):
        parameter_vectors.append(x)

    try:
        parameter_vectors.insert(0, np.asarray(guess))
        t_start = time.perf_counter()
        # do the optimization and capture the per-iteration information from stdout
        with forcefully_redirect_stdout() as out:
            result = minimize(
                fun=optimizer_function,
                x0=guess,
                method='L-BFGS-B',
                options={
                    'disp': True,
                    'gtol': 1e-5,
                    'ftol': 1e-5,

                },
                callback=callback)

        t_end = time.perf_counter()
        # grab the extra data and put everything on the optimizationresult
        cost_by_iter = parse_cost_by_iter_lbfgsb(out['txt'])
        result.x_iter = parameter_vectors
        result.fun_iter = cost_by_iter
        result.time = t_end - t_start
        return result
    finally:
        if pool is not None:
            pool.close()
            pool.join()
