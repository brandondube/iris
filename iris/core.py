"""Core optimization routines for wavefront sensing."""
from functools import partial

from prysm import FringeZernike, MTF
from prysm.mtf_utils import mtf_ts_extractor
from prysm.mathops import sqrt


def config_codex_params_to_pupil(config, codex, params, defocus=0):
    """Convert a config dictionary, codex dictionary, and parameter vector to a pupil.

    Parameters
    ----------
    config : `dict`
        dictionary with keys efl, fno, wavelength, samples
    codex : `dict`
        dict with integer, string key value pairs, e.g. {0: 'Z1', 1: 'Z9'}
    params : iterable
        sequence of optimization parameters
    defocus : float, optional
        amount of defocus applied in the same units as params

    Returns
    -------
    `prysm.Pupil`
        a pupil object

    """
    s = config
    pupil_pass_zernikes = {key: value for (key, value) in zip(codex.values(), params)}
    pupil_pass_zernikes['Z4'] += defocus
    return FringeZernike(**pupil_pass_zernikes,
                         base=1,
                         epd=s.efl / s.fno,
                         wavelength=s.wvl,
                         samples=s.samples,
                         rms_norm=s.focus_normed)


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
    Simply the sum of the square of differences between the model and truth data.

    """
    global diffraction
    difference_t = true_tan - sim_tan
    difference_s = true_sag - sim_sag
    return difference_t, difference_s


def _mtf_cost_core_diffractiondiv(difference_t, difference_s):
    """Adjust the MTF differences by the diffraction limit.

    Parameters
    ----------
    difference_t : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data
    difference_s : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data

    Returns
    -------
    difference_t : `numpy.ndarray`
        adjusted difference of measured and modeled tangential MTF data
    difference_s : `numpy.ndarray`
        adjusted difference of measured and modeled sagittal MTF data

    """
    global diffraction
    return difference_t / diffraction, difference_s / diffraction


def _mtf_cost_core_manhattan(difference_t, difference_s):
    """Adjust the raw difference of MTF to the Manhattan distance.

    Parameters
    ----------
    difference_t : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data
    difference_s : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data

    Returns
    -------
    `float`
        scalar cost function

    Notes
    -----
    see NIST - https://xlinux.nist.gov/dads/HTML/manhattanDistance.html

    """
    t = (abs(difference_t)).sum()
    s = (abs(difference_s)).sum()
    return t + s


def _mtf_cost_core_euclidian(difference_t, difference_s):
    """Adjust the raw difference of MTF to the Euclidian distance.

    Parameters
    ----------
    difference_t : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data
    difference_s : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data

    Returns
    -------
    `float`
        scalar cost function

    Notes
    -----
    see NIST - https://xlinux.nist.gov/dads/HTML/euclidndstnc.html

    """
    t = (sqrt(difference_t ** 2)).sum()
    s = (sqrt(difference_s ** 2)).sum()
    return t + s


def _mtf_cost_core_sumsquarediff(difference_t, difference_s):
    """Adjust the raw difference of MTF to the sum of the square of the differences.

    Parameters
    ----------
    difference_t : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data
    difference_s : `numpy.ndarray`
        raw difference of measured and modeled tangential MTF data

    Returns
    -------
    `float`
        scalar cost function

    """
    t = (difference_t ** 2).sum()
    s = (difference_s ** 2).sum()
    return t + s


def average_mse_focusplanes(costfcn):
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
    Adjustment effectively integrates over frequency axis and normalizes to
    average over focus planes.

    """
    return sum(costfcn) / len(costfcn) * (setup_parameters.freqs[1] - setup_parameters.freqs[0])


def realize_focus_plane(params, t_true, s_true, defocus):
    """Compute the cost function for a single focal plane.

    Parameters
    ----------
    base_wavefront : `prysm.Pupil`
        a prysm Pupil object
    t_true : `numpy.ndarray`
        array of true MTF values
    s_true : `numpy.ndarray`
        array of true MTF values
    defocus_wavefront : float
        amount of defocus, in same units as params

    Returns
    -------
    `float`
        value of the cost function for this focus plane realization

    """
    global setup_parameters, decoder_ring
    prop_wvfront = config_codex_params_to_pupil(setup_parameters, decoder_ring, params, defocus)
    mtf = MTF.from_pupil(prop_wvfront, setup_parameters.efl)
    t, s = mtf_ts_extractor(mtf, setup_parameters.freqs)
    return mtf_cost_fcn(t_true, s_true, t, s)


def optfcn(wavefrontcoefs):
    """Optimization routine used to compare simulation data to measurement data.

    Parameters
    ----------
    wavefrontcoefs : iterable
        a vector of wavefront coefficients

    Returns
    -------
    `float`
        cost function value

    """
    # generate a "base pupil" with some aberration content
    global setup_parameters, decoder_ring, pool, t_true, s_true, defocus

    if pool is not None:
        rfp_mp = partial(realize_focus_plane, wavefrontcoefs)
        costfcn = pool.starmap(rfp_mp, zip(t_true, s_true, defocus))
    else:
        costfcn = []
        for t, s, defocus_ in zip(t_true, s_true, defocus):
            costfcn.append(realize_focus_plane(wavefrontcoefs, t, s, defocus_))

    return average_mse_focusplanes(costfcn)


def prepare_globals(arg_dict):
    """Initialize global variables inside process pool for windows support of shared read-only global state.

    Parameters
    ----------
    arg_dict : `dict`
        dictionary of key/value pairs of variable names and values to expose at the global level

    Notes
    -----
    globals() returns the global object specific to this file; these globals will not be shared with
    other modules.

    """
    globals().update(arg_dict)
