"""Core optimization routines for wavefront sensing."""
from functools import partial

import numpy as np

from prysm import FringeZernike, MTF
from prysm.mtf_utils import mtf_ts_extractor
from prysm.macros import thrufocus_mtf_from_wavefront_array


def config_codex_params_to_pupil(config, codex, params):
    """Convert a config dictionary, codex dictionary, and parameter vector to a pupil.

    Parameters
    ----------
    config : `dict`
        dictionary with keys efl, fno, wavelength, samples
    codex : `dict`
        dict with integer, string key value pairs, e.g. {0: 'Z1', 1: 'Z9'}
    params : iterable
        sequence of optimization parameters

    Returns
    -------
    `prysm.Pupil`
        a pupil object

    """
    s = config
    pupil_pass_zernikes = {key: value for (key, value) in zip(codex.values(), params)}
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
    t = ((true_tan - sim_tan) ** 2).sum()
    s = ((true_sag - sim_sag) ** 2).sum()
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
    return sum(costfcn) / len(costfcn)  # * setup_parameters.freq_step


def realize_focus_plane(base_wavefront, t_true, s_true, defocus_wavefront):
    """Compute the cost function for a single focal plane.

    Parameters
    ----------
    base_wavefront : `prysm.Pupil`
        a prysm Pupil object
    t_true : `numpy.ndarray`
        array of true MTF values
    s_true : `numpy.ndarray`
        array of true MTF values
    defocus_wavefront : `prysm.Pupil`
        a prysm Pupil object

    Returns
    -------
    `float`
        value of the cost function for this focus plane realization

    """
    global setup_parameters
    prop_wvfront = base_wavefront + defocus_wavefront
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
    global setup_parameters, decoder_ring, pool, t_true, s_true, defocus_pupils
    t, s = np.asarray(t_true).reshape(21, 90), np.asarray(s_true).reshape(21, 90)
    pupil = config_codex_params_to_pupil(setup_parameters, decoder_ring, wavefrontcoefs)

    dat_t, dat_s = thrufocus_mtf_from_wavefront_array(pupil, setup_parameters)
    diff_t = ((t - dat_t) ** 2).sum()
    diff_s = ((s - dat_s) ** 2).sum()

    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].imshow(t, aspect='auto')
    # axs[1].imshow(dat_t, aspect='auto')
    # axs[0].set_title('true')
    # axs[1].set_title('model')
    # axs[1].set(xlabel=f'{diff_t + diff_s}')
    # plt.show()
    return diff_t + diff_s


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
