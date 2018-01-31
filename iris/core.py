"""Core optimization routines for wavefront sensing."""
from functools import partial
from prysm import FringeZernike, MTF
from prysm.mtf_utils import mtf_ts_extractor


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
    return sum(costfcn) / len(costfcn) * setup_parameters['freq_step']


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
    mtf = MTF.from_pupil(prop_wvfront, setup_parameters['efl'])
    t, s = mtf_ts_extractor(mtf, setup_parameters['freqs'])
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
    s = setup_parameters
    efl, fno, wavelength, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    pupil_pass_zernikes = {key: value for (key, value) in zip(decoder_ring.values(), wavefrontcoefs)}
    pupil = FringeZernike(**pupil_pass_zernikes, base=1,
                          epd=efl / fno, wavelength=wavelength, samples=samples)

    # for each focus plane, compute the cost function
    if pool is not None:
        rfp_mp = partial(realize_focus_plane, pupil)
        costfcn = pool.starmap(rfp_mp, zip(t_true, s_true, defocus_pupils))
    else:
        costfcn = []
        for t, s, defocus in zip(t_true, s_true, defocus_pupils):
            costfcn.append(realize_focus_plane(pupil, t, s, defocus))
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
