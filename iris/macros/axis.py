"""Macros for performing simulations, etc."""

import numpy as np

from prysm import FringeZernike
from prysm.macros import thrufocus_mtf_from_wavefront

from iris.util import make_focus_range_realistic_number_of_microns, prepare_document
from iris.recipes import sph_from_focusdiverse_axial_mtf


def run_azimuthalzero_simulation(truth=(0, 0.2, 0, 0), guess=(0, 0, 0, 0)):
    """Run a complete simulation generating and retrieving azimuthal order zero terms.

    Parameters
    ----------
    truth : `tuple`, optional
        truth coefficients, in waves RMS
    guess : `tuple`, optional
        guess coefficients, in waves RMS

    Returns
    -------
    `dict`
        document, see `~iris.prepare_document`

    """
    efl = 50
    fno = 2
    lambda_ = 0.55
    extinction = 1000 / (fno * lambda_)
    freqs = np.arange(0, extinction, 10)[1:]  # skip 0
    sim_params = {
        'efl': efl,
        'fno': fno,
        'wavelength': lambda_,
        'samples': 128,
        'focus_planes': 21,  # TODO: see if this many is necessary
        'focus_range_waves': 2,
        'freqs': freqs,
        'freq_step': 10,
    }
    cfg = make_focus_range_realistic_number_of_microns(sim_params, 5)
    efl, fno, wavelength, samples = cfg['efl'], cfg['fno'], cfg['wavelength'], cfg['samples']
    pupil = FringeZernike(Z9=0.2, epd=efl / fno, wavelength=wavelength, samples=samples)
    truth_df = thrufocus_mtf_from_wavefront(pupil, cfg)

    decoder_ring = {
        0: 'Z4',
        1: 'Z9',
        2: 'Z16',
        3: 'Z25',
    }

    sim_result = sph_from_focusdiverse_axial_mtf(cfg, truth_df, decoder_ring, guess)
    return prepare_document(
        sim_params=cfg,
        codex=decoder_ring,
        truth_params=truth,
        truth_rmswfe=pupil.rms,
        normed=False,
        optimization_result=sim_result)
