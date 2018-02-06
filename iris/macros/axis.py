"""Macros for performing simulations, etc."""
from prysm.macros import thrufocus_mtf_from_wavefront, SimulationConfig
from prysm.mathops import sqrt, floor

from iris.utilities import make_focus_range_realistic_number_of_microns, prepare_document
from iris.recipes import sph_from_focusdiverse_axial_mtf
from iris.core import config_codex_params_to_pupil


def run_azimuthalzero_simulation(truth=(0, 0.125, 0, 0), guess=(0, 0.0, 0, 0), cfg=None):
    """Run a complete simulation generating and retrieving azimuthal order zero terms.

    Parameters
    ----------
    truth : `tuple`, optional
        truth coefficients, in waves RMS
    guess : `tuple`, optional
        guess coefficients, in waves RMS
    cfg : `prysm.macros.SimulationConfig`, optional
        simulation configuration; if None, use a built in default

    Returns
    -------
    `dict`
        document, see `~iris.prepare_document`

    """
    if cfg is None:
        efl = 50
        fno = 2
        lambda_ = 0.55
        extinction = 1000 / (fno * lambda_)
        freqs = tuple(range(10, floor(extinction), 10))
        cfg = SimulationConfig(
            efl=efl,
            fno=fno,
            wvl=lambda_,
            samples=128,
            freqs=freqs,
            focus_range_waves=1 / 2 * sqrt(3),  # waves / Zernike/Hopkins / norm(Z4)
            focus_zernike=True,
            focus_normed=True,
            focus_planes=21)
        cfg = make_focus_range_realistic_number_of_microns(cfg, 5)
    decoder_ring = {
        0: 'Z4',
        1: 'Z9',
        2: 'Z16',
        3: 'Z25',
    }

    pupil = config_codex_params_to_pupil(cfg, decoder_ring, truth)
    truth_df = thrufocus_mtf_from_wavefront(pupil, cfg)

    sim_result = sph_from_focusdiverse_axial_mtf(cfg, truth_df, decoder_ring, guess)

    residuals = []
    for coefs in sim_result.x_iter:
        p2 = config_codex_params_to_pupil(cfg, decoder_ring, coefs)
        residuals.append((pupil - p2).rms)

    res = prepare_document(
        sim_params=cfg,
        codex=decoder_ring,
        truth_params=truth,
        truth_rmswfe=pupil.rms,
        rmswfe_iter=residuals,
        normed=False,
        optimization_result=sim_result)

    import pickle

    truth_df.to_csv('truth_df.csv', index=False)
    with open('config.pkl', 'wb') as pkl:
        pickle.dump(cfg, pkl)
    with open('truth_wavefront.pkl', 'wb') as pkl:
        pickle.dump(pupil, pkl)
    with open('optimization_result.pkl', 'wb') as pkl:
        pickle.dump(res, pkl)

    return res
