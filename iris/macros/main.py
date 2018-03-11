"""Macros for performing simulations, etc."""
from prysm.macros import thrufocus_mtf_from_wavefront, SimulationConfig
from prysm.mathops import floor, sqrt

from iris.utilities import (
    make_focus_range_realistic_number_of_microns,
    prepare_document_local,
    prepare_document_global,
)
from iris.recipes import opt_routine_lbfgsb, opt_routine_basinhopping
from iris.core import config_codex_params_to_pupil
from iris.rings import W1, W2

efl, fno, lambda_ = 50, 2, 0.55
extinction = 1000 / (fno * lambda_)
DEFAULT_CONFIG = SimulationConfig(
    efl=efl,
    fno=fno,
    wvl=lambda_,
    samples=128,
    freqs=tuple(range(10, floor(extinction), 10)),
    focus_range_waves=1 / 2 * sqrt(3),  # waves / Zernike/Hopkins / norm(Z4)
    focus_zernike=True,
    focus_normed=True,
    focus_planes=21)
DEFAULT_CONFIG = make_focus_range_realistic_number_of_microns(DEFAULT_CONFIG, 5)


def run_simulation(truth=(0, 0.125, 0, 0), guess=(0, 0.0, 0, 0), cfg=None, solver='global',
                   decoder_ring=None, solver_opts=None, core_opts=None):
    """Run a complete simulation generating and retrieving azimuthal order zero terms.

    Parameters
    ----------
    truth : `tuple`, optional
        truth coefficients, in waves RMS
    guess : `tuple`, optional
        guess coefficients, in waves RMS
    cfg : `prysm.macros.SimulationConfig`, optional
        simulation configuration; if None, use a built in default
    solver : `str`, optional
        whether to use a local or a global optimizer
    decoder_ring : `dict`, optional
        a decoder ring, a dictionary that looks like {0: 'Z1', 1: 'Z2' ...}, if None defaults to
        W1 from iris/rings.py if guess is of length 4, and W2 if guess is of length 16
    solver_opts : `dict` or None, optional
        kwd:value pairs to pass to solver, if None defaults are chosen by the solver function
    core_opts : `dict` or None, optional
        kwd:value pairs to pass to optimization core, if None defaults chosen by the optimization core

    Returns
    -------
    `dict`
        document, see `~iris.utilities.prepare_document`

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    if decoder_ring is None:
        if len(guess) == 16:
            decoder_ring = W2
        else:
            decoder_ring = W1

    if solver.lower() == 'local':
        solver, prepare_document, flag = opt_routine_lbfgsb, prepare_document_local, 'local'
    else:
        solver, prepare_document, flag = opt_routine_basinhopping, prepare_document_global, 'global'

    pupil = config_codex_params_to_pupil(cfg, decoder_ring, truth)
    truth_df = thrufocus_mtf_from_wavefront(pupil, cfg)
    if solver_opts is not None and core_opts is not None:
        sim_result = solver(cfg, truth_df, decoder_ring, guess, **{**solver_opts, 'core_opts': core_opts})
    elif solver_opts is not None:
        sim_result = solver(cfg, truth_df, decoder_ring, guess, **solver_opts)
    elif core_opts is not None:
        sim_result = solver(cfg, truth_df, decoder_ring, guess, **{'core_opts': core_opts})
    else:
        sim_result = solver(cfg, truth_df, decoder_ring, guess)

    if flag == 'local':
        residuals = []
        for coefs in sim_result.x_iter:
            p2 = config_codex_params_to_pupil(cfg, decoder_ring, coefs)
            residuals.append((pupil - p2).rms)
    else:
        residuals = [[]]
        for idx, iteration in enumerate(sim_result.x_iter):
            for coefs in iteration:
                p2 = config_codex_params_to_pupil(cfg, decoder_ring, coefs)
                residuals[idx].append((pupil - p2).rms)
            residuals.append([])
        del residuals[-1]  # clean up trailing list

    res = prepare_document(
        sim_params=cfg,
        codex=decoder_ring,
        truth_params=truth,
        truth_rmswfe=pupil.rms,
        rmswfe_iter=residuals,
        normed=True,
        optimization_result=sim_result)
    return res
