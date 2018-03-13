"""Main recipe."""
import time
from multiprocessing import Pool, cpu_count
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize, basinhopping

from iris.core import prepare_globals, optfcn
from iris.forcefully_redirect_stdout import forcefully_redirect_stdout
from iris.utilities import parse_cost_by_iter_lbfgsb, split_lbfgsb_iters
from iris.recipes.axis import grab_axial_data

from prysm.otf import diffraction_limited_mtf

# make a namedtuple that holds optimization setup variables
OptSetup = namedtuple('OptSetup', ['focus_diversity', 't_true', 's_true', 'diffraction'])


def opt_routine_lbfgsb(sys_parameters, truth_dataframe, codex, guess=(0, 0, 0, 0),
                       ftol=1e-2, parallel=False, nthreads=None, core_opts=None):
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
    ftol : `float`
        cost function tolerance
    parallel : `bool`, optional
        whether to run optimization in parallel.  Defaults to true
    nthreads : `int`, optional
        number of threads to use for parallel optimization; if None, defaults to number of logical threads - 1
    core_otps: `tuple` or None, optional
        options to pass to the optimizaiton core

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
    setup_data = prep_data(sys_parameters, truth_dataframe)
    pool = prep_globals(setup_data, sys_parameters, codex, parallel, nthreads)

    parameter_vectors = []

    def callback(x):
        parameter_vectors.append(x.copy())

    if core_opts is None:
        args = (None, None)
    else:
        args = core_opts

    try:
        parameter_vectors.append(np.asarray(guess))
        t_start = time.perf_counter()
        # do the optimization and capture the per-iteration information from stdout
        with forcefully_redirect_stdout() as out:
            result = minimize(
                fun=optfcn,
                x0=guess,
                method='L-BFGS-B',
                options={
                    'disp': True,
                    'ftol': ftol,
                },
                args=args,
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


def opt_routine_basinhopping(sys_parameters, truth_dataframe, codex, guess=(0, 0, 0, 0),
                             ftol=1e-3, max_starts=100, parallel=False, nthreads=None, core_opts=None):
    """Pseudoglobal basin-hopping based optimization routine.

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
    ftol : `float`
        cost function tolerance
    max_starts : `int`
        maximum number of pseudorandom starting guesses to make
    parallel : `bool`, optional
        whether to run optimization in parallel.  Defaults to true
    nthreads : `int`, optional
        number of threads to use for parallel optimization; if None, defaults to number of logical threads - 1
    core_otps: `tuple` or None, optional
        options to pass to the optimizaiton core

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
    # extract data and prepare the global variables
    setup_data = prep_data(sys_parameters, truth_dataframe)
    pool = prep_globals(setup_data, sys_parameters, codex, parallel, nthreads)

    # prepare args (optimization subroutine) for optimizer, defaults if not given
    if core_opts is None:
        args = (None, None)
    else:
        args = core_opts

    # prepare for the logging callbacks
    global nbasinit
    nbasinit = 1
    parameters_certain = [[]]
    parameters_uncertain = [[]]

    # global callback exits optimization is sufficiently low minimum found, prepares iter for local
    # local callback logs parameter vectors
    def cb_global(x, f, accept):
        global nbasinit  # declare nit as global
        if f < ftol:     # if the cost function is small enough, declare success
            return True
        elif nbasinit >= max_starts:  # if there have been the maximum number of starts, stop
            return True
        nbasinit += 1    # if not, increment the counter and make a new parameter history list
        parameters_certain.append([])
        parameters_uncertain.append([])

    def cb_local(x):
        global nbasinit
        parameters_certain[nbasinit - 1].append(x.copy())

    def optwrapper(x, *args):
        global nbasinit
        parameters_uncertain[nbasinit - 1].append(x.copy())
        return optfcn(x, *args)

    try:
        t_start = time.perf_counter()
        # do the optimization and capture the per-iteration information from stdout
        with forcefully_redirect_stdout() as out:
            result = basinhopping(
                func=optwrapper,
                x0=guess,
                minimizer_kwargs={
                    'args': args,
                    'method': 'L-BFGS-B',
                    'options': {
                        'disp': True,
                        'ftol': ftol,
                    },
                    'callback': cb_local,
                },
                callback=cb_global,
                stepsize=0.05,
                T=0.05,  # 0.1 might be more appropriate, try later.
                interval=2,
                seed=1234)

        t_end = time.perf_counter()
        txt = out['txt']

        # extract the cost function history from L-BFGS-B output
        iter_outs = split_lbfgsb_iters(txt)
        cost_iters = [parse_cost_by_iter_lbfgsb(txtbuffer) for txtbuffer in iter_outs]

        # use the manual parameter history to get access to the first iteration of each local
        # minimization attempt
        for pcertain, puncertain in zip(parameters_certain, parameters_uncertain):
            pcertain.insert(0, puncertain[0])

        # finally, because basinhopping does not call the callback after the first iteration,
        # manually split the first iteration of the parameter histories
        # accept that it is too difficult to recover the initial guess of the second iteration
        # due to this splitting action, and set it to the same value as the second iteration
        len_ = len(cost_iters[0])
        iteration_two = parameters_certain[0][len_:]
        iteration_two.insert(0, iteration_two[0])
        parameters_certain.insert(1, iteration_two)
        del parameters_certain[0][len_:]

        # store things on the optimization result object
        result.x_iter = parameters_certain
        result.fun_iter = cost_iters
        result.time = t_end - t_start
        return result
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def prep_data(sys_parameters, truth_df):
    """Extract data needed for optimization from the system parameters and truth data.

    Parameters
    ----------
    sys_parameters : `prysm.macros.SetupParameters`
        a setupparameters namedtuple
    truth_df : `pandas.DataFrame`
        a pandas DF with columns Field, Focus, Azimuth, MTF

    Returns
    -------
    `OptSetup`
        optimization setup namedtuple with focus diversity, t and s truth data, and diffraction data

    """
    (focus_diversity,
     ax_t, ax_s) = grab_axial_data(sys_parameters, truth_df)

    # casting ndarray to list makes it a list of arrays where the first index
    # is the focal plane and the second frequency.
    t_true, s_true = list(ax_t), list(ax_s)

    # now compute diffraction for the setup and use it as a normailzation
    diffraction = diffraction_limited_mtf(sys_parameters.fno, sys_parameters.wvl, frequencies=sys_parameters.freqs)

    return OptSetup(
        focus_diversity=focus_diversity,
        t_true=t_true,
        s_true=s_true,
        diffraction=diffraction)


def prep_globals(setup_data, setup_parameters, codex, parallel, nthreads):
    """Prepare the global variables used in the optimimzation routine.

    Parameters
    ----------
    setup_data : `OptSetup`
        optimization setup namedtuple with focus diversity, t and s truth data, and diffraction data
    setup_parameters : `prysm.macros.SetupParameters`
        a setupparameters namedtuple
    codex : `dict`
        dictionary of key, value pairs where keys are ints and values are strings.  Maps parameter
        numbers to zernike numbers, e.g. {0: 'Z1', 1: 'Z9'} maps (10, 11) to {'Z1': 10, 'Z9': 11}
    parallel : `bool`
        whether the optimization is parallel or not
    nthreads : `int`, optional
        number of threads to use for parallel optimization; if None, defaults to number of logical threads - 1

    Returns
    -------
    pool : `multiprocessing.Pool`
        a multiprocessing pool

    """
    # declare some state for this run as global variables to speed up access in multiprocess pool
    _globals = {
        't_true': setup_data.t_true,
        's_true': setup_data.s_true,
        'defocus': setup_data.focus_diversity,
        'setup_parameters': setup_parameters,
        'decoder_ring': codex,
        'diffraction': setup_data.diffraction,
    }
    if parallel is True:
        if nthreads is None:
            nproc = cpu_count() - 1
        else:
            nproc = nthreads
        pool = Pool(processes=nproc, initializer=prepare_globals, initargs=[_globals])
    else:
        pool = None
    prepare_globals({**_globals, 'pool': pool})
    return pool
