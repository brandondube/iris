"""Misc. utilities."""
import re
import io
from operator import itemgetter

import numpy as np

from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement
from prysm.macros import SimulationConfig


def round_to_int(value, integer):
    """Round a value to the nearest integer.

    Parameters
    ----------
    value : `float` or `int`
        a value to round
    integer : `int`
        what int to round to (closest 5, 10, etc)

    Returns
    -------
    `int`
        rounded value

    """
    return integer * round(float(value) / integer)


def make_focus_range_realistic_number_of_microns(cfg, round_focus_to=5):
    """Create a modified config dictionary that has a focus range which is realistic for an MTF bench.

    Parameters
    ----------
    config : `prysm.macros.SimulationConfig`
        dict with keys focus_range_waves, fno, wavelength
    round_focus_to : `int`, optional
        integer number of microns to round focus to

    Returns
    -------
    `prysm.macros.SimulationConfig`
        modified SimulationConfig with focus range updaterounded to the specified number of microns

    """
    # get the amount of defocus diversity in microns
    focus_wvs = cfg.focus_range_waves
    focus_um = defocus_to_image_displacement(focus_wvs, cfg.fno, cfg.wvl, cfg.focus_zernike, cfg.focus_normed)

    # round the focus range in microns and map it back into the given units
    focus_round = round_to_int(focus_um, round_focus_to)
    focus_round_wvs = image_displacement_to_defocus(focus_round, cfg.fno, cfg.wvl, cfg.focus_zernike, cfg.focus_normed)

    # unpack the namedtuple to a dict, modify the focus_range_waves field, and convert back to a
    # namedtuple; return the new instance.
    cfg_dict = cfg._asdict()
    cfg_dict['focus_range_waves'] = focus_round_wvs
    return SimulationConfig(**cfg_dict)


def split_lbfgsb_iters(string):
    """Split captured output from L-BFGS-B into a sequence of outputs, each containing a single chunk.

    Parameters
    ----------
    string : `str`
        a string output from L-BFGS-B

    Returns
    -------
    `list`
        a list of strings, each is an output from L-BFGS-B

    """
    starts = [s.start() for s in re.finditer('RUNNING', string)]
    blobs = []
    for idx in range(len(starts)):
        if idx is not len(starts) - 1:
            s = string[starts[idx]:starts[idx + 1]]
        else:
            s = string[starts[idx]:]
        blobs.append(s)
    return blobs


def parse_cost_by_iter_lbfgsb(string):
    """Parse the cost function history from L-BFGS-B optimizer print statements.

    Parameters
    ----------
    string : `str`
        A string containing the entire L-BFGS-B output

    Returns
    -------
    `list`
        a list of cost function values by iteration

    """
    # use regex to find lines that include "at iteration" information.
    lines_with_cost_function_values = \
        re.findall(r'At iterate\s*\d*?\s*f=\s*-*?\d*.\d*D[+-]\d*', string)

    # grab the value from each line and convert to python floats
    fortran_values = [s.split()[-1] for s in lines_with_cost_function_values]
    # fortran uses "D" to denote double and "raw" exp notation,
    # fortran value 3.0000000D+02 is equivalent to
    # python value  3.0000000E+02 with double precision
    return [float(s.replace('D', 'E')) for s in fortran_values]


def prepare_document_local(sim_params, codex, truth_params, truth_rmswfe, rmswfe_iter, normed, optimization_result):
        """Prepare a document (dict) for insertion into the results database.

        Parameters
        ----------
        sim_params : `dict`
            dictionary with keys for efl, fno, wavelength, samples, focus_planes, focus_range_waves, freqs
        codex : `dict`
            dictionary with integer keys in [0, len(truth_params)] and values 'Z1'..'Z49'; each key
                is a position in the parameter vector and each value is the appropriate zernike.
        truth_parms: iterable
            truth parameters
        truth_rmswfe: `float`
            RMS WFE of the truth
        rmswfe_iter : iterable
            rms wavefront error for each iteration
        normed : `bool`
            if the coefficients should be made unit RMS value
        optimization_result : object
            object with keys x, x_iter, fun, fun_iter, time

        Returns
        -------
        `dict`
            dictionary with keys, types:
                  # not going on database
                - sim_params, `~prysm.macro.SimulationConfig`
                - codex, `dict`
                - truth_params, `tuple`
                - zernike_norm, `bool`
                - result_iter, `list`
                - cost_iter, `list`
                - rrmswfe_iter, `list`
                - result_final, `tuple`
                  # going on database
                - truth_rmswfe, `float`
                - cost_first, `float`
                - cost_final, `float`
                - rrmswfe_first, `float`
                - rrmswfe_final, `float`
                - time, `float`
                - nit, `int`
                - nfev, `int`

        """
        x, xiter, f, fiter, t = itemgetter('x', 'x_iter', 'fun', 'fun_iter', 'time')(optimization_result)
        return {
            'global': False,
            'sim_params': sim_params,
            'codex': codex,
            'truth_params': truth_params,
            'zernike_normed': normed,
            'result_iter': xiter,
            'cost_iter': fiter,
            'rrmswfe_iter': rmswfe_iter,
            'result_final': x,
            'truth_rmswfe': truth_rmswfe,
            'cost_first': fiter[0],
            'cost_final': f,
            'rrmswfe_first': rmswfe_iter[0],
            'rrmswfe_final': rmswfe_iter[-1],
            'time': t,
            'nit': optimization_result.nit,
            'nfev': optimization_result.nfev,
            'nrandomstart': False,
        }


def prepare_document_global(sim_params, codex, truth_params, truth_rmswfe, rmswfe_iter, normed, optimization_result):
        """Prepare a document (dict) for insertion into the results database.

        Parameters
        ----------
        sim_params : `dict`
            dictionary with keys for efl, fno, wavelength, samples, focus_planes, focus_range_waves, freqs
        codex : `dict`
            dictionary with integer keys in [0, len(truth_params)] and values 'Z1'..'Z49'; each key
                is a position in the parameter vector and each value is the appropriate zernike.
        truth_parms: iterable
            truth parameters
        truth_rmswfe: `float`
            RMS WFE of the truth
        rmswfe_iter : iterable
            rms wavefront error for each iteration
        normed : `bool`
            if the coefficients should be made unit RMS value
        optimization_result : `object`
            object with keys x, x_iter, fun, fun_iter, time

        Returns
        -------
        `dict`
            dictionary with keys, types:
                  # not going on database
                - sim_params, `~prysm.macro.SimulationConfig`
                - codex, `dict`
                - truth_params, `tuple`
                - zernike_norm, `bool`
                - result_iter, `list`
                - cost_iter, `list`
                - rrmswfe_iter, `list`
                - result_final, `tuple`
                  # going on database
                - truth_rmswfe, `float`
                - cost_first, `float`
                - cost_final, `float`
                - rrmswfe_first, `float`
                - rrmswfe_final, `float`
                - time, `float`
                - nit, `int`
                - nfev, `int`

        Notes
        -----
        Similar to the _local variant, but sets the global attr to true and xxxx_iter attrs are
        iterables of iterables, where each outer iterable corresponds to a local optimization
        attempt.  i.e., for a situation where there were 3 random starting guesses with 10, 20 and
        30 iterations, fun_iter will have len of 3, fun_iter[0] will have len of 10, and so on

        """
        x, xiter, f, fiter, t = itemgetter('x', 'x_iter', 'fun', 'fun_iter', 'time')(optimization_result)
        nstart = len(xiter)
        return {
            'global': True,
            'sim_params': sim_params,
            'codex': codex,
            'truth_params': truth_params,
            'zernike_normed': normed,
            'result_iter': xiter,
            'cost_iter': fiter,
            'rrmswfe_iter': rmswfe_iter,
            'result_final': x,
            'truth_rmswfe': truth_rmswfe,
            'cost_first': fiter[0][0],
            'cost_final': f,
            'rrmswfe_first': rmswfe_iter[0][0],
            'rrmswfe_final': rmswfe_iter[-1][-1],
            'time': t,
            'nit': optimization_result.nit,
            'nfev': optimization_result.nfev,
            'nrandomstart': nstart,
        }


def pgm_img_to_array(imgstr):
    """Convert a string version of an ASCII pgm file to a numpy array.

    Parameters
    ----------
    imgstr : `str`
        image as a string

    Returns
    -------
    `numpy.ndarray`
        2D ndarray of the image

    Notes
    -----
    Operates under the assumption that the image will be to the specification
    used by the trioptics imagemaster HR; that is, the header shall be 3 lines
    long.

    """
    string = io.StringIO(imgstr)
    return np.loadtxt(string, dtype=np.int16, skiprows=3)
