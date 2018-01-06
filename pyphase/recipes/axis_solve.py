''' Solve for aberrations on the optical axis given some truth MTF values
    and lens parameters.
'''
from time import time
from functools import partial
from multiprocessing import Pool

import numpy as np

from scipy.optimize import minimize

from prysm import FringeZernike, Seidel, MTF
from prysm.thinlens import image_displacement_to_defocus
from prysm.mtf_utils import mtf_ts_extractor

from pyphase.util import mtf_cost_fcn, net_costfcn_reducer, parse_cost_by_iter_lbfgsb
from pyphase.forcefully_redirect_stdout import forcefully_redirect_stdout


def grab_axial_data(setup_parameters, truth_dataframe):
    # extract the axial T,S MTF data
    s = setup_parameters
    axial_mtf_data = truth_dataframe[truth_dataframe.Field == 0]
    focuspos = np.unique(axial_mtf_data.Focus.as_matrix())
    wvfront_defocus = image_displacement_to_defocus(focuspos, s['fno'], s['wavelength'])
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


def realize_focus_plane(setup_parameters, base_wavefront, t_true, s_true, defocus_wavefront):
    prop_wvfront = base_wavefront + defocus_wavefront
    mtf = MTF.from_pupil(prop_wvfront, setup_parameters['efl'])
    t, s = mtf_ts_extractor(mtf, setup_parameters['freqs'])
    return mtf_cost_fcn(t_true, s_true, t, s)


def optfcn(setup_parameters, defocus_pupils, tan, sag, pool, wavefrontcoefs):
    # generate a "base pupil" with some aberration content
    s = setup_parameters
    zdefocus, zsph3, zsph5, zsph7 = wavefrontcoefs
    efl, fno, wavelength, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    pupil = FringeZernike(Z4=zdefocus, Z9=zsph3, Z16=zsph5, Z25=zsph7, base=1,
                          epd=efl / fno, wavelength=wavelength, samples=samples)

    # for each focus plane, compute the cost function
    rfp_mp = partial(realize_focus_plane,
                     setup_parameters,
                     pupil)
    costfcn = pool.starmap(rfp_mp, zip(tan, sag, defocus_pupils))
    return net_costfcn_reducer(costfcn)


def optfcn_seq(setup_parameters, wvfront_defocus, tan, sag, wavefrontcoefs):
    # generate a "base pupil" with some aberration content
    s = setup_parameters
    zdefocus, zsph3, zsph5, zsph7 = wavefrontcoefs
    efl, fno, wavelength, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    pupil = FringeZernike(Z4=zdefocus, Z9=zsph3, Z16=zsph5, Z25=zsph7, base=1,
                          epd=efl / fno, wavelength=wavelength, samples=samples)

    # for each focus plane, compute the cost function
    costfcn = []
    for tan, sag, defocus in zip(tan, sag, wvfront_defocus):
        costfcn.append(realize_focus_plane(s, pupil, tan, sag, defocus))
    return net_costfcn_reducer(costfcn)


def sph_from_focusdiverse_axial_mtf(setup_parameters, truth_dataframe, guess=[0, 0, 0, 0]):
    # extract the data
    (focus_diversity,
     ax_t, ax_s) = grab_axial_data(setup_parameters, truth_dataframe)

    # casting ndarray to list makes it a list of arrays where the first index
    # is the focal plane and the second frequency.
    ax_t, ax_s = list(ax_t), list(ax_s)

    # precompute the defocus wavefronts to accelerate solving
    s = setup_parameters
    efl, fno, wvl, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    defocus_pupils = []
    for focus in focus_diversity:
        defocus_pupils.append(Seidel(W020=focus, epd=efl / fno, wavelength=wvl, samples=samples))
    pool = Pool()
    optimizer_function = partial(optfcn,
                                 setup_parameters,
                                 defocus_pupils,
                                 ax_t,
                                 ax_s,
                                 pool)

    parameter_vectors = []

    def callback(x):
        parameter_vectors.append(x)

    try:
        t_start = time()
        # do the optimization and capture the per-iteration information from stdout
        with forcefully_redirect_stdout() as txt:
            result = minimize(
                fun=optimizer_function,
                x0=guess,
                method='L-BFGS-B',
                options={'disp': True},
                callback=callback)

        t_end = time()
        # grab the extra data
        cost_by_iter = parse_cost_by_iter_lbfgsb(txt.captured)

        # add the guess to the front of the parameter vectors
        # cost_init = optfcn_seq(setup_parameters, focus_diversity, ax_t, ax_s, guess)
        parameter_vectors.insert(0, np.asarray(guess))
        # cost_by_iter.insert(0, cost_init)
        result.x_iter = parameter_vectors
        result.fun_iter = cost_by_iter
        result.time = t_end - t_start
        pool.close()
        pool.join()
        return result
    except Exception as e:
        pool.close()
        pool.join()
        raise e
