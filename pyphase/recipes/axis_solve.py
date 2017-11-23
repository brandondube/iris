''' Solve for aberrations on the optical axis given some truth MTF values
    and lens parameters.
'''
from functools import partial
from multiprocessing import Pool

import numpy as np

from scipy.optimize import minimize

from prysm import FringeZernike, Seidel, MTF
from prysm.thinlens import image_displacement_to_defocus
from prysm.mtf_utils import mtf_ts_extractor

from pyphase.util import mtf_cost_fcn, parse_cost_by_iter_lbfgsb
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

def realize_focus_plane(setup_parameters, base_wavefront, t_true, s_true, wvs_defocus):
    # pull metadata out of the lens
    s = setup_parameters
    efl, fno, wvl, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    # build a focus wavefront, add it to the base aberrations,
    # propagate to MTF, and compute the cost fcn.
    focus_wvfront = Seidel(W020=wvs_defocus, epd=efl/fno, wavelength=wvl, samples=samples)
    prop_wvfront = base_wavefront.merge(focus_wvfront)
    mtf = MTF.from_pupil(prop_wvfront, efl)
    t, s = mtf_ts_extractor(mtf, s['freqs'])
    return mtf_cost_fcn(t_true, s_true, t, s)


def optfcn(setup_parameters, wvfront_defocus, tan, sag, pool, wavefrontcoefs):
    # generate a "base pupil" with some aberration content
    s = setup_parameters
    zsph3, zsph5, zsph7 = wavefrontcoefs
    efl, fno, wavelength, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    pupil = FringeZernike(Z9=zsph3, Z16=zsph5, Z25=zsph7, base=1,
                          epd=efl/fno, wavelength=wavelength, samples=samples)
    # for each focus plane, compute the cost function
    rfp_mp = partial(realize_focus_plane,
                     setup_parameters,
                     pupil)
    costfcn = pool.starmap(rfp_mp, zip(tan, sag, wvfront_defocus))
    return np.asarray(costfcn).sum()

def sph_from_focusdiverse_axial_mtf(setup_parameters, truth_dataframe):
    # extract the data
    (focus_diversity,
     ax_t, ax_s) = grab_axial_data(setup_parameters, truth_dataframe)

    # casting ndarray to list makes it a list of arrays where the first index
    # is the focal plane and the second frequency.
    ax_t, ax_s = list(ax_t), list(ax_s)
    pool = Pool()
    optimizer_function = partial(optfcn,
                                 setup_parameters,
                                 focus_diversity,
                                 ax_t,
                                 ax_s,
                                 pool)

    parameter_vectors = []

    def callback(x):
        parameter_vectors.append(x)

    try:
        with forcefully_redirect_stdout() as txt:
            result = minimize(
                optimizer_function,
                [0, 0, 0],
                method='L-BFGS-B',
                options={
                    'disp': True
                },
                callback=callback)

        cost_by_iter = parse_cost_by_iter_lbfgsb(txt.captured)
        result.x_iter = parameter_vectors
        result.fun_iter = cost_by_iter
        pool.close()
        pool.join()
        return result
    except Exception as e:
        pool.close()
        pool.join()
        raise e
