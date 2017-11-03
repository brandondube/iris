''' Solve for aberrations on the optical axis given some truth MTF values
    and lens parameters.
'''
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from scipy.optimize import minimize

from prysm import FringeZernike, Seidel, MTF, PSF
from prysm.thinlens import image_displacement_to_defocus

from pyphase import mtf_ts_extractor, mtf_cost_fcn
from pyphase.util import MTFDataCube

def grab_axial_data(setup_parameters, truth_dataframe):
    # extract the axial T,S MTF data
    s = setup_parameters
    freqs = np.unique(truth_dataframe.Freq.as_matrix())
    axial_mtf_data = truth_dataframe[truth_dataframe.Field == 0]
    focuspos = np.unique(axial_mtf_data.Focus.as_matrix())
    wvfront_defocus = image_displacement_to_defocus(focuspos, s['fno'], s['wavelength'])
    ax_t = []
    ax_s = []
    for pos in focuspos:
        fd = axial_mtf_data[axial_mtf_data.Focus == pos]
        ax_t.append(fd[fd.Azimuth == 'Tan']['MTF'].as_matrix())
        ax_s.append(fd[fd.Azimuth == 'Sag']['MTF'].as_matrix())

    freqs = np.asarray(freqs)
    wvfront_defocus = np.asarray(wvfront_defocus)
    ax_t = np.asarray(ax_t)
    ax_s = np.asarray(ax_s)
    return freqs, wvfront_defocus, ax_t, ax_s

def grab_mtf_cubes(truth_dataframe):
    # copy the datafarme for manipulation
    df = truth_dataframe.copy()
    df.Fields = df.Field.round(4)
    df.Focus = df.Focus.round(6)
    sorted_df = df.sort_values(by=['Focus', 'Field', 'Freq'])
    T = sorted_df[sorted_df.Azimuth == 'Tan']
    S = sorted_df[sorted_df.Azimuth == 'Sag']
    focus = np.unique(df.Focus.as_matrix())
    fields = np.unique(df.Fields.as_matrix())
    freqs = np.unique(df.Freq.as_matrix())
    d1, d2, d3 = len(focus), len(fields), len(freqs)
    t_mat = T.as_matrix.reshape((d1, d2, d3))
    s_mat = S.as_matrix.reshape((d1, d2, d3))
    t_cube = MTFDataCube(data=t_max, focus=focus, field=fields, freq=freqs, azimuth='Tan')
    s_cube = MTFDataCube(data=s_max, focus=focus, field=fields, freq=freqs, azimuth='Sag')
    return t_cube, s_cube

def realize_focus_plane(setup_parameters, base_wavefront, freqs, t_true, s_true, wvs_defocus):
    # pull metadata out of the lens
    s = setup_parameters
    efl, fno, wvl, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    # build a focus wavefront, add it to the base aberrations,
    # propagate to MTF, and compute the cost fcn.
    focus_wvfront = Seidel(W020=wvs_defocus, epd=efl /
                           fno, wavelength=wvl, samples=samples)
    prop_wvfront = base_wavefront.merge(focus_wvfront)
    psf = PSF.from_pupil(prop_wvfront, efl)
    mtf = MTF.from_psf(psf)
    t, s = mtf_ts_extractor(mtf, freqs)
    return mtf_cost_fcn(t_true, s_true, t, s)


def optfcn(setup_parameters, freqs, wvfront_defocus, ax_t, ax_s, pool, wavefrontcoefs):
    # generate a "base pupil" with some aberration content
    s = setup_parameters
    zsph3, zsph5, zsph7 = wavefrontcoefs
    efl, fno, wavelength, samples = s['efl'], s['fno'], s['wavelength'], s['samples']
    pupil = FringeZernike(Z8=zsph3, Z15=zsph5, Z24=zsph7,
                          epd=efl/fno, wavelength=wavelength, samples=samples)
    # for each focus plane, compute the cost function
    rfp_mp = partial(realize_focus_plane,
                     setup_parameters,
                     pupil,
                     freqs,
                     ax_t,
                     ax_s)
    costfcn = pool.map(rfp_mp, wvfront_defocus)
    return np.asarray(costfcn).sum()

def thrufocus_spherical_cost_gradient(true_t, true_s, compute_t, compute_s):
    # forward model:
    #   build pupil
    #   fft
    #   ||^2
    #   fft
    #   magnitude

    # reverse model
    #   
    Mbar = 2 * (compute_t - true_s) + (compute_s - true_s)
    pass


def sph_from_focusdiverse_axial_mtf(setup_parameters, truth_dataframe):
    # extract the data
    (freqs,
     focus_diversity,
     ax_t, ax_s) = grab_axial_data(setup_parameters, truth_dataframe)

    pool = Pool(cpu_count()+1)
    optimizer_function = partial(optfcn,
                                 setup_parameters,
                                 freqs,
                                 focus_diversity,
                                 ax_t,
                                 ax_s,
                                 pool)
    try:
        abers = minimize(optimizer_function, [0, 0, 0], method='Powell')
        pool.close()
        pool.join()
        return abers
    except Exception as e:
        pool.close()
        pool.join()
        raise e
