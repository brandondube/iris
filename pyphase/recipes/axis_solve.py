''' Solve for aberrations on the optical axis given some truth MTF values
    and lens parameters.
'''
from functools import partial

import numpy as np

from scipy import minimize

from prysm import Seidel, MTF, PSF
from prysm.thinlens import image_displacement_to_defocus

from pyphase import mtf_ts_extractor, mtf_cost_fcn


def grab_axial_data(base_lens, truth_dataframe):
    # extract the axial T,S MTF data
    freqs = np.unique(truth_dataframe.Freq.as_matrix())
    axial_mtf_data = truth_dataframe[truth_dataframe.Field == 0]
    focuspos = np.unique(axial_mtf_data.Focus.as_matrix())
    wvfront_defocus = image_displacement_to_defocus(focuspos, base_lens.fno, base_lens.wavelength)
    ax_t = []
    ax_s = []
    for pos in focuspos:
        fd = axial_mtf_data[axial_mtf_data.Focus == pos]
        ax_t.append(fd[fd.Azimuth == 'Tan']['MTF'].as_matrix())
        ax_s.append(fd[fd.Azimuth == 'Sag']['MTF'].as_matrix())
    
    return freqs, wvfront_defocus, ax_t, ax_s

def realize_focus_plane(base_lens, freqs, base_wavefront, wvs_defocus, t_true, s_true):
    # pull metadata out of the lens
    efl, fno, wvl, samples = base_lens.efl, base_lens.fno, base_lens.wavelength, base_lens.samples
    # build a focus wavefront, add it to the base aberrations, propagate to MTF, and compute the cost fcn.
    focus_wvfront = Seidel(W020=wvs_defocus, epd=efl /
                           fno, wavelength=wvl, samples=samples)
    prop_wvfront = base_wavefront.merge(focus_wvfront)
    psf = PSF.from_pupil(prop_wvfront, efl)
    mtf = MTF.from_psf(psf)
    t, s = mtf_ts_extractor(mtf, freqs)
    return mtf_cost_fcn(t_true, s_true, t, s)


def optfcn(base_lens, freqs, wvfront_defocus, ax_t, ax_s, wavefrontcoefs):
    # generate a "base pupil" with some aberration content
    w040, w060, w080 = wavefrontcoefs
    opt_lens = base_lens.clone()
    opt_lens.aberrations = {
        'W040': w040,
        'W060': w060,
        'W080': w080,
    }
    opt_lens.autofocus()
    base_p = opt_lens._make_pupil(0)

    # for each focus plane, compute the cost function
    rfp_mp = partial(realize_focus_plane,
                     opt_lens,
                     freqs,
                     base_p)
    #mpclient.prepare_imports(rfp_mp, globals())
    #costfcn = mpclient.map(rfp_mp, wvfront_defocus, ax_t, ax_s)
    costfcn = []
    for defocus, t, s in zip(wvfront_defocus, ax_t, ax_s):
        costfcn.append(rfp_mp(defocus, t, s))
    return np.asarray(costfcn).sum()


def sph_from_focusdiverse_axial_mtf(base_lens, truth_dataframe):
    # extract the data
    freqs, focus_diversity, ax_t, ax_s = grab_axial_data(
        base_lens, truth_dataframe)

    optimizer_function = partial(optfcn,
                                 base_lens,
                                 freqs,
                                 focus_diversity,
                                 ax_t,
                                 ax_s)
    abers = minimize(optimizer_function, [0, 0, 0], method='Nelder-Mead')

    return abers
