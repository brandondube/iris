from functools import partial
from collections import OrderedDict

import numpy as np

from mystic.solvers import PowellDirectionalSolver
from mystic.termination import VTR

from prysm import FringeZernike, Seidel, PSF, MTF, config
from prysm.thinlens import image_displacement_to_defocus
config.set_precision(32)

def seidel_coefs_to_dict(wavefrontcoefs):
    w020, w040, w060, w080, w131, w151, w171, w220, w222, w242 = wavefrontcoefs
    return OrderedDict(\
        W020=w020,
        W040=w040,
        W060=w060,
        W080=w080,
        W131=w131,
        W151=w151,
        W171=w171,
        W220=w220,
        W222=w222,
        W242=w242)

def seidel_dict_to_coefs(dictionary):
    return list(dictionary.values())

def mtf_cost_fcn(true_tan, true_sag, sim_tan, sim_sag):
    ''' A cost function that compares a measured or simulated T/S MTF to a
        simulated one.
    '''
    t = ((true_tan-sim_tan)**2).sum()
    s = ((true_sag-sim_sag)**2).sum()
    return t+s

def mtf_ts_extractor(psf, freqs):
    ''' Extracts the T and S MTF from a PSF object.
    '''
    mtf = MTF.from_psf(psf)
    tan = mtf.exact_polar(freqs=freqs, azimuths=0)
    sag = mtf.exact_polar(freqs=freqs, azimuths=90)
    return tan, sag

def axis_sph_constraint(wavefrontcoefs):
    ''' Constrains wavefront coefficients with field dependence to be zero.
    '''
    pass

def seidel_solve_fcn(truth_tan, truth_sag, freqs, efl, fno, wavelength, field, wavefrontcoefs):
    abers = seidel_coefs_to_dict(wavefrontcoefs)
    pupil = Seidel(**abers, epd=efl/fno, wavelength=wavelength, samples=256, field=field)
    psf = PSF.from_pupil(pupil, efl=efl, padding=2)
    
    t, s = mtf_ts_extractor(psf, freqs)
    return mtf_cost_fcn(truth_tan, truth_sag, t, s)

def seidel_solve_fcn_focusdiv(truth_tans, truth_sags, freqs, focuses, efl, fno, wavelength, field, wavefrontcoefs):
    abers = seidel_coefs_to_dict(wavefrontcoefs)
    focus_wvs = image_displacement_to_defocus(focuses, fno, wavelength)
    total_cost = 0
    for focus, t, s in zip(focus_wvs, truth_tans, truth_sags):
        lcl_abers = abers.copy()
        lcl_abers['W020'] += focus
        lcl_wavefrontcoefs = seidel_dict_to_coefs(lcl_abers)
        total_cost += seidel_solve_fcn(t, s, freqs, efl, fno, wavelength, field, lcl_wavefrontcoefs)
    
    return total_cost

def seidel_solve_fcn_fldconstant_only(truth_tan, truth_sag, freqs, efl, fno, wavelength, wavefrontcoefs):
    pupil = Seidel(W020=wavefrontcoefs[0], W040=wavefrontcoefs[1], W060=wavefrontcoefs[2], W080=wavefrontcoefs[3],
                 epd=efl/fno, wavelength=wavelength, samples=256)
    psf = PSF.from_pupil(pupil, efl=efl, padding=2)

    t, s = mtf_ts_extractor(psf, freqs)
    return mtf_cost_fcn(truth_tan, truth_sag, t, s)

def seidel_solve_fcn_fldconstant_only_focusdiv(truth_tans, truth_sags, freqs, focuses, efl, fno, wavelength, wavefrontcoefs):
    focus_wvs = image_displacement_to_defocus(focuses, fno, wavelength)
    total_cost = 0
    for focus, t, s in zip(focus_wvs, truth_tans, truth_sags):
        lcl_coefs = list(wavefrontcoefs)
        lcl_coefs[0] += focus
        total_cost += seidel_solve_fcn_fldconstant_only(t, s, freqs, efl, fno, wavelength, lcl_coefs)
    
    return total_cost

def generic_solve_fcn(truth_tan, truth_sag, freqs, efl, fno, wavelength, wavefrontcoefs):
    z4, z9, z16, z25, z7, z8, z14, z15, z23, z24, z5, z6 = wavefrontcoefs

    pupil = FringeZernike(base=1,
                          Z4=z4, Z9=z9, Z16=z16, Z25=z25,                   # spherical
                          Z7=z7, Z8=z8, Z14=z14, Z15=z15, Z23=z23, Z24=z24, # coma,
                          Z5=z5, Z6=z6,                                     # astigmatism
                          epd=efl/fno, wavelength=wavelength, samples=256,
                          opd_unit='nm', rms_norm=True)
    psf = PSF.from_pupil(pupil, efl=efl, padding=2)

    t, s = mtf_ts_extractor(psf, freqs)
    return mtf_cost_fcn(truth_tan, truth_sag, t, s)

def generic_solve_fcn_fd(truth_tan, truth_sag, freqs, efl, fno, wavelength, focus_div, wavefrontcoefs):
    z4, z9, z16, z25, z7, z8, z14, z15, z23, z24, z5, z6 = wavefrontcoefs

    mtfgrid_t = []
    mtfgrid_s = []
    for focus in focus_div:
        pupil = FringeZernike(base=1,
                              Z4=z4+focus,
                              Z9=z9, Z16=z16, Z25=z25,                          # spherical
                              Z7=z7, Z8=z8, Z14=z14, Z15=z15, Z23=z23, Z24=z24, # coma,
                              Z5=z5, Z6=z6,                                     # astigmatism
                              epd=efl/fno, wavelength=wavelength, samples=256,
                              opd_unit='nm', rms_norm=True)
        psf = PSF.from_pupil(pupil, efl=efl, padding=2)
        mtf = MTF.from_psf(psf)

        tan = mtf.exact_polar(freqs=freqs, azimuths=0)
        sag = mtf.exact_polar(freqs=freqs, azimuths=90)
        mtfgrid_t.append(tan)
        mtfgrid_s.append(sag)
    
    mtfgrid_t = np.asarray(mtfgrid_t)
    mtfgrid_s = np.asarray(mtfgrid_s)
    component_t = np.sum((truth_tan-mtfgrid_t)**2)
    component_s = np.sum((truth_sag-mtfgrid_s)**2)
    return component_t+component_s

def phase_from_mtf(mtf_tan, mtf_sag, freqs, efl, fno, wavelength, focus_diversity=None):
    ''' Uses a single tangential and sagittal MTF capture to retrieve
        the wavefront of an optical system.
    '''
    mtf_tan, mtf_sag = np.asarray(mtf_tan), np.asarray(mtf_sag)
    if focus_diversity is None:
        opt_fcn = partial(generic_solve_fcn, mtf_tan, mtf_sag, freqs, efl, fno, wavelength)
    else:
        opt_fcn = partial(generic_solve_fcn_fd, mtf_tan, mtf_sag, freqs, efl, fno, wavelength, focus_diversity)

    
    # deciding the # of dims for  powell
    # defocus
    # sph3
    # sph5
    # sph7
    # cma3          (x2)
    # cma5_aperture (x2)
    # cma7_aperture (x2)
    # astig         (x2)
    # = 1+1+1+1+2+2+2+2
    # = 12 dims
    solver = PowellDirectionalSolver(12)
    solver.SetInitialPoints([0] * 12)
    #solver.SetTermination(VTR(tolerance=0.005))
    solver.SetEvaluationLimits(generations=25)
    solver.Solve(opt_fcn)
    solver.Finalize()
    return solver.solution_history

def makesolver(numdims):
    solver = PowellDirectionalSolver(numdims)
    solver.SetInitialPoints([0] * numdims)
    solver.SetEvaluationLimits(generations=25)
    return solver