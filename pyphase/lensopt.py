from functools import partial

from multiprocessing.connection import Client
import numpy as np
import pandas as pd

from mystic.solvers import PowellDirectionalSolver
from mystic.termination import VTR

from scipy.optimize import minimize

from prysm import FringeZernike, PSF, MTF, Lens

from pyphase.core import (
    makesolver,
    seidel_solve_fcn,
    seidel_solve_fcn_focusdiv,
    seidel_solve_fcn_fldconstant_only,
    seidel_solve_fcn_fldconstant_only_focusdiv,
)

def lens_based_phase_retrieval(lens, mtfdata):
    ''' Uses an intelligent routine with multiple stages of optimization to
        retrieva the aberration coefficients for a lens.

        Pandas DataFrames are used to pass MTF data.  All measurements are
        expected to occur at the same frequencies.  The dataframe should look
        like the following:

        Field   Focus   Freq    Azimuth  MTF
        ------------------------------------
        0         0      10       T     0.9
        0         0      20       T     0.8
        ...
        -------------------------------------

    '''

    # extract the axial MTF data
    freqs = np.unique(mtfdata.Freq.as_matrix())
    axial_mtf_data = mtfdata[mtfdata.Field == 0]
    focuspos = np.unique(axial_mtf_data.Focus.as_matrix())
    ax_t = []
    ax_s = []
    for pos in focuspos:
        fd = axial_mtf_data[axial_mtf_data.Focus == pos]
        ax_t.append(fd[fd.Azimuth == 'Tan']['MTF'].as_matrix())
        ax_s.append(fd[fd.Azimuth == 'Sag']['MTF'].as_matrix())

    def send_current_state_to_plot(client, xk):
        client.send(xk)
        return

    # optimize for W020, W040, W060, W080 based on axial MTF data
    try:
        # if len errors, we only have one focus position.  Otherwise use focus
        # diverse optimiation.
        len(focuspos)
        #optfcn = partial(seidel_solve_fcn_focusdiv, ax_t, ax_s, freqs, focuspos,
        #                 lens.efl, lens.fno, lens.wavelength, 1)

        # set up a connection to the plotter
        address = ('localhost', 12345)
        conn = Client(address, authkey=b'pyphase_live')
        print('client connected')
        # make the callback and cost functions
        callback_fcn = partial(send_current_state_to_plot, conn)
        optfcn = partial(seidel_solve_fcn_fldconstant_only_focusdiv, ax_t, ax_s, freqs, focuspos, lens.efl, lens.fno, lens.wavelength)

        # kick off an optimizer
        start = [0,0,0,0]
        result = minimize(optfcn, start, method='Nelder-Mead', callback=callback_fcn)
        conn.send('quit')
        conn.close()
        return result
        
    except TypeError:
        #optfcn = partial(seidel_solve_fcn, ax_t[0], ax_s[0], freqs,
        #                 lens.efl, lens.fno, lens.wavelegth, 1)
        return solver

    # todo: solve for other coefs