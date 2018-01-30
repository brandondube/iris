import time
from itertools import product

import numpy as np

from prysm import FringeZernike, Seidel, MTF
from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement

from iris.util import round_to_int
from iris.recipes import sph_from_focusdiverse_axial_mtf


def generate_axial_truth_coefs(max_val, num_steps, symmetric=True):
    '''Generates a cartesian product of W040, W060, and W080 subject to a best
    focus constraint for minimum RMS wavefront error.

    Note that for a large num_steps, the output will be large;
    num_steps=10 will produce 1,000 items.

    Parameters
    ----------
    max_val : TYPE
        Description
    num_steps : TYPE
        Description
    symmetric : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    '''
    if symmetric is True:
        lower = -max_val
    else:
        lower = 0

    w040 = np.linspace(lower, max_val, num_steps)
    w060 = np.linspace(lower, max_val, num_steps)
    w080 = np.linspace(lower, max_val, num_steps)

    # take the product
    coefs = list(product(w040, w060, w080))

    # here, figure out what w020 is for best focus and add it to the
    # coefficients
    return coefs
