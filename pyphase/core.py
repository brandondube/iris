import time
from itertools import product

import numpy as np
import pandas as pd

from prysm import FringeZernike, Seidel, PSF, MTF, config
from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement

from prysm.mtf_utils import mtf_ts_extractor, mtf_ts_to_dataframe

from pyphase.mongoq import JobQueue
from pyphase.util import round_to_int
from pyphase.recipes import sph_from_focusdiverse_axial_mtf

#config.set_precision(32)
config.set_zernike_base(1)

def thrufocus_mtf_from_wavefront(focused_wavefront, sim_params, focus_diversity):
    ''' Creates a thru-focus T/S MTF curve at each frequency requested from a
        focused wavefront.
    '''
    s = sim_params
    focusdiv_wvs = image_displacement_to_defocus(focus_diversity, s['fno'], s['wavelength'])
    dfs = []
    for focus, displacement in zip(focusdiv_wvs, focus_diversity):
        wvfront_defocus = Seidel(W020=focus,
                                 samples=s['samples'],
                                 epd=s['efl']/s['fno'],
                                 wavelength=s['wavelength'])
        psf = PSF.from_pupil(focused_wavefront.merge(wvfront_defocus), efl=s['efl'])
        mtf = MTF.from_psf(psf)
        tan, sag = mtf_ts_extractor(mtf, s['freqs'])
        dfs.append(mtf_ts_to_dataframe(tan, sag, freqs, focus=displacement))

    return pd.concat(dfs)

def generate_axial_truth_coefs(max_val, num_steps, symmetric=True):
    ''' Generates a cartesian product of W040, W060, and W080 subject to a best
        focus constraint for minimum RMS wavefront error.

        Note that for a large num_steps, the output will be large;
        num_steps=10 will produce 1,000 items.
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

class AxialWorker(object):
    ''' Works through the spherical aberration related axial work queue.
    '''
    def __init__(self, database):
        ''' Creates a new worker.
        '''
        self.db = database
        self.q = JobQueue(database, name='axial_queue')
        self.hopkins = [None, None, None, None]
        self.zernike = [None, None, None, None]

        efl = 50
        fno = 2.8
        lambda_ = 0.55
        extinction = 1000/(fno*lambda_)
        freqs = np.arange(0, extinction, 10)[1:] # skip 0
        self.sim_params = {
            'efl': efl,
            'fno': fno,
            'wavelength': lambda_,
            'samples': 128,
            'focus_planes': 21, # TODO: see if this many is necessary
            'focus_range_waves': 2,
            'freqs': freqs,
        }

    def prepare_queue(self):
        ''' Prepares the queue
        '''
        if self.q.is_empty:
            coefficients = generate_axial_truth_coefs(0.33, 10, symmetric=True)
            self.q.put(coefficients)

        return self.q

    def work_for(self, minutes):
        ''' work through the queue for some number of minutes.  Will stop taking
            jobs from the queue when minutes has elapsed.
        '''
        start = time.time()
        q = self.prepare_axial_work_queue()
        #next_job = q.get()
        pass

    def prepare_document(self, result_parameters, coefs_by_iter, erf_by_iter):
        ''' prepares a document (dict) for insertion into the results database
        '''
        doc = {
            'truth_hopkins': {
                'W020': self.hopkins[0],
                'W040': self.hopkins[1],
                'W060': self.hopkins[2],
                'W080': self.hopkins[3],
            },
            'truth_zernike': {
                'Z4': self.zernike[0],
                'Z9': self.zernike[1],
                'Z16': self.zernike[2],
                'Z25': self.zernike[3],
            },
            'sim_params': self.sim_params,
            'retrieved_zernike': result_parameters,
            'coefs_by_iter': coefs_by_iter,
            'erf_by_iter': erf_by_iter,
        }

        return doc

    def publish_document(self, doc):
        self.db.axial_results.insert_one(doc)

    def execute_queue_item(self, hopkins_coefs):
        sp = self.sim_params
        # first, compute the amount of focus diversity in microns and round it
        focusdiv_wvs = sp['focus_range_waves']
        focusdiv_um = round_to_int(defocus_to_image_displacement(
            focusdiv_wvs,
            self.sim_params['fno'],
            self.sim_params['wavelength']))

        # convert hopkins coefs to zernikes
        zerns = [1, 2, 3, 4]

        # get the zernikes we want from the fit
        self.zernike = [zerns[3], zerns[8], zerns[15], zerns[24]]
        z4, z9, z16, z24 = self.zernike
        efl, fno, wavelength, samples = sp['efl'], sp['fno'], sp['wavelength'], sp['samples']
        pupil = FringeZernike(Z4=z4, Z9=z9, Z16=z16, Z24=z24, epd=efl/fno,
                              wavelength=wavelength, samples=samples)

        truth_df = thrufocus_mtf_from_wavefront(pupil, self.sim_params, focusdiv_um)
        values = sph_from_focusdiverse_axial_mtf(self.sim_params, truth_df)
