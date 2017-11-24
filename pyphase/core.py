import time
from itertools import product

import numpy as np
import pandas as pd

from prysm import FringeZernike, Seidel, MTF
from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement

from prysm.mtf_utils import mtf_ts_extractor, mtf_ts_to_dataframe

from pyphase.mongoq import JobQueue
from pyphase.util import round_to_int
from pyphase.recipes import sph_from_focusdiverse_axial_mtf


def thrufocus_mtf_from_wavefront(focused_wavefront, sim_params):
    ''' Creates a thru-focus T/S MTF curve at each frequency requested from a
        focused wavefront.
    '''
    s = sim_params
    focusdiv_wvs = np.linspace(-s['focus_range_waves'], s['focus_range_waves'], s['focus_planes'])
    focusdiv_um = defocus_to_image_displacement(focusdiv_wvs, s['fno'], s['wavelength'])
    dfs = []
    for focus, displacement in zip(focusdiv_wvs, focusdiv_um):
        wvfront_defocus = Seidel(W020=focus,
                                 samples=s['samples'],
                                 epd=s['efl']/s['fno'],
                                 wavelength=s['wavelength'])
        mtf = MTF.from_pupil(focused_wavefront.merge(wvfront_defocus), efl=s['efl'])
        tan, sag = mtf_ts_extractor(mtf, s['freqs'])
        dfs.append(mtf_ts_to_dataframe(tan, sag, s['freqs'], focus=displacement))

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
    def __init__(self, mongoclient, database):
        ''' Creates a new worker.
        '''
        self.db = mongoclient[database]
        self.q = JobQueue(self.db, name='axial_queue')
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

    def prepare_document(self, result_parameters, coefs_by_iter, erf_by_iter, rmswfe_by_iter):
        ''' prepares a document (dict) for insertion into the results database
        '''
        doc = {
            'sim_params': self.sim_params,
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
            'retrieved_zernike': {
                'Z4': result_parameters.Z4,
                'Z9': result_parameters.Z9,
                'Z16': result_parameters.Z16,
                'Z25': result_parameters.Z25,
            },
            'coefs_by_iter': coefs_by_iter,
            'erf_by_iter': erf_by_iter,
            'rmswfe_by_iter': rmswfe_by_iter,
        }

        return doc

    def publish_document(self, doc):
        self.db.axial_montecarlo_results.insert_one(doc)

    def execute_queue_item(self, hopkins_coefs):
        sp = self.sim_params
        # first, compute the amount of focus diversity in microns and round it
        focusdiv_wvs = sp['focus_range_waves']
        focusdiv_um = round_to_int(defocus_to_image_displacement(
            focusdiv_wvs,
            sp['fno'],
            sp['wavelength']), 5) # round to nearest 5 um

        # copy and update the focus diversity in the config dict
        cfg = self.sim_params.copy()
        cfg['focus_range_waves'] = image_displacement_to_defocus(
            focusdiv_um,
            sp['fno'],
            sp['wavelength'])

        # convert hopkins coefs to zernikes
        zerns = [1, 2, 3, 4]

        # get the zernikes we want from the fit
        self.zernike = [zerns[3], zerns[8], zerns[15], zerns[24]]
        z4, z9, z16, z24 = self.zernike
        efl, fno, wavelength, samples = sp['efl'], sp['fno'], sp['wavelength'], sp['samples']
        pupil = FringeZernike(Z4=z4, Z9=z9, Z16=z16, Z24=z24, epd=efl/fno,
                              wavelength=wavelength, samples=samples)

        truth_df = thrufocus_mtf_from_wavefront(pupil, cfg)
        result = sph_from_focusdiverse_axial_mtf(self.sim_params, truth_df)

        p_by_iter, e_by_iter = result.x_iter, result.cost_by_iter
        rmswfe_by_iter = []
        for p in p_by_iter:
            zdefocus, zsph3, zsph5, zsph7 = p
            efl, fno, wavelength, samples = sp['efl'], sp['fno'], sp['wavelength'], sp['samples']
            pup = FringeZernike(Z4=zdefocus, Z9=zsph3, Z16=zsph5, Z25=zsph7, base=1,
                                epd=efl/fno, wavelength=wavelength, samples=samples)

            p_err = pupil - pup
            rmswfe_by_iter.append(p_err.rms)
        return # TODO: this isn't what's done
