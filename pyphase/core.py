import time
from itertools import product

import numpy as np

from prysm import FringeZernike, Seidel, MTF
from prysm.thinlens import image_displacement_to_defocus, defocus_to_image_displacement

from pyphase.mongoq import JobQueue
from pyphase.util import round_to_int
from pyphase.recipes import sph_from_focusdiverse_axial_mtf


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
        extinction = 1000 / (fno * lambda_)
        freqs = np.arange(0, extinction, 10)[1:]  # skip 0
        self.sim_params = {
            'efl': efl,
            'fno': fno,
            'wavelength': lambda_,
            'samples': 128,
            'focus_planes': 21,  # TODO: see if this many is necessary
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

    def publish_document(self, doc):
        self.db.axial_montecarlo_results.insert_one(doc)

    def execute_queue_item(self):
        sp = self.sim_params
        # first, compute the amount of focus diversity in microns and round it
        focusdiv_wvs = sp['focus_range_waves']
        focusdiv_um = round_to_int(defocus_to_image_displacement(
            focusdiv_wvs,
            sp['fno'],
            sp['wavelength']), 5)  # round to nearest 5 um

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
        z4, z9, z16, z25 = self.zernike
        efl, fno, wavelength, samples = sp['efl'], sp['fno'], sp['wavelength'], sp['samples']
        pupil = FringeZernike(Z4=z4, Z9=z9, Z16=z16, Z25=z25, epd=efl / fno,
                              wavelength=wavelength, samples=samples)

        truth_df = thrufocus_mtf_from_wavefront(pupil, cfg)
        result = sph_from_focusdiverse_axial_mtf(self.sim_params, truth_df)

        p_by_iter, e_by_iter = result.x_iter, result.fun_iter
        rmswfe_by_iter = []
        for p in p_by_iter:
            zdefocus, zsph3, zsph5, zsph7 = p
            efl, fno, wavelength, samples = sp['efl'], sp['fno'], sp['wavelength'], sp['samples']
            pup = FringeZernike(Z4=zdefocus, Z9=zsph3, Z16=zsph5, Z25=zsph7, base=1,
                                epd=efl / fno, wavelength=wavelength, samples=samples)

            p_err = pupil - pup
            rmswfe_by_iter.append(p_err.rms)
        return prepare_document(self.sim_params, self.hopkins, self.zernike, result.x, p_by_iter, e_by_iter, rmswfe_by_iter, result.time, pupil.rms)


def prepare_document(sim_params, hopkins, zernike, result_parameters, coefs_by_iter, erf_by_iter, rmswfe_by_iter, solve_time, true_rms):
        ''' prepares a document (dict) for insertion into the results database
        '''
        doc = {
            'sim_params': sim_params,
            'truth_hopkins': {
                'W020': hopkins[0],
                'W040': hopkins[1],
                'W060': hopkins[2],
                'W080': hopkins[3],
            },
            'truth_zernike': {
                'Z4': zernike[0],
                'Z9': zernike[1],
                'Z16': zernike[2],
                'Z25': zernike[3],
            },
            'truth_rmswfe': 0,
            'zernike_normed': False,
            'retrieved_zernike': {
                'Z4': result_parameters.Z4,
                'Z9': result_parameters.Z9,
                'Z16': result_parameters.Z16,
                'Z25': result_parameters.Z25,
            },
            'coefs_by_iter': coefs_by_iter,
            'erf_by_iter': erf_by_iter,
            'rmswfe_by_iter': rmswfe_by_iter,
            'solve_time': solve_time,
        }

        return doc
