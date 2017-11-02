'''
This file contains recipes for common tasks, e.g. computing an MTF from a Hopkins
(in prysm, misgnomered as Seidel) wavefront expansion and f/#
'''

from prysm import Seidel, FringeZernike, PSF, MTF, Lens

def SeidelSphericalToMTF(w040, w060, w080, fno):
    samples = 256
    wavelength = 0.5
    efl = 50 # can be cancelled out, but not done here
    lens = Lens(efl=efl, fno=fno, wavelength=wavelength, samples=samples,
                fields=[0,1], aberrations={
                    'W040': w040,
                    'W060': w060,
                    'W080': w080,
                })
    lens.autofocus()
    m = lens._make_mtf(0)
    return m

def SeidelSphericalToMTF_ts(w040, w060, w080, fno, freqs):
    mtf = SeidelSphericalToMTF(w040, w060, w080, fno)
    tan = mtf.exact_polar(freqs=freqs, azimuths=0)
    sag = mtf.exact_polar(freqs=freqs, azimuths=90)
    return tan, sag