''' Wraps prysm classes to introduce gradient operations for mathematical
    operations.
'''

from pyrsm import FringeZernike, PSF, MTF

class FringeZernike(FringeZernike):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PSF(PSF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MTF(MTF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)