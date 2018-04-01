from pathlib import Path
from itertools import product

import numpy as np

from iris.data import PersistentQueue

spherical_amounts = np.asarray([0, 0.05, 0.1, 0.2])
coma_amounts = np.asarray([0.05, 0.1, 0.2])
angles = np.radians(np.linspace(0, 90, 18))

sets = product(spherical_amounts, coma_amounts)
sph_amounts_tiled, coma_amounts_tiled = [], []
for s in sets:
    sph_amounts_tiled.append(s[0])
    coma_amounts_tiled.append(s[1])

s = len(coma_amounts_tiled)

sph_amounts_tiled, coma_amounts_tiled = np.asarray(sph_amounts_tiled), np.asarray(coma_amounts_tiled)

sphcoef = (np.ones(18) * sph_amounts_tiled.reshape((s, 1))).ravel()
xcoef, ycoef = np.cos(angles), np.sin(angles)  # flipped -- optics convention
xcoef = (xcoef * coma_amounts_tiled.reshape((s, 1))).ravel()
ycoef = (ycoef * coma_amounts_tiled.reshape((s, 1))).ravel()

out = np.zeros((sphcoef.shape[0], 16))
out[:, 5] = sphcoef
out[:, 10] = sphcoef / -2
out[:, 15] = sphcoef / 4
out[:, 3] = xcoef
out[:, 4] = xcoef

chunks = np.split(out, 4)

rootc = Path(__file__).parent / 'data' / 'coma-vs-angle'
roota = Path(__file__).parent / 'data' / 'astigmatism-vs-angle'
for idx, chunk in enumerate(chunks):
    p = rootc / str(idx) / 'queue.pkl'
    q = PersistentQueue(p, overwrite=True)
    items = list(chunk)
    q.put_many(items)

    p = roota / str(idx) / 'queue.pkl'
    q = PersistentQueue(p, overwrite=True)
    items = list(chunk)
    q.put_many(items)
