from pathlib import Path
from itertools import product

import numpy as np

from iris.data import PersistentQueue

spherical_amounts = np.asarray([0, 0.05, 0.1, 0.2])
coma_amounts = np.asarray([0.05, 0.1, 0.2])
angles = np.radians(np.linspace(0, 90, 19))

sets = product(spherical_amounts, coma_amounts)
sph_amounts_tiled, coma_amounts_tiled = [], []
for s in sets:
    sph_amounts_tiled.append(s[0])
    coma_amounts_tiled.append(s[1])

s = len(coma_amounts_tiled)

sph_amounts_tiled, coma_amounts_tiled = np.asarray(sph_amounts_tiled), np.asarray(coma_amounts_tiled)

sphcoef = (np.ones(angles.shape) * sph_amounts_tiled.reshape((s, 1))).ravel()
xcoef, ycoef = np.cos(angles), np.sin(angles)  # flipped -- optics convention
xcoef = (xcoef * coma_amounts_tiled.reshape((s, 1))).ravel()
ycoef = (ycoef * coma_amounts_tiled.reshape((s, 1))).ravel()

out_coma = np.zeros((sphcoef.shape[0], 16))
out_coma[:, 5] = sphcoef
out_coma[:, 10] = sphcoef / -2
out_coma[:, 15] = sphcoef / 4
out_coma[:, 3] = xcoef
out_coma[:, 4] = ycoef

out_astig = np.zeros((sphcoef.shape[0], 16))
out_astig[:, 5] = sphcoef
out_astig[:, 10] = sphcoef / -2
out_astig[:, 15] = sphcoef / 4
out_astig[:, 1] = xcoef
out_astig[:, 2] = ycoef

root = Path(__file__).parent / 'data' / 'coma-vs-angle'
p = root / 'queue.pkl'
q = PersistentQueue(p, overwrite=True)
items = list(out_coma)
q.put_many(items)


root = Path(__file__).parent / 'data' / 'astigmatism-vs-angle'
p = root / 'queue.pkl'
q = PersistentQueue(p, overwrite=True)
items = list(out_astig)
q.put_many(items)
