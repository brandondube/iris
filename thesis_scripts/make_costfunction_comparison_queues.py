"""Creates four identical sets of five queues containing random truths for cost function comparison."""
from pathlib import Path
import numpy as np

from iris.data import PersistentQueue

# folders for each queue
root = Path(__file__).parent / 'data' / 'cost_function_comparison'
stem1 = root / 'diffraction_euclidian'
stem2 = root / '____________euclidian'
stem3 = root / 'diffraction_manhattan'
stem4 = root / '____________manhattan'
stems = (stem1, stem2, stem3, stem4)
qs = []
for stem in stems:
    counter, local = 1, []
    for i in range(5):
        p = stem / str(counter)
        local.append(PersistentQueue(p / 'queue.pkl'), overwrite=True)
        counter += 1
    qs.append(local)

# data
np.random.seed(1234)
fudge = np.random.normal(loc=1, scale=0.25, size=500)
z16_factor, z25_factor = -0.5, 0.5 ** 2
truths_z9 = np.random.random((500))
truths_z16, truths_z25 = truths_z9 * fudge * z16_factor, truths_z9 * fudge * z25_factor
truths_z4 = np.zeros(truths_z9.shape)
alltruth = np.stack((truths_z4, truths_z9, truths_z16, truths_z25), axis=1)
segments = np.split(alltruth, 5, axis=0)

# queues = [PersistentQueue(p) for p in q_paths]
for outer in qs:
    for q, data in zip(outer, segments):
        d = list(data)
        q.put_many(d)
