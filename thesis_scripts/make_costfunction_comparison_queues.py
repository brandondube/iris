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
        p.mkdir(parents=True, exist_ok=True)
        local.append(PersistentQueue(p / 'queue.pkl', overwrite=True)
        counter += 1
    qs.append(local)

# set the seed and generate 'fudge factor' arrays centered on 1 with stdev 1/2 and 500 samples e/a
np.random.seed(1234)
fudge1 = np.random.normal(loc=1, scale=0.5, size=500)
fudge2 = np.random.normal(loc=1, scale=0.5, size=500)
z16_factor, z25_factor = -0.5, 0.5 ** 2

# no defocus
# uniform normal distributions of size 500 and range [0, 0.25]
truths_z4 = np.zeros(500)
truths_z9 = np.random.random(truths_z4.shape) * .25

# Z16 and Z25 are derived from Z9 and have value -1/2 and +1/4 respectively
# but contain gaussian noise that makes this relationship fuzzy
truths_z16, truths_z25 = truths_z9 * fudge1 * z16_factor, truths_z9 * fudge2 * z25_factor

# stack all truths together and split them into 5 chunks for 5 workers per datatype
alltruth = np.stack((truths_z4, truths_z9, truths_z16, truths_z25), axis=1)
segments = np.split(alltruth, 5, axis=0)

for outer in qs:
    for q, data in zip(outer, segments):
        d = list(data)
        q.put_many(d)
