"""Launches a worker with the appropriate queue and database based on an index into a config file."""
import sys
from itertools import product
from pathlib import Path

import yaml

from iris.data import PersistentQueue, Database
from iris.worker import Worker
from iris.core import _mtf_cost_core_diffractiondiv, _mtf_cost_core_sumsquarediff, _mtf_cost_core_manhattan

# grab the config file path / name from the command line
cfgpath, cfgidx = sys.argv[1], int(sys.argv[2])
with open(cfgpath) as fid:
    cfg = yaml.load(fid)

# build a list of path names and chunk numbers
names, nums = cfg['names'], cfg['chunks']
nums = [str(num) for num in nums]
sets = list(product(names, nums))
target = sets[cfgidx]

# construct optimization directives
s = target[0]
if s[0] == 'd':
    diff = True
else:
    diff = False
if s.split('_')[-1] == 'manhattan':
    opt = _mtf_cost_core_manhattan
else:
    opt = _mtf_cost_core_sumsquarediff

if diff:
    optargs = (_mtf_cost_core_diffractiondiv, opt)
else:
    optargs = (opt)

# shield for multiprocessing
if __name__ == '__main__':
    root = Path(cfg['root']).expanduser()
    base = root / target[0] / target[1]
    q = PersistentQueue(base / 'queue.pkl')
    db = Database(base, fields=['truth_rmswfe', 'cost_first', 'cost_final', 'rrmswfe_final', 'time', 'nit'])
    w = Worker( q, db, optmode='global', optopts={'parallel': True, 'nthreads': 23}, optcoreeopts=optargs, work_time=60 * 7)
    w.start()
