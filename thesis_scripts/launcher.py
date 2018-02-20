"""Launches a worker with the appropriate queue and database based on an index into a config file."""
import sys
from itertools import product
from pathlib import Path

import yaml

from iris.data import PersistentQueue, Database
from iris.worker import Worker

cfgpath, cfgidx = sys.argv[1], sys.argv[2]
with open(cfgpath) as fid:
    cfg = yaml.load(fid)

names, nums = cfg['names'], cfg['chunks']
nums = [str(num) for num in nums]
sets = list(product(names, nums))
target = sets[cfgidx]

if __name__ == '__main__':
    p = Path(cfg['root'] / target[0] / target[1])
    q = PersistentQueue(p / 'queue.pkl')
    db = Database(p)
    w = Worker(q, db, optmode='global', optparallel=True, optnumthreads=23, work_time=60 * 7)
    w.start()
