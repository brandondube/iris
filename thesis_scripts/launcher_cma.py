"""Launches a worker with the appropriate queue and database based on an index into a config file."""
import sys
from pathlib import Path

import yaml

from iris.data import PersistentQueue, Database
from iris.rings import W3
from iris.worker import Worker

# grab the config file path / name from the command line
cfgpath, cfgidx = sys.argv[1], int(sys.argv[2])
with open(cfgpath) as fid:
    cfg = yaml.load(fid)

chunk = cfg['chunks'][cfgidx]

# shield for multiprocessing
if __name__ == '__main__':
    root = Path(cfg['root']).expanduser()
    base = root / str(chunk)
    q = PersistentQueue(base / 'queue.pkl')
    db = Database(base)#, fields=['truth_rmswfe', 'cost_first', 'cost_final', 'rrmswfe_final', 'time', 'nit'])
    w = Worker(q,
               db,
               optmode='global',
               simopts={
                   'guess': [0] * 18,
                   'decoder_ring': W3
                   },
               optopts={
                   'parallel': False,
                   'nthreads': 6,
                   'ftol': 1e-7,
               },
               work_time=60 * 17)
    w.start()
