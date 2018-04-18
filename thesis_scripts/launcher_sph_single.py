"""Launches a worker with the appropriate queue and database based on an index into a config file."""
from pathlib import Path


from iris.data import PersistentQueue, Database
from iris.rings import W1
from iris.worker import Worker

# shield for multiprocessing
if __name__ == '__main__':
    root = Path(__file__).parent / 'data' / 'rotationally-invariant-case'
    q = PersistentQueue(root / 'queue.pkl')
    db = Database(root, fields=['truth_rmswfe', 'cost_first', 'cost_final', 'rrmswfe_final', 'time', 'nit'])
    w = Worker(q,
               db,
               optmode='global',
               simopts={
                   'guess': [0] * 18,
                   'decoder_ring': W1,
                   },
               optopts={
                   'parallel': True,
                   'nthreads': 7,
                   'ftol': 1e-7,
               },
               work_time=60 * 17)
    w.start()
