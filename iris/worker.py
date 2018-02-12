"""A worker that works for a period of time or number of jobs."""
import time

from iris.macros import run_azimuthalzero_simulation


class Worker(object):
    """A worker."""

    def __init__(self, queue, database, work_time=None, work_jobs=None):
        """Create a new worker.

        Parameters
        ----------
        queue : `iris.data.PersistentQueue`
            a persistent queue object
        database : `iris.
        work_time : numeric, optional
            time to work for, minutes
        work_jobs : `int`, optional
            number of jobs to complete

        Raises
        ------
        ValueError
        if work_time and work_jobs are both None

        """
        if work_time is None and work_jobs is None:
            raise ValueError('work time and work jobs cannot both be None')

        self.mode = None
        if work_time is not None:
            self.start_time = time.monotonic()
            self.end_time = self.start_time + (60 * work_time)
            self.mode = 'time'
        else:
            self.start_job = 0
            self.end_job = work_jobs
            self.current_job = 0
            self.mode = 'jobs'

        self.q = queue
        self.db = database
        self.work_time = work_time
        self.work_jobs = work_jobs
        self.status = 'stopped'

    def do_job(self):
        """Do a job."""
        try:
            item = self.q.peek()
        except IndexError:
            print('stopping - queue exhausted')
            self.end()
            return

        result = run_azimuthalzero_simulation(truth=item)
        self.db.append(result)
        self.q.mark_done()

    def start(self):
        """Begin working and block."""
        self.status = 'working'
        try:
            while self.status == 'working':
                self.do_job()
                if self.mode == 'jobs':
                    self.current_job += 1
                    if self.current_job >= self.end_job:
                        self.status = 'stopped'
                else:
                    now = time.monotonic()
                    if now > self.end_time:
                        self.status = 'stopped'
        except KeyboardInterrupt:
            print('stopping - user requested')
            self.end()

    def end(self):
        """End working and clean up."""
        self.status = 'stopped'
        if self.mode == 'jobs':
            self.current_job = 0
