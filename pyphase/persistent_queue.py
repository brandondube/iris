''' A queue object that persists to disk as a pickle with each operation.
'''
import pickle
from collections import deque


class PersistentQueue(deque):

    def __init__(self, write_path, write_time=None, write_on_action=True, maxsize=0):
        ''' Creates a new PersistentQueue, a Queue object which persists to disk
            at an interval determined by write_time or write_on_action.

        Args:
            write_path (`str`): where to persist the queue too.

            write_time (`number` or `None`): if is None, does not periodically
                write.  If a number, saves to disk asyncronously every `number`
                seconds.  Note that this mode of operation is not currently used.

            write_on_action (`bool`): whether to write on actions or not.  If
                this value is True, write_time will not be used.

            maxsize (`int`): maxsize, passed on to the Queue constructor.

        Returns:
            PersistentQueue: a new persistent queue.

        '''
        if write_on_action is True:
            self.write_on_action = True
            self.write_time = None
        elif write_time is not None:
            self.write_on_action = False
            self.write_time = write_time

        self.path = write_path

        super().__init__(maxsize)

    def persist(self):
        with open(self.path, mode='b', encoding='utf-8') as file:
            pickle.dump(self, file, protocol=-1)

    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        self.persist()

    def get(self, block=True, timeout=None)
