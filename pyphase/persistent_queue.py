''' A queue object that persists to disk as a pickle with each operation.
'''
import pickle
from collections import deque


class PersistentQueue(deque):

    def __init__(self, path):
        ''' Creates a new PersistentQueue, a deque-based object which functions
            like a FIFO queue and caches to disk on each call.  The queue is not
            thread safe and internally uses two deques to support safe failure
            upon consumption of a piece of the queue.

        Args:
            path (`str`): where to persist the queue too.

        Returns:
            PersistentQueue: a new persistent queue.

        Notes:
            This object is not thread-safe and will have unexpected behavior if
            shared between threads.

            The intended / expected behavior is as follows:
                - put() adds items to the end of the queue and persists it to
                    disk.

                - the user will use peek() to nondestructively retrieve the top
                    item on the queue.  When a task using that data is complete,
                    mark_done() will be called, which will persist the queue to
                    disk.

                - use of get() destructively returns the first item from the
                    queue and persists it to disk.  Use this where the consuming
                    code of that object will not fail.

        '''

        self.path = path
        self.q = deque()

    def persist(self):
        with open(self.path, mode='b', encoding='utf-8') as file:
            pickle.dump(self, file, protocol=-1)

    def put(self, item):
        self.q.append(item)
        self.persist()

    def get(self):
        return self.q.popleft()
        self.persist()

    def peek(self):
        item = self.q.popleft()
        self.q.appendleft(item)
        return item

    def mark_done(self):
        self.q.popleft()
        self.persist()
