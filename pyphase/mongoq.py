''' A stdlib.Queue like JobQueue backed by a mongodb collection.
'''
from datetime import datetime
import time
import pymongo

class JobQueue(object):
    def __init__(self, db):
        ''' Creates a new JobQueue instance.  Mimics the Queue object from
            stdlib using a collection in a mongo db.

        Args:
            db (mongodb collection): a collection to use as a JobQueue.  If the
                collection does not exist, it will be created as a new JobQueue.

        Returns:
            JobQueue.  New JobQueue.

        '''
        self.db = db

        if not self._exists():
            print('Creating jobqueue collection.')
            self._create()
        self.q = self.db['jobqueue']

    def _create(self, capped=True):
        ''' Creates a Capped Collection.
        '''
        # TODO - does the size parameter mean number of docs or bytesize?
        try:
            self.db.create_collection('jobqueue', autoIndexId=True)
        except:
            raise Exception('Collection "jobqueue" already created')

    def _exists(self):
        ''' Ensures that the jobqueue collection exists in the DB.
        '''
        return 'jobqueue' in self.db.collection_names()

    def valid(self):
        ''' Checks to see if the jobqueue is a capped collection.
        '''
        opts = self.db['jobqueue'].options()
        if opts.get('capped', False):
            return True
        else:
            return False

    def next(self):
        ''' Runs the next job in the queue.
        '''
        # get a db cursor, finds the next job with a waiting status
        cursor = self.q.find({'status': 'waiting'})

        # get the next row - will raise if the q is empty.
        row = cursor.next()
        row['status'] = 'done'
        row['ts']['started'] = datetime.now()
        row['ts']['done'] = datetime.now()
        self.q.save(row)
        try:
            return row
        except:
            raise Exception('There are no jobs in the queue')

    def put(self, data):
        ''' Puts a doc into the work queue.
        '''
        doc = dict(
            ts={'created': datetime.now(),
                'started': datetime.now(),
                'done': datetime.now()},
            status='waiting',
            data=data)
        try:
            self.q.insert(doc, manipulate=False)
        except:
            raise Exception('could not add to queue')
        return True

    def get(self):
        ''' Gets the top doc from the q.  Mimics the stdlib.Queue api.
        '''
        return self.next()

    def __iter__(self):
        ''' Iterates through all docs in the queue
            and waits for new jobs when queue is empty.
        '''
        # may be bugged and need a while loop to force iteration of all of the
        # entries in the q
        cursor = self.q.find({'status': 'waiting'},
                             cursor_type=pymongo.CursorType.TAILABLE_AWAIT)
        for row in cursor:
            self.q.update({
                '_id': row['_id'],
                'status': 'waiting'}, {
                    '$set': {
                        'status': 'working',
                        'ts.started': datetime.now()}})

            yield row
            # forked from discogs/pymjq/jobqueue.py -- why does this appear
            # below the yield?
            row['status'] = 'done'
            row['ts']['done'] = datetime.now()
            self.q.save(row)

    def queue_count(self):
        ''' Returns the number of jobs waiting in the queue.
        '''
        cursor = self.q.find({'status': 'waiting'})
        if cursor:
            return cursor.count()

    def clear_queue(self):
        ''' Drops the queue collection.
        '''
        self.q.drop()
